"""
evaluator.py — Offline evaluation on held-out AirSim trajectories

Metrics computed:
    - Task Success Rate (TSR): fraction of episodes reaching goal within threshold
    - L2 Action Error: mean L2 between predicted and GT actions
    - Displacement Error (ADE/FDE): trajectory-level prediction quality

For online AirSim evaluation (live simulator), see airsim_env.py.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional
from torch.utils.data import DataLoader


class OfflineEvaluator:
    """
    Evaluates Alpamayo-Drone on the val split without running AirSim.

    This is what you'll use for iterative model comparison during
    ablation studies — no simulator required.
    """

    def __init__(
        self,
        model,
        val_loader: DataLoader,
        device: torch.device,
        action_horizon: int = 4,
    ):
        self.model = model
        self.val_loader = val_loader
        self.device = device
        self.action_horizon = action_horizon

    @torch.no_grad()
    def evaluate(self) -> dict:
        self.model.eval()

        l2_errors = []
        ade_errors = []    # Average Displacement Error (mean over horizon)
        fde_errors = []    # Final Displacement Error (last step only)

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            actions_gt = batch["actions"].to(self.device)   # (B, H, 4)
            images = batch["images"].to(self.device)

            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images,
                actions_gt=None,    # inference mode
            )
            actions_pred = out["actions"]                    # (B, H, 4)

            # L2 per sample per step → mean
            diff = actions_pred - actions_gt                 # (B, H, 4)
            l2 = diff.norm(dim=-1)                           # (B, H)
            l2_errors.append(l2.mean().item())

            # ADE: mean over horizon
            ade_errors.append(l2.mean(dim=1).mean().item())

            # FDE: last step
            fde_errors.append(l2[:, -1].mean().item())

        return {
            "l2_action_error": np.mean(l2_errors),
            "ade":             np.mean(ade_errors),
            "fde":             np.mean(fde_errors),
        }


class AirSimOnlineEvaluator:
    """
    Online evaluator that connects to a running AirSim instance.

    Requires:
        pip install airsim
        AirSim simulator running at cfg['eval']['airsim_ip']:port
    """

    def __init__(self, model, cfg: dict, device: torch.device, tokenizer):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.tokenizer = tokenizer
        self.ec = cfg["eval"]

    def _connect(self):
        try:
            import airsim
        except ImportError:
            raise ImportError(
                "airsim package not installed. "
                "Run: pip install airsim"
            )
        client = airsim.MultirotorClient(
            ip=self.ec["airsim_ip"],
            port=self.ec["airsim_port"],
        )
        client.confirmConnection()
        client.enableApiControl(True)
        client.armDisarm(True)
        return client, airsim

    @torch.no_grad()
    def evaluate_task(self, task: str) -> dict:
        client, airsim = self._connect()
        n_episodes = self.ec["n_episodes"]
        successes = 0
        step_counts = []

        for ep in range(n_episodes):
            client.reset()
            client.enableApiControl(True)
            client.armDisarm(True)
            client.takeoffAsync().join()

            goal = self._sample_goal(task, airsim)
            success, steps = self._run_episode(client, airsim, goal, task)
            if success:
                successes += 1
            step_counts.append(steps)

            print(f"  Episode {ep+1}/{n_episodes}: "
                  f"{'✓' if success else '✗'}  steps={steps}")

        tsr = successes / n_episodes
        print(f"[Eval] Task={task}  TSR={tsr:.2%}  "
              f"avg_steps={np.mean(step_counts):.1f}")
        return {"task": task, "tsr": tsr, "avg_steps": float(np.mean(step_counts))}

    def _sample_goal(self, task: str, airsim):
        """Return a goal position for the given task type."""
        import random
        if task == "point_nav":
            return airsim.Vector3r(
                random.uniform(20, 50),
                random.uniform(-20, 20),
                -10,   # AirSim uses NED: negative Z = up
            )
        elif task == "hover_stabilize":
            # Return current position as goal (hold)
            pose = self._get_client_pose()
            return pose.position
        else:
            return airsim.Vector3r(30, 0, -10)

    def _get_obs_tokens(self, client, instruction: str):
        """Capture image from AirSim and tokenize with instruction."""
        import airsim
        import numpy as np
        from PIL import Image
        from data.airsim_dataset import IMAGE_TRANSFORM

        responses = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
        ])
        img_raw = responses[0]
        img_1d = np.frombuffer(img_raw.image_data_uint8, dtype=np.uint8)
        img = img_1d.reshape(img_raw.height, img_raw.width, 3)
        img_pil = Image.fromarray(img)
        # (1, S=1, 3, 224, 224): a single-frame history for online rollout.
        img_tensor = (
            IMAGE_TRANSFORM(img_pil).unsqueeze(0).unsqueeze(0).to(self.device)
        )

        prompt = f"Instruction: {instruction}\nAction:"
        enc = self.tokenizer(
            prompt, return_tensors="pt",
            max_length=512, padding="max_length", truncation=True,
        )
        return (
            enc["input_ids"].to(self.device),
            enc["attention_mask"].to(self.device),
            img_tensor,
        )

    def _run_episode(self, client, airsim, goal, task: str):
        max_steps = self.ec["max_episode_steps"]
        instruction = self._task_instruction(task, goal)

        for step in range(max_steps):
            input_ids, attn_mask, images = self._get_obs_tokens(client, instruction)
            out = self.model(
                input_ids=input_ids, attention_mask=attn_mask, images=images
            )
            action = out["actions"][0, 0].cpu().numpy()   # take first predicted step

            # Send velocity command: [vx, vy, vz, yaw_rate]
            client.moveByVelocityAsync(
                float(action[0]), float(action[1]), float(action[2]),
                duration=0.1,
            ).join()
            client.rotateByYawRateAsync(float(action[3]), duration=0.1).join()

            # Check success
            if self._check_success(client, airsim, goal, task):
                return True, step + 1

        return False, max_steps

    def _check_success(self, client, airsim, goal, task: str) -> bool:
        pos = client.getMultirotorState().kinematics_estimated.position
        dist = np.sqrt(
            (pos.x_val - goal.x_val) ** 2 +
            (pos.y_val - goal.y_val) ** 2 +
            (pos.z_val - goal.z_val) ** 2
        )
        threshold = {
            "point_nav": self.ec["point_nav_success_dist_m"],
            "hover_stabilize": self.ec["hover_success_drift_m"],
            "obstacle_avoidance": self.ec["point_nav_success_dist_m"],
        }.get(task, 2.0)
        return dist < threshold

    def _task_instruction(self, task: str, goal) -> str:
        if task == "point_nav":
            return (f"Fly to position x={goal.x_val:.1f} "
                    f"y={goal.y_val:.1f} z={goal.z_val:.1f}.")
        elif task == "hover_stabilize":
            return "Hold your current position and altitude."
        else:
            return "Navigate to the goal while avoiding obstacles."

    def evaluate_all(self) -> list[dict]:
        results = []
        for task in self.ec["tasks"]:
            r = self.evaluate_task(task)
            results.append(r)
        return results
