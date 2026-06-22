"""
eval/openvla_evaluator.py — offline scorer for the OpenVLA-family arms
(pretrained / scratch / prismatic), the metric the cross-embodiment ablation is
compared on.

Teacher-forced: one forward per batch, argmax at the 7 action-token positions.
Reports, on the val split:
    - action_token_accuracy : top-1 over the 7 action tokens (quantization-free)
    - action_l2             : mean per-sample L2 of the 4-DoF drone action error
    - per_dim_mae           : MAE for [vx, vy, vz, yaw_rate], physical units

Both predicted and gold tokens are decoded through the DRONE q01/q99 stats
(data.openvla_dataset.denormalize_action), NOT OpenVLA's built-in predict_action
denorm — that uses the Open-X dataset statistics and would report on the wrong
scale. Scoring pred-vs-gold-decoded (rather than pred-vs-raw) makes the
quantization floor identical across arms, so it cancels in any comparison.
"""

import numpy as np
import torch

from models.action_tokenizer import DRONE_TO_OPENVLA_IDX
from data.openvla_dataset import denormalize_action


def _move(batch, device, pixel_dtype):
    """Move a batch to device; cast pixel_values (tensor or dinosiglip dict)."""
    out = {}
    for k, v in batch.items():
        if k == "pixel_values":
            out[k] = ({kk: vv.to(device, pixel_dtype) for kk, vv in v.items()}
                      if isinstance(v, dict) else v.to(device, pixel_dtype))
        else:
            out[k] = v.to(device)
    return out


def _score_tokens(p_ids, g_ids, action_tokenizer, norm_stats):
    """Score one sample's 7 action tokens.

    p_ids, g_ids: (7,) predicted / gold action-token ids (numpy int).
    Returns (n_token_correct:int, abs_err:(4,) float, l2:float) where the errors
    are in physical drone units [vx, vy, vz, yaw_rate].
    """
    n_correct = int((p_ids == g_ids).sum())
    p_norm7 = action_tokenizer.decode_token_ids_to_actions(p_ids)   # (7,) in [-1,1]
    g_norm7 = action_tokenizer.decode_token_ids_to_actions(g_ids)
    p_phys = denormalize_action(p_norm7[DRONE_TO_OPENVLA_IDX], norm_stats)  # (4,)
    g_phys = denormalize_action(g_norm7[DRONE_TO_OPENVLA_IDX], norm_stats)
    err = np.abs(p_phys - g_phys).astype(np.float64)
    return n_correct, err, float(np.linalg.norm(p_phys - g_phys))


@torch.no_grad()
def evaluate_openvla(model, loader, action_tokenizer, norm_stats, device,
                     bf16=True, n_action_tokens=7):
    """Teacher-forced offline evaluation over `loader`. Returns a metrics dict."""
    model.eval()
    n_dims = len(DRONE_TO_OPENVLA_IDX)
    tok_correct = tok_total = 0
    abs_err_sum = np.zeros(n_dims, dtype=np.float64)
    l2_sum = 0.0
    n_samples = 0

    for batch in loader:
        batch = _move(batch, device, torch.bfloat16)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=bf16 and device.type == "cuda"):
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                labels=batch["labels"],
            )
        logits = out.logits.float()              # (B, T, V)
        labels = batch["labels"]                 # (B, T)

        # Causal shift: logits[:, i] predicts the token at labels[:, i+1].
        pred = logits[:, :-1].argmax(-1)         # (B, T-1)
        gold = labels[:, 1:]                     # (B, T-1)
        mask = gold != -100                      # supervised = 7 action tokens + EOS

        for b in range(gold.shape[0]):
            pos = mask[b].nonzero(as_tuple=True)[0]      # contiguous: a0..a6, eos
            act_pos = pos[:n_action_tokens]              # first 7 = action tokens
            p_ids = pred[b, act_pos].cpu().numpy()
            g_ids = gold[b, act_pos].cpu().numpy()

            n_corr, err, l2 = _score_tokens(p_ids, g_ids, action_tokenizer, norm_stats)
            tok_correct += n_corr
            tok_total += n_action_tokens
            abs_err_sum += err
            l2_sum += l2
            n_samples += 1

    n = max(n_samples, 1)
    return {
        "action_token_accuracy": tok_correct / max(tok_total, 1),
        "action_l2": l2_sum / n,
        "per_dim_mae": {
            "vx": abs_err_sum[0] / n,
            "vy": abs_err_sum[1] / n,
            "vz": abs_err_sum[2] / n,
            "yaw_rate": abs_err_sum[3] / n,
        },
        "n_samples": n_samples,
    }
