"""
eval/openvla_evaluator.py — offline scorer for the OpenVLA-family arms
(pretrained / scratch / prismatic), the metric the cross-embodiment ablation is
compared on.

Teacher-forced: one forward per batch, argmax at the 7 action-token positions.
Reports, on the val split:
    - action_token_accuracy : top-1 over the 4 REAL drone dims (quantization-free).
                              Scored on dims [0,1,2,5] only: drone_to_openvla
                              writes constant zeros into roll/pitch/gripper, so
                              including those 3 free tokens puts every arm just
                              above 3/7=0.4286 and roughly halves the gap between
                              arms. action_token_accuracy_all7 keeps the diluted
                              number for reference.
    - action_l2             : mean per-sample L2 of the 4-DoF drone action error
    - per_dim_mae           : MAE for [vx, vy, vz, yaw_rate], physical units
    - val_loss              : teacher-forced loss on held-out data, directly
                              comparable to the marginal floor the trainer prints

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
    Returns (n_drone_correct:int, n_all_correct:int, abs_err:(4,) float,
    l2:float) where the errors are in physical drone units.

    n_drone_correct counts ONLY dims [0,1,2,5]. The other three are held at a
    constant neutral bin, so counting them rewards every arm equally for free
    and shrinks the between-arm gap the ablation exists to measure.
    """
    n_correct = int((p_ids[DRONE_TO_OPENVLA_IDX] == g_ids[DRONE_TO_OPENVLA_IDX]).sum())
    n_correct_all = int((p_ids == g_ids).sum())
    p_norm7 = action_tokenizer.decode_token_ids_to_actions(p_ids)   # (7,) in [-1,1]
    g_norm7 = action_tokenizer.decode_token_ids_to_actions(g_ids)
    p_phys = denormalize_action(p_norm7[DRONE_TO_OPENVLA_IDX], norm_stats)  # (4,)
    g_phys = denormalize_action(g_norm7[DRONE_TO_OPENVLA_IDX], norm_stats)
    err = np.abs(p_phys - g_phys).astype(np.float64)
    return n_correct, n_correct_all, err, float(np.linalg.norm(p_phys - g_phys))


@torch.no_grad()
def evaluate_openvla(model, loader, action_tokenizer, norm_stats, device,
                     amp_dtype=torch.bfloat16, amp_enabled=True, n_action_tokens=7):
    """Teacher-forced offline evaluation over `loader`. Returns a metrics dict.

    amp_dtype/amp_enabled come from the run's config via resolve_amp: bf16 on the
    lab GPU, fp16 on a T4 (Turing has NO bf16 tensor cores — casting pixels or
    autocasting to bf16 there is emulated and can error), fp32 to debug. Must
    match the dtype the model was built/trained in, or the eval mis-scores."""
    model.eval()
    n_dims = len(DRONE_TO_OPENVLA_IDX)
    tok_correct = tok_total = 0
    tok_correct_all = tok_total_all = 0
    abs_err_sum = np.zeros(n_dims, dtype=np.float64)
    l2_sum = 0.0
    loss_sum = 0.0
    n_samples = 0
    offset_reported = False

    for batch in loader:
        batch = _move(batch, device, amp_dtype)
        with torch.autocast("cuda", dtype=amp_dtype,
                            enabled=amp_enabled and device.type == "cuda"):
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                labels=batch["labels"],
            )
        logits = out.logits.float()              # (B, T_out, V)
        labels = batch["labels"]                 # (B, T_txt)

        # Prismatic splices its image-patch embeddings INTO the sequence, so its
        # logits run T_txt + n_patches (286 vs 30 here) while labels stay on the
        # text stream. The HF OpenVLA path returns logits already aligned to
        # input_ids, i.e. offset 0, which is why only the prismatic arm was hit.
        # Without this correction the shift below reads predictions out of the
        # image-patch region: it scored that arm at exactly 0.0000 token accuracy
        # — below the ~1/256 chance rate — while it was training to a healthy
        # loss. Patches sit ahead of every supervised position, so dropping the
        # leading `offset` predictions re-aligns the two tails.
        offset = logits.shape[1] - labels.shape[1]
        if offset < 0:
            raise ValueError(
                f"logits ({logits.shape[1]}) shorter than labels "
                f"({labels.shape[1]}); the scorer cannot align them")
        if offset and not offset_reported:
            print(f"[Eval] logits are {offset} longer than labels "
                  f"(spliced image patches) — realigning before scoring")
            offset_reported = True

        # Causal shift: logits[:, i] predicts the token at labels[:, i+1].
        pred = logits[:, :-1].argmax(-1)[:, offset:]   # (B, T_txt-1)
        gold = labels[:, 1:]                           # (B, T_txt-1)
        if pred.shape[1] != gold.shape[1]:
            raise ValueError(
                f"alignment failed: pred {pred.shape[1]} vs gold {gold.shape[1]}")
        mask = gold != -100                      # supervised = 7 action tokens + EOS

        # Teacher-forced loss on the val split. Alignment-independent (the model
        # computes it internally), directly comparable to the marginal floor the
        # trainer prints, and measured on held-out data — so it is the number
        # that separates learning from memorization.
        if getattr(out, "loss", None) is not None:
            loss_sum += float(out.loss) * gold.shape[0]

        for b in range(gold.shape[0]):
            pos = mask[b].nonzero(as_tuple=True)[0]      # contiguous: a0..a6, eos
            act_pos = pos[:n_action_tokens]              # first 7 = action tokens
            p_ids = pred[b, act_pos].cpu().numpy()
            g_ids = gold[b, act_pos].cpu().numpy()

            n_corr, n_corr_all, err, l2 = _score_tokens(
                p_ids, g_ids, action_tokenizer, norm_stats)
            tok_correct += n_corr
            tok_total += n_dims
            tok_correct_all += n_corr_all
            tok_total_all += n_action_tokens
            abs_err_sum += err
            l2_sum += l2
            n_samples += 1

    n = max(n_samples, 1)
    return {
        "action_token_accuracy": tok_correct / max(tok_total, 1),
        "action_token_accuracy_all7": tok_correct_all / max(tok_total_all, 1),
        "action_l2": l2_sum / n,
        "val_loss": loss_sum / n,
        "per_dim_mae": {
            "vx": abs_err_sum[0] / n,
            "vy": abs_err_sum[1] / n,
            "vz": abs_err_sum[2] / n,
            "yaw_rate": abs_err_sum[3] / n,
        },
        "n_samples": n_samples,
    }
