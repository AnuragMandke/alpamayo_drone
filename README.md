# Alpamayo-Drone: Cross-Embodiment VLA Finetuning for UAV Navigation

Finetunes the Alpamayo-R1 VLA architecture on UAV navigation tasks via LoRA,
with a FlowMatchingDecoder as the domain-agnostic policy head.

## Project Structure

```
alpamayo_drone/
├── configs/
│   └── default.yaml          # All hyperparameters in one place
├── data/
│   ├── airsim_dataset.py     # AirSim trajectory dataset
│   └── transforms.py         # Observation preprocessing
├── models/
│   ├── vit_encoder.py        # Visual encoder (ViT-B/16)
│   ├── backbone.py           # Qwen2.5-style LLM backbone with GQA + RoPE
│   ├── flow_matching.py      # FlowMatchingDecoder (action head)
│   ├── alpamayo.py           # Full model assembly
│   └── lora.py               # LoRA injection utilities
├── training/
│   ├── trainer.py            # Training loop
│   ├── losses.py             # Flow matching loss
│   └── scheduler.py          # LR scheduler
├── eval/
│   ├── evaluator.py          # Task success rate + metrics
│   └── airsim_env.py         # AirSim environment wrapper
├── scripts/
│   ├── train.py              # Entry point: training
│   ├── eval.py               # Entry point: evaluation
│   └── download_data.py      # AirSim dataset download helper
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
python scripts/download_data.py          # downloads AirSim trajectories
python scripts/train.py --config configs/default.yaml
python scripts/eval.py  --config configs/default.yaml --ckpt outputs/best.pt
```

## Key Design Decisions

- **Base model:** Qwen2.5-3B-Instruct as Alpamayo-R1 proxy (Alpamayo-R1 weights
  are not public; vision is provided by a separate frozen ViT, so a text-only
  backbone suffices)
- **Finetuning:** LoRA injected into backbone attention (rank 16, α 32)
- **Action head:** FlowMatchingDecoder — 8-step DDIM, action dim = 4 (vx, vy, vz, yaw_rate)
- **Frozen:** ViT encoder + first 12 backbone layers; only LoRA + vision projection + decoder trained
- **Hardware target:** 6GB VRAM (RTX 4050 laptop) with 4-bit quantized base + bf16 autocast
