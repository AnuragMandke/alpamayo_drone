"""
models/openvla_policy.py — OpenVLA-7B as a drone policy (transfer arm) and a
from-scratch control arm, for the cross-embodiment ablation.

  build_openvla_policy(init="pretrained")  -> robot-pretrained OpenVLA + LoRA
  build_openvla_policy(init="scratch")     -> same architecture, random init,
                                              fully trainable (no pretraining)

The model is OpenVLA's native AutoModelForVision2Seq: it consumes
(input_ids, attention_mask, pixel_values, labels) and returns a cross-entropy
loss over the action-token positions directly — we reuse its action head, so
training is just `model(**batch).loss`.

ABLATION NOTE — the cleanest control that isolates the *robot* pretraining
specifically is the Prismatic VLM base (vision-language pretrained, never
robot-trained) + the same LoRA finetuning. That requires OpenVLA's training
repo (TRI-ML/prismatic-vlms), not plain HF AutoModel. The "scratch" arm here
(random-init, full fine-tune) is a weaker but fully self-contained control:
pretrained+LoRA vs from-scratch tests whether the pretraining matters at all.
Add the Prismatic-base control on the lab GPU as the gold-standard follow-up.

THIS MODULE REQUIRES THE OPENVLA ENV (transformers==4.40.1; see
requirements-openvla.txt) AND A >=16GB GPU. It cannot run in the main env.
"""

import torch
import torch.nn as nn

OPENVLA_ID = "openvla/openvla-7b"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


def _bnb_config():
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def build_openvla_policy(
    init: str = "pretrained",          # "pretrained" | "scratch"
    load_in_4bit: bool = True,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.0,
):
    """
    Returns (model, processor). The model exposes the native OpenVLA forward
    (returns .loss when `labels` are passed) and .predict_action for eval.
    """
    from transformers import AutoModelForVision2Seq, AutoProcessor, AutoConfig

    processor = AutoProcessor.from_pretrained(OPENVLA_ID, trust_remote_code=True)
    bnb = _bnb_config() if load_in_4bit else None

    if init == "pretrained":
        model = AutoModelForVision2Seq.from_pretrained(
            OPENVLA_ID,
            quantization_config=bnb,
            # A 4-bit model must be dispatched to the GPU at load time; without a
            # device_map the weights stay on CPU, bnb's Linear4bit can't be moved
            # with .to(), and the train loop's batch.to("cuda") then mismatches.
            # Pin to a single GPU ({"": 0}) rather than "auto" so a 7B model is
            # never split across CPU/GPU (which breaks the LoRA/checkpoint path).
            device_map={"": 0} if bnb is not None else None,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model = _apply_lora(model, lora_rank, lora_alpha, lora_dropout, quantized=bnb is not None)
        print(f"[OpenVLA] pretrained + LoRA(r={lora_rank}) — transfer arm")

    elif init == "scratch":
        # Same architecture and SAME LoRA setup as the transfer arm, but weights
        # are RANDOMLY initialized instead of robot-pretrained — so the arms
        # differ only in initialization, which is the clean ablation. We freeze
        # the random base and train only LoRA: full fine-tuning a 7B from scratch
        # needs ~70GB of AdamW state (OOMs the lab GPU) and can't converge on
        # ~12k samples anyway. This is the self-contained "no-pretraining" lower
        # bound; the Prismatic VLM base is the stronger gold-standard control.
        config = AutoConfig.from_pretrained(OPENVLA_ID, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_config(config, trust_remote_code=True)
        model = model.to(torch.bfloat16)
        model = _apply_lora(model, lora_rank, lora_alpha, lora_dropout, quantized=False)
        print(f"[OpenVLA] random init + LoRA(r={lora_rank}) — from-scratch control arm")

    else:
        raise ValueError(f"init must be 'pretrained' or 'scratch', got {init!r}")

    return model, processor


def _apply_lora(model, rank, alpha, dropout, quantized):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if quantized:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )
    else:
        # Non-quantized base (scratch control): get_peft_model freezes the base
        # and trains only LoRA, but we still checkpoint to keep the frozen 14GB
        # bf16 backbone's activations off the GPU. enable_input_require_grads is
        # required so gradients flow to the LoRA adapters through a frozen base.
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=LORA_TARGETS,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]
