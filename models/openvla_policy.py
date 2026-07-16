"""
models/openvla_policy.py — OpenVLA-7B as a drone policy (transfer arm) and a
from-scratch control arm, for the cross-embodiment ablation.

  build_openvla_policy(init="pretrained")  -> robot-pretrained OpenVLA + LoRA
  build_openvla_policy(init="scratch")     -> same architecture, random init,
                                              fully trainable (no pretraining)
  build_prismatic_policy()                 -> Prismatic VLM base (VL-pretrained,
                                              robot-naive) + the SAME LoRA recipe

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
build_prismatic_policy() below implements the gold-standard control: it shares
OpenVLA's architecture AND its vision-language pretraining, differing only in
having no robot/action pretraining — so OpenVLA > Prismatic isolates the *robot*
transfer, while OpenVLA ~= Prismatic would mean the gain is generic VL features.

THIS MODULE REQUIRES THE OPENVLA ENV (transformers==4.40.1; see
requirements-openvla.txt) AND A >=16GB GPU. It cannot run in the main env.
"""

import torch
import torch.nn as nn

OPENVLA_ID = "openvla/openvla-7b"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Prismatic VLM base: same DINOv2+SigLIP vision + Llama-2-7B as OpenVLA, with the
# SAME vision-language pretraining, but NEVER robot/action-trained — the clean
# control that isolates *robot* pretraining. VERIFY the exact id against the
# TRI-ML/prismatic-vlms model registry before a run.
PRISMATIC_ID = "prism-dinosiglip-224px+7b"


def _bnb_config(compute_dtype=torch.bfloat16):
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )


def build_openvla_policy(
    init: str = "pretrained",          # "pretrained" | "scratch"
    load_in_4bit: bool = True,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.0,
    compute_dtype=torch.bfloat16,      # torch.float16 on Turing (T4): no bf16 HW
):
    """
    Returns (model, processor). The model exposes the native OpenVLA forward
    (returns .loss when `labels` are passed) and .predict_action for eval.

    compute_dtype selects the activation/weight dtype. Ampere+ (A100, 30xx/40xx,
    lab GPU) supports torch.bfloat16 — the default. Turing GPUs (Colab/Kaggle T4)
    and Pascal (P100) have no bf16 tensor cores, so pass torch.float16 there.
    """
    from transformers import AutoModelForVision2Seq, AutoProcessor, AutoConfig

    processor = AutoProcessor.from_pretrained(OPENVLA_ID, trust_remote_code=True)
    bnb = _bnb_config(compute_dtype) if load_in_4bit else None

    if init == "pretrained":
        model = AutoModelForVision2Seq.from_pretrained(
            OPENVLA_ID,
            quantization_config=bnb,
            # A 4-bit model must be dispatched to the GPU at load time; without a
            # device_map the weights stay on CPU, bnb's Linear4bit can't be moved
            # with .to(), and the train loop's batch.to("cuda") then mismatches.
            # Pin to a single GPU ({"": 0}) rather than "auto" so a 7B model is
            # never split across CPU/GPU (which breaks the LoRA/checkpoint path).
            # This dispatch is why accelerate is PINNED (requirements-openvla.txt):
            # accelerate >=0.33 with bnb >=0.43.2 dispatches a single-device 4-bit
            # model via .to(), which transformers 4.40.1 refuses -> load fails here.
            device_map={"": 0} if bnb is not None else None,
            torch_dtype=compute_dtype,
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
        model = model.to(compute_dtype)
        model = _apply_lora(model, lora_rank, lora_alpha, lora_dropout, quantized=False)
        print(f"[OpenVLA] random init + LoRA(r={lora_rank}) — from-scratch control arm")

    else:
        raise ValueError(f"init must be 'pretrained' or 'scratch', got {init!r}")

    return model, processor


def build_prismatic_policy(
    lora_rank: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.0,
):
    """
    Control arm: the Prismatic VLM base `prism-dinosiglip-224px+7b` — vision-
    language pretrained but NEVER robot/action-trained — finetuned with the SAME
    action-token LoRA recipe as the OpenVLA transfer arm. This isolates *robot*
    pretraining: if OpenVLA+LoRA beats this, the gain is robot-specific transfer,
    not generic vision-language features.

    Returns (model, tokenizer, image_transform). The native forward returns .loss
    when `labels` are passed. Prismatic's LLM is Llama-2-7B, so the 256 action
    token ids and models.action_tokenizer.ActionTokenizer carry over UNCHANGED —
    only the pixel format differs (dinosiglip emits a {"dino","siglip"} dict, not
    a stacked 6-channel tensor; see data.openvla_dataset.PrismaticDroneDataset).

    REQUIRES the prismatic-vlms package (pip install
    git+https://github.com/TRI-ML/prismatic-vlms) and a >=24GB GPU (bf16 full
    backbone; no 4-bit on the custom wrapper here). UNVALIDATED on this machine —
    written against the Prismatic API; verify on the lab GPU. In particular,
    confirm Prismatic's forward expects `labels` aligned to the TEXT stream and
    masks the inserted image tokens internally (as OpenVLA's HF path does).
    """
    from prismatic import load
    from .lora import inject_lora, count_trainable_params

    vlm = load(PRISMATIC_ID).to(torch.bfloat16)

    # Parity with the OpenVLA arm: freeze EVERYTHING (vision backbone, projector,
    # LM head), then inject LoRA into the LLM attention only. inject_lora itself
    # freezes the llm base and unfreezes only the LoRA A/B matrices, but we must
    # first freeze the vision backbone + projector (separate submodules) too.
    for p in vlm.parameters():
        p.requires_grad_(False)
    inject_lora(
        vlm.llm_backbone.llm,
        target_modules=LORA_TARGETS,
        rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout,
    )

    # Gradient-checkpoint through the frozen ~14GB bf16 base; input_require_grads
    # is required so gradients reach the LoRA adapters through a frozen base.
    llm = vlm.llm_backbone.llm
    if hasattr(llm, "gradient_checkpointing_enable"):
        llm.gradient_checkpointing_enable()
    if hasattr(llm, "enable_input_require_grads"):
        llm.enable_input_require_grads()

    s = count_trainable_params(vlm)
    print(f"[Prismatic] {PRISMATIC_ID} + LoRA(r={lora_rank}) — "
          f"VL-pretrained / robot-naive control arm")
    print(f"[Prismatic] trainable {s['trainable']:,} / {s['total']:,} "
          f"({s['pct_trainable']:.2f}%)")

    tokenizer = vlm.llm_backbone.get_tokenizer()
    image_transform = vlm.vision_backbone.get_image_transform()
    return vlm, tokenizer, image_transform


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
