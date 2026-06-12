"""
alpamayo.py — Full Alpamayo-Drone model

Architecture:
    ViT encoder (frozen)  →  patch tokens
    OpenVLA backbone      →  context embedding   (LoRA finetuned)
    FlowMatchingDecoder   →  UAV action sequence  (fully trained)

The backbone's causal LM head is discarded; instead the last-layer
hidden states are passed as memory to the FlowMatchingDecoder.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from .flow_matching import FlowMatchingDecoder
from .lora import inject_lora, count_trainable_params


class AlpamayoDrone(nn.Module):
    """
    End-to-end drone VLA model.

    forward() returns a dict with:
        'loss'    — scalar CFM loss  (training only; None at inference)
        'actions' — (B, action_horizon, action_dim)  (inference only)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        mc = cfg["model"]
        fm = mc["flow_matching"]

        # ------------------------------------------------------------------
        # 1.  Load backbone (OpenVLA-7B as Alpamayo-R1 proxy)
        # ------------------------------------------------------------------
        print(f"[Model] Loading backbone: {mc['base_model_id']}")

        bnb_config = None
        if mc.get("load_in_4bit", False):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        self.backbone = AutoModel.from_pretrained(
            mc["base_model_id"],
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            mc["base_model_id"], trust_remote_code=True
        )

        # Freeze bottom N layers
        n_freeze = mc.get("freeze_first_n_layers", 12)
        self._freeze_backbone_layers(n_freeze)

        # ------------------------------------------------------------------
        # 2.  Inject LoRA into backbone attention
        # ------------------------------------------------------------------
        lora_cfg = mc["lora"]
        inject_lora(
            self.backbone,
            target_modules=lora_cfg["target_modules"],
            rank=lora_cfg["rank"],
            alpha=lora_cfg["alpha"],
            dropout=lora_cfg["dropout"],
        )

        stats = count_trainable_params(self.backbone)
        print(
            f"[Model] Backbone trainable params: "
            f"{stats['trainable']:,} / {stats['total']:,} "
            f"({stats['pct_trainable']:.2f}%)"
        )

        # ------------------------------------------------------------------
        # 3.  FlowMatchingDecoder (fully trainable)
        # ------------------------------------------------------------------
        # Infer context_dim from backbone hidden size
        context_dim = self.backbone.config.hidden_size

        self.decoder = FlowMatchingDecoder(
            context_dim=context_dim,
            action_dim=fm["action_dim"],
            hidden_dim=fm["hidden_dim"],
            num_layers=fm["num_layers"],
            num_heads=fm["num_heads"],
            num_diffusion_steps=fm["num_diffusion_steps"],
            num_train_steps=fm["num_train_steps"],
            sigma_min=fm["sigma_min"],
        )

    # ------------------------------------------------------------------

    def _freeze_backbone_layers(self, n: int):
        """Freeze embeddings + first n transformer layers of the backbone."""
        # Freeze embeddings
        if hasattr(self.backbone, "embed_tokens"):
            for p in self.backbone.embed_tokens.parameters():
                p.requires_grad_(False)

        # Freeze first n layers (handles both 'layers' and 'h' attribute names)
        layers = None
        for attr in ["layers", "h", "encoder.layer"]:
            try:
                layers = self.backbone
                for part in attr.split("."):
                    layers = getattr(layers, part)
                break
            except AttributeError:
                layers = None

        if layers is not None:
            for layer in list(layers)[:n]:
                for p in layer.parameters():
                    p.requires_grad_(False)
            print(f"[Model] Froze first {n} backbone layers")

    # ------------------------------------------------------------------

    def encode_observation(
        self,
        input_ids: torch.Tensor,        # (B, seq_len)  tokenized obs + instruction
        attention_mask: torch.Tensor,   # (B, seq_len)
    ) -> torch.Tensor:
        """
        Run backbone forward pass, return last hidden states.
        Shape: (B, seq_len, hidden_dim)
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return outputs.last_hidden_state   # (B, seq_len, hidden_dim)

    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,          # (B, seq_len)
        attention_mask: torch.Tensor,     # (B, seq_len)
        actions_gt: torch.Tensor = None,  # (B, action_horizon, action_dim) — training only
    ) -> dict:
        context = self.encode_observation(input_ids, attention_mask)

        if actions_gt is not None:
            loss = self.decoder.cfm_loss(actions_gt, context)
            return {"loss": loss, "actions": None}
        else:
            actions = self.decoder.sample(context, action_horizon=4)
            return {"loss": None, "actions": actions}
