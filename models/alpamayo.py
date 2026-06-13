"""
alpamayo.py — Full Alpamayo-Drone model

Architecture:
    ViT encoder (frozen)  →  per-frame visual tokens ┐
                                                       ├→ fused context memory
    LLM backbone          →  text hidden states       ┘   (LoRA finetuned)
    FlowMatchingDecoder   →  UAV action sequence           (fully trained)

The backbone's causal LM head is discarded; instead its last-layer hidden
states are concatenated with the projected ViT frame embeddings and passed
together as cross-attention memory to the FlowMatchingDecoder. This makes the
model an actual vision-language-action policy: the action head attends to both
the instruction tokens and the observed image history.

Set model.use_vision: false in the config to ablate the visual pathway and
fall back to a text-only policy.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, ViTModel
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
        # 1.  Load backbone (plain LLM as Alpamayo-R1 proxy; vision is
        #     provided separately by the ViT pathway below)
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

        # device_map only for the quantized path: a quantized model is
        # dispatched by accelerate and must never be moved with .to() again.
        load_kwargs = dict(
            dtype=torch.bfloat16,
            trust_remote_code=mc.get("trust_remote_code", False),
        )
        if bnb_config is not None:
            load_kwargs["quantization_config"] = bnb_config
            load_kwargs["device_map"] = "auto"
        self.backbone_dispatched = bnb_config is not None

        self.backbone = AutoModel.from_pretrained(
            mc["base_model_id"], **load_kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            mc["base_model_id"],
            trust_remote_code=mc.get("trust_remote_code", False),
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

        # Infer context_dim from backbone hidden size
        context_dim = self.backbone.config.hidden_size

        # ------------------------------------------------------------------
        # 2.5  Vision encoder (ViT) — frozen; one summary token per frame
        # ------------------------------------------------------------------
        self.use_vision = mc.get("use_vision", True)
        if self.use_vision:
            print(f"[Model] Loading vision encoder: {mc['vit_model']}")
            self.vit = ViTModel.from_pretrained(
                mc["vit_model"], dtype=torch.bfloat16
            )
            if mc.get("vit_freeze", True):
                for p in self.vit.parameters():
                    p.requires_grad_(False)
                # ViT-base has zero hidden/attention dropout, so leaving it in
                # train() mode is numerically identical to eval(); we keep it
                # frozen via requires_grad only.
                print("[Model] Froze ViT encoder")
            # Project each frame's CLS embedding into the backbone hidden size
            # so visual tokens can be concatenated with text hidden states and
            # consumed by the decoder's existing context_proj unchanged.
            self.vis_proj = nn.Linear(self.vit.config.hidden_size, context_dim)
            print(
                f"[Model] Vision projection: "
                f"{self.vit.config.hidden_size} -> {context_dim}"
            )
        else:
            self.vit = None
            self.vis_proj = None
            print("[Model] use_vision=False — text-only policy")

        # ------------------------------------------------------------------
        # 3.  FlowMatchingDecoder (fully trainable)
        # ------------------------------------------------------------------
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
        # 4.  Action normalization stats (identity until set_action_stats).
        #     The decoder integrates from N(0, I), so it is trained on
        #     normalized actions; forward() normalizes targets and
        #     denormalizes samples so callers only ever see physical units.
        # ------------------------------------------------------------------
        self.register_buffer("action_mean", torch.zeros(fm["action_dim"]))
        self.register_buffer("action_std", torch.ones(fm["action_dim"]))

    def set_action_stats(self, mean, std):
        mean = torch.as_tensor(mean, dtype=torch.float32)
        std = torch.as_tensor(std, dtype=torch.float32)
        self.action_mean.copy_(mean.to(self.action_mean.device))
        self.action_std.copy_(std.to(self.action_std.device))
        print(f"[Model] Action stats set: mean={mean.tolist()} std={std.tolist()}")

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

    def prepare_device(self, device: torch.device):
        """
        Move the model to `device`.

        Use this instead of .to(device): a 4-bit backbone is dispatched by
        accelerate at load time and raises if moved again, so only the
        non-dispatched submodules are moved in that case.
        """
        if self.backbone_dispatched:
            for name, module in self.named_children():
                if name != "backbone":
                    module.to(device)
            # Top-level buffers (action stats) are not children — move them too
            for name, buf in self.named_buffers(recurse=False):
                setattr(self, name, buf.to(device))
        else:
            self.to(device)
        return self

    # ------------------------------------------------------------------

    def _encode_frames(self, images: torch.Tensor, out_dtype: torch.dtype):
        """
        Encode an image history into one visual token per frame.

        images: (B, S, 3, H, W)  →  (B, S, context_dim)
        """
        B, S = images.shape[:2]
        flat = images.reshape(B * S, *images.shape[2:])       # (B*S, 3, H, W)
        vit_dtype = next(self.vit.parameters()).dtype
        vit_out = self.vit(pixel_values=flat.to(vit_dtype))
        cls = vit_out.last_hidden_state[:, 0]                  # (B*S, vit_hidden)
        cls = cls.reshape(B, S, -1).to(self.vis_proj.weight.dtype)
        vis = self.vis_proj(cls)                              # (B, S, context_dim)
        return vis.to(out_dtype)

    def encode_observation(
        self,
        input_ids: torch.Tensor,        # (B, seq_len)  tokenized obs + instruction
        attention_mask: torch.Tensor,   # (B, seq_len)
        images: torch.Tensor = None,    # (B, S, 3, H, W)  frame history
    ) -> torch.Tensor:
        """
        Run the backbone, then (if vision is enabled and images are provided)
        prepend the projected per-frame ViT embeddings to the text hidden
        states. The concatenation along the sequence axis becomes the decoder's
        cross-attention memory.

        Returns (B, S + seq_len, hidden_dim) with vision, else (B, seq_len, hidden_dim).
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        text_ctx = outputs.last_hidden_state   # (B, seq_len, hidden_dim)

        if self.use_vision and images is not None:
            vis = self._encode_frames(images, out_dtype=text_ctx.dtype)
            return torch.cat([vis, text_ctx], dim=1)

        return text_ctx

    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,          # (B, seq_len)
        attention_mask: torch.Tensor,     # (B, seq_len)
        images: torch.Tensor = None,      # (B, S, 3, H, W)  frame history
        actions_gt: torch.Tensor = None,  # (B, action_horizon, action_dim) — training only
    ) -> dict:
        context = self.encode_observation(input_ids, attention_mask, images)
        # The bf16 backbone context must match the decoder's param dtype when
        # running without autocast (e.g. plain eval); under autocast this cast
        # is a no-op for the subsequent matmuls.
        context = context.to(next(self.decoder.parameters()).dtype)

        if actions_gt is not None:
            actions_norm = (actions_gt - self.action_mean) / self.action_std
            loss = self.decoder.cfm_loss(actions_norm, context)
            return {"loss": loss, "actions": None}
        else:
            actions = self.decoder.sample(context, action_horizon=4)
            actions = actions * self.action_std + self.action_mean
            return {"loss": None, "actions": actions}
