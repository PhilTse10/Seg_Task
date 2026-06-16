from pathlib import Path

import torch
import torch.nn as nn
from monai.networks.nets.swin_unetr import SwinTransformer


def _unwrap_checkpoint(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint

    for key in ("state_dict", "model", "encoder", "backbone"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def _clean_state_dict(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        if key == "mask_token" or key.endswith(".mask_token"):
            continue
        new_key = key
        for prefix in ("module.", "encoder.", "backbone.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


class EchoCareBackbone(nn.Module):
    """Frozen EchoCare Swin encoder."""

    def __init__(
        self,
        checkpoint_path,
        feature_size=128,
        in_channels=3,
        use_checkpoint=False,
        strict=False,
    ):
        super().__init__()
        self.encoder = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=[8] * 2,
            patch_size=[2] * 2,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            mlp_ratio=4.0,
            qkv_bias=True,
            use_checkpoint=use_checkpoint,
            spatial_dims=2,
            use_v2=True,
        )

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"EchoCare checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = _clean_state_dict(_unwrap_checkpoint(checkpoint))
        msg = self.encoder.load_state_dict(state_dict, strict=strict)
        print(f"Loaded EchoCare encoder weights from {checkpoint_path}: {msg}")

        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

    def forward(self, x):
        return list(self.encoder(x))
