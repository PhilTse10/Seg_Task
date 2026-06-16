from pathlib import Path

import timm
import torch
import torch.nn as nn


def _unwrap_checkpoint(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint

    for key in ("state_dict", "model", "encoder", "backbone"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def _extract_visual_state_dict(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        if not key.startswith("base_clip.visual.trunk."):
            continue
        cleaned[key.replace("base_clip.visual.trunk.", "")] = value
    return cleaned


class USCLIPBackbone(nn.Module):
    """Frozen Ultrasound-CLIP ViT-B/16 image encoder."""

    def __init__(
        self,
        checkpoint_path,
        layer_indices=(2, 5, 8, 11),
        model_name="vit_base_patch16_224",
        strict=True,
    ):
        super().__init__()
        if not layer_indices:
            raise ValueError("layer_indices must contain at least one ViT block index.")

        self.layer_indices = tuple(layer_indices)
        self.encoder = timm.create_model(model_name, pretrained=False, num_classes=0)

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Ultrasound-CLIP checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        visual_state = _extract_visual_state_dict(_unwrap_checkpoint(checkpoint))
        if not visual_state:
            raise KeyError(
                "No visual trunk weights found in checkpoint. "
                "Expected keys prefixed with 'base_clip.visual.trunk.'."
            )

        msg = self.encoder.load_state_dict(visual_state, strict=strict)
        print(f"Loaded Ultrasound-CLIP visual encoder from {checkpoint_path}: {msg}")

        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

    def _tokens_to_map(self, tokens):
        patch_tokens = tokens[:, 1:, :]
        batch_size, num_tokens, channels = patch_tokens.shape
        spatial_size = int(num_tokens ** 0.5)
        if spatial_size * spatial_size != num_tokens:
            raise ValueError(f"Expected square patch grid, got {num_tokens} tokens.")
        return patch_tokens.transpose(1, 2).reshape(batch_size, channels, spatial_size, spatial_size)

    def forward(self, x):
        x = self.encoder.patch_embed(x)
        x = self.encoder._pos_embed(x)
        x = self.encoder.norm_pre(x)

        features = []
        for block_idx, block in enumerate(self.encoder.blocks):
            x = block(x)
            if block_idx in self.layer_indices:
                features.append(self._tokens_to_map(x))

        if len(features) != len(self.layer_indices):
            raise RuntimeError(
                f"Expected {len(self.layer_indices)} feature maps, got {len(features)}."
            )
        return features
