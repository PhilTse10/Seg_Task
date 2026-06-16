from pathlib import Path

import torch
import torch.nn as nn

from encoder.openus_vmamba.dino_vmamba import Backbone_DINOv2_VSSM_2


class OpenUSBackbone(nn.Module):
    """Frozen OpenUS VMamba encoder with multiscale segmentation features."""

    def __init__(
        self,
        openus_weights,
        checkpoint_key="teacher",
    ):
        super().__init__()
        self.backbone = Backbone_DINOv2_VSSM_2(pretrained=None, seg_head=True)

        openus_weights = Path(openus_weights)
        if not openus_weights.is_file():
            raise FileNotFoundError(f"OpenUS checkpoint not found: {openus_weights}")

        checkpoint = torch.load(openus_weights, map_location="cpu", weights_only=False)
        if checkpoint_key not in checkpoint:
            raise KeyError(
                f"Checkpoint key '{checkpoint_key}' not found in {openus_weights}. "
                f"Available keys: {list(checkpoint.keys())}"
            )

        state_dict = {
            key.replace("module.", "").replace("backbone.", ""): value
            for key, value in checkpoint[checkpoint_key].items()
            if key.replace("module.", "").startswith("backbone.")
        }
        msg = self.backbone.load_state_dict(state_dict, strict=False)
        print(f"Loaded OpenUS {checkpoint_key} backbone from {openus_weights}: {msg}")

        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def forward(self, x):
        return self.backbone(x)
