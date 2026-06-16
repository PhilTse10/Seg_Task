# 统一调用encoder的方法
from pathlib import Path
from typing import Any, Dict, Tuple

from encoder.echocare_backbone import EchoCareBackbone
from encoder.openus_backbone import OpenUSBackbone
from encoder.ultrafedfm_backbone import UltraFedFMBackbone
from encoder.usclip_backbone import USCLIPBackbone
from encoder.usfm_backbone import OfficialUSFMBackbone

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHT_ROOT = PROJECT_ROOT / "weight"

ENCODER_NAMES = ("echocare", "openus", "usclip", "ultrafedfm", "usfm")

ENCODER_CONFIGS: Dict[str, Dict[str, Any]] = {
    "echocare": {
        "in_channels": (128, 256, 512, 1024, 2048),
        "weight": WEIGHT_ROOT / "ECHO" / "echocare_encoder.pth",
    },
    "openus": {
        "in_channels": (96, 192, 384, 768),
        "openus_weights": WEIGHT_ROOT / "OpenUS" / "openus" / "openus_cpt0150.pth",
        "checkpoint_key": "teacher",
    },
    "usclip": {
        "in_channels": (768, 768, 768, 768),
        "weight": WEIGHT_ROOT / "US-CLIP" / "checkpoints.pt",
        "layer_indices": (2, 5, 8, 11),
    },
    "ultrafedfm": {
        "in_channels": (768, 768, 768, 768),
        "weight": WEIGHT_ROOT / "UltraFedFM" / "checkpoint.pth",
        "layer_indices": (2, 5, 8, 11),
    },
    "usfm": {
        "in_channels": (768, 768, 768, 768),
        "weight": WEIGHT_ROOT / "USFM" / "USFM_latest.pth",
        "out_indices": (3, 5, 7, 11),
    },
}

# 数据集路径
CSV_PRESETS = {
    "train10": "/sdb1/liran/downsteam_code/3VT/dataset_train10.csv",
    "train20": "/sdb1/liran/downsteam_code/3VT/dataset_train20.csv",
    "train50": "/sdb1/liran/downsteam_code/3VT/dataset_train50.csv",
    "train100": "/sdb1/liran/downsteam_code/3VT/dataset_train100.csv",
}


def get_encoder_config(name: str) -> Dict[str, Any]:
    if name not in ENCODER_CONFIGS:
        raise ValueError(f"Unknown encoder '{name}'. Choose from: {ENCODER_NAMES}")
    return ENCODER_CONFIGS[name]


def build_encoder(name: str, img_size: int = 224, strict_backbone_load: bool = False):
    config = get_encoder_config(name)

    if name == "echocare":
        backbone = EchoCareBackbone(
            checkpoint_path=config["weight"],
            feature_size=128,
            use_checkpoint=False,
            strict=strict_backbone_load,
        )
    elif name == "openus":
        backbone = OpenUSBackbone(
            openus_weights=config["openus_weights"],
            checkpoint_key=config["checkpoint_key"],
        )
    elif name == "usclip":
        backbone = USCLIPBackbone(
            checkpoint_path=config["weight"],
            layer_indices=config["layer_indices"],
            strict=strict_backbone_load,
        )
    elif name == "ultrafedfm":
        backbone = UltraFedFMBackbone(
            checkpoint_path=config["weight"],
            layer_indices=config["layer_indices"],
            strict=strict_backbone_load,
        )
    elif name == "usfm":
        backbone = OfficialUSFMBackbone(
            img_size=img_size,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            out_indices=config["out_indices"],
            pretrained_path=str(config["weight"]),
            require_pretrained=True,
            freeze=True,
        )
    else:
        raise ValueError(f"Unsupported encoder: {name}")

    return backbone


def get_decoder_in_channels(name: str) -> Tuple[int, ...]:
    return tuple(get_encoder_config(name)["in_channels"])
