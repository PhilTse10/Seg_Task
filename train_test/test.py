#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

from dataset.csv_seg_dataset import CSVSegDataset
from dataset.transforms import get_transforms
from decoder.dpt_decoder import DPTDecoderHead
from encoder.factory import CSV_PRESETS, ENCODER_NAMES, build_encoder, get_decoder_in_channels
from utils.metrics import (
    accumulate_sample_metrics,
    average_accumulated_metrics,
    build_result_metrics,
    compute_foreground_metrics,
    init_metric_accumulators,
)


def parse_mask_values(value):
    return [int(item) for item in value.split(",")]


def load_decoder_checkpoint(path, decoder):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" in checkpoint:
        decoder.load_state_dict(checkpoint["state_dict"])
    else:
        decoder.load_state_dict(checkpoint)
    print(f"Loaded decoder checkpoint: {path}")
    return checkpoint


def build_test_loader(args):
    _, test_transform = get_transforms(img_size=args.img_size, mask_values=args.mask_values)
    dataset = CSVSegDataset(args.csv_file, split=args.split, transform=test_transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader, dataset


def save_masks(preds, filenames, save_dir, mask_values):
    save_dir.mkdir(parents=True, exist_ok=True)
    mask_values = np.array(mask_values, dtype=np.uint8)
    for pred, filename in zip(preds, filenames):
        name = f"{Path(str(filename)).stem}.png"
        pred_np = pred.cpu().numpy().astype(np.int64)
        mask = mask_values[pred_np]
        Image.fromarray(mask, mode="L").save(save_dir / name)


@torch.no_grad()
def evaluate(args, backbone, decoder, loader, criterion):
    totals, counts = init_metric_accumulators()
    total_loss = 0.0
    total_samples = 0
    mask_dir = Path(args.output_dir) / f"predicted_masks_{args.encoder}_{args.cpk_name}"

    for step, (images, masks, filenames) in enumerate(loader, start=1):
        images = images.cuda(non_blocking=True)
        masks = masks.cuda(non_blocking=True)

        features = backbone(images)
        logits = decoder(features, output_size=images.shape[2:])
        loss = criterion(logits, masks)
        preds = logits.argmax(dim=1)

        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        for idx in range(batch_size):
            sample_metrics = compute_foreground_metrics(preds[idx], masks[idx], args.num_classes)
            accumulate_sample_metrics(totals, counts, sample_metrics)
        total_samples += batch_size

        if args.save_masks:
            save_masks(preds, filenames, mask_dir, args.mask_values)

        if step % args.print_freq == 0:
            running = build_result_metrics(
                average_accumulated_metrics(totals, counts),
                total_loss / max(total_samples, 1),
                total_samples,
            )
            print(
                f"Processed [{step}/{len(loader)}] "
                f"loss={running['loss']:.4f} "
                f"foreground_iou={running['foreground_iou']:.4f} "
                f"foreground_dice={running['foreground_dice']:.4f} "
                f"hd={running['hd']:.4f} "
                f"hd95={running['hd95']:.4f}"
            )

    return build_result_metrics(
        average_accumulated_metrics(totals, counts),
        total_loss / max(total_samples, 1),
        total_samples,
    )


def resolve_csv_file(args):
    if args.csv_preset:
        if args.csv_preset not in CSV_PRESETS:
            raise ValueError(f"Unknown csv_preset '{args.csv_preset}'. Choose from: {list(CSV_PRESETS)}")
        return CSV_PRESETS[args.csv_preset]
    if args.csv_file:
        return args.csv_file
    return CSV_PRESETS["train100"]


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for segmentation evaluation.")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.csv_file = resolve_csv_file(args)

    loader, _ = build_test_loader(args)
    backbone = build_encoder(
        args.encoder,
        img_size=args.img_size,
        strict_backbone_load=args.strict_backbone_load,
    ).cuda()
    decoder = DPTDecoderHead(
        in_channels=get_decoder_in_channels(args.encoder),
        num_classes=args.num_classes,
        features=args.dpt_features,
        dropout=args.dpt_dropout,
    ).cuda()

    load_decoder_checkpoint(args.checkpoint, decoder)
    decoder.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_index)

    metrics = evaluate(args, backbone, decoder, loader, criterion)
    result = {
        "encoder": args.encoder,
        "split": args.split,
        "csv_file": args.csv_file,
        "checkpoint": args.checkpoint,
        **metrics,
    }
    print(json.dumps(result, indent=2))

    result_path = Path(args.output_dir) / f"test_{args.encoder}_{args.split}.json"
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"Saved metrics to {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Test DPT decoder with frozen encoder backbone")
    parser.add_argument("--encoder", required=True, choices=ENCODER_NAMES)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--csv_preset",
        default="",
        choices=["", "train10", "train20", "train50", "train100"],
    )
    parser.add_argument("--csv_file", default="")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--cpk_name", default="best")
    parser.add_argument("--num_classes", default=3, type=int)
    parser.add_argument("--mask_values", default="0,3,4", type=parse_mask_values)
    parser.add_argument("--ignore_index", default=255, type=int)
    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--dpt_features", default=256, type=int)
    parser.add_argument("--dpt_dropout", default=0.1, type=float)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--print_freq", default=20, type=int)
    parser.add_argument("--save_masks", action="store_true")
    parser.add_argument("--strict_backbone_load", action="store_true")
    args = parser.parse_args()
    main(args)
