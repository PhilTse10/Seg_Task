#!/usr/bin/env python3
import argparse
import json
import random
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
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


def fix_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_datasets(args):
    train_transform, val_transform = get_transforms(
        img_size=args.img_size,
        mask_values=args.mask_values,
    )
    train_ds = CSVSegDataset(args.csv_file, split="train", transform=train_transform)
    val_ds = CSVSegDataset(args.csv_file, split="val", transform=val_transform)
    return train_ds, val_ds


def train_one_epoch(backbone, decoder, loader, optimizer, criterion, epoch, print_freq):
    backbone.eval()
    decoder.train()

    total_loss = 0.0
    start = time.time()
    for step, (images, masks, _) in enumerate(loader, start=1):
        images = images.cuda(non_blocking=True)
        masks = masks.cuda(non_blocking=True)

        with torch.no_grad():
            features = backbone(images)

        logits = decoder(features, output_size=images.shape[2:])
        loss = criterion(logits, masks)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if step % print_freq == 0:
            print(
                f"Epoch [{epoch}] step [{step}/{len(loader)}] "
                f"loss={total_loss / step:.4f} lr={optimizer.param_groups[0]['lr']:.6f}"
            )

    return {"loss": total_loss / max(len(loader), 1), "time": time.time() - start}


@torch.no_grad()
def validate(backbone, decoder, loader, criterion, num_classes):
    backbone.eval()
    decoder.eval()

    totals, counts = init_metric_accumulators()
    total_loss = 0.0
    total_samples = 0

    for images, masks, _ in loader:
        images = images.cuda(non_blocking=True)
        masks = masks.cuda(non_blocking=True)

        features = backbone(images)
        logits = decoder(features, output_size=images.shape[2:])
        loss = criterion(logits, masks)
        preds = logits.argmax(dim=1)

        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        for idx in range(batch_size):
            sample_metrics = compute_foreground_metrics(preds[idx], masks[idx], num_classes)
            accumulate_sample_metrics(totals, counts, sample_metrics)
        total_samples += batch_size

    return build_result_metrics(
        average_accumulated_metrics(totals, counts),
        total_loss / max(total_samples, 1),
        total_samples,
    )


def save_checkpoint(args, decoder, optimizer, scheduler, epoch, best_metrics, suffix):
    checkpoint = {
        "epoch": epoch,
        "encoder": args.encoder,
        "state_dict": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_metrics": best_metrics,
        "args": vars(args),
    }
    path = Path(args.output_dir) / f"checkpoint_{args.encoder}_dpt_seg_{suffix}.pth"
    torch.save(checkpoint, path)
    print(f"Saved checkpoint: {path}")


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for segmentation training.")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    fix_random_seeds(args.seed)
    torch.backends.cudnn.benchmark = True

    train_ds, val_ds = build_datasets(args)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Data loaded: {len(train_ds)} train, {len(val_ds)} val")

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

    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_index)

    best_metrics = {
        "foreground_iou": 0.0,
        "foreground_dice": 0.0,
        "hd": float("inf"),
        "hd95": float("inf"),
    }
    log_path = Path(args.output_dir) / f"log_{args.log_name}.txt"

    for epoch in range(args.epochs):
        train_stats = train_one_epoch(
            backbone,
            decoder,
            train_loader,
            optimizer,
            criterion,
            epoch,
            args.print_freq,
        )
        scheduler.step()

        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            val_stats = validate(backbone, decoder, val_loader, criterion, args.num_classes)
            improved = (
                val_stats["foreground_iou"] > best_metrics["foreground_iou"]
                or val_stats["foreground_dice"] > best_metrics["foreground_dice"]
                or val_stats["hd95"] < best_metrics["hd95"]
            )
            best_metrics["foreground_iou"] = max(best_metrics["foreground_iou"], val_stats["foreground_iou"])
            best_metrics["foreground_dice"] = max(best_metrics["foreground_dice"], val_stats["foreground_dice"])
            best_metrics["hd"] = min(best_metrics["hd"], val_stats["hd"])
            best_metrics["hd95"] = min(best_metrics["hd95"], val_stats["hd95"])

            log_stats = {
                "epoch": epoch,
                "encoder": args.encoder,
                "train_loss": train_stats["loss"],
                "loss": val_stats["loss"],
                "foreground_iou": val_stats["foreground_iou"],
                "foreground_dice": val_stats["foreground_dice"],
                "hd": val_stats["hd"],
                "hd95": val_stats["hd95"],
                "num_samples": val_stats["num_samples"],
            }
            print(json.dumps(log_stats, indent=2))
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(log_stats) + "\n")

            if improved:
                save_checkpoint(args, decoder, optimizer, scheduler, epoch + 1, best_metrics, "best")

        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(args, decoder, optimizer, scheduler, epoch + 1, best_metrics, f"epoch_{epoch + 1}")

    print(
        f"Training complete [{args.encoder}]. "
        f"Best foreground IoU={best_metrics['foreground_iou']:.4f}, "
        f"Best foreground Dice={best_metrics['foreground_dice']:.4f}, "
        f"Best HD={best_metrics['hd']:.4f}, "
        f"Best HD95={best_metrics['hd95']:.4f}"
    )


def resolve_csv_file(args):
    if args.csv_preset:
        if args.csv_preset not in CSV_PRESETS:
            raise ValueError(f"Unknown csv_preset '{args.csv_preset}'. Choose from: {list(CSV_PRESETS)}")
        return CSV_PRESETS[args.csv_preset]
    if args.csv_file:
        return args.csv_file
    return CSV_PRESETS["train100"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train DPT decoder with frozen encoder backbone")
    parser.add_argument("--encoder", required=True, choices=ENCODER_NAMES)
    parser.add_argument(
        "--csv_preset",
        default="",
        choices=["", "train10", "train20", "train50", "train100"],
        help="Shortcut for the 3VT CSV presets.",
    )
    parser.add_argument("--csv_file", default="", help="Custom CSV path (overridden by --csv_preset if set).")
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--log_name", default="")
    parser.add_argument("--num_classes", default=3, type=int)
    parser.add_argument("--mask_values", default="0,3,4", type=parse_mask_values)
    parser.add_argument("--ignore_index", default=255, type=int)
    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--dpt_features", default=256, type=int)
    parser.add_argument("--dpt_dropout", default=0.1, type=float)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--min_lr", default=0.0, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--val_freq", default=1, type=int)
    parser.add_argument("--save_freq", default=10, type=int)
    parser.add_argument("--print_freq", default=20, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--strict_backbone_load", action="store_true")
    args = parser.parse_args()

    args.csv_file = resolve_csv_file(args)
    if not args.log_name:
        preset = args.csv_preset or "custom"
        args.log_name = f"{args.encoder}_dpt_{preset}"

    main(args)
