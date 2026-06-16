import csv
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class CSVSegDataset(Dataset):
    """Segmentation dataset backed by a CSV with image_path, mask_path, split columns."""

    def __init__(self, csv_file, split, transform=None):
        self.csv_file = Path(csv_file)
        self.split = split
        self.transform = transform
        self.samples = []

        if not self.csv_file.is_file():
            raise FileNotFoundError(f"CSV file not found: {self.csv_file}")

        with open(self.csv_file, newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {"image_path", "mask_path", "split"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

            for row in reader:
                if row["split"].strip().lower() == split.lower():
                    self.samples.append(
                        {
                            "image_path": row["image_path"].strip(),
                            "mask_path": row["mask_path"].strip(),
                        }
                    )

        if not self.samples:
            raise ValueError(f"No samples with split='{split}' found in {self.csv_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = Path(sample["image_path"])
        mask_path = Path(sample["mask_path"])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image, mask = self.transform(image, mask)
        else:
            image = transforms.ToTensor()(image)
            mask = torch.from_numpy(np.array(mask)).long()

        return image, mask, mask_path.name
