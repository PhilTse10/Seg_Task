import random

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F


def get_transforms(img_size, mask_values=None):
    if mask_values is not None:
        mask_value_to_index = {int(value): index for index, value in enumerate(mask_values)}
    else:
        mask_value_to_index = None

    def resize(image, mask):
        image = F.resize(image, [img_size, img_size])
        mask = F.resize(mask, [img_size, img_size], interpolation=Image.NEAREST)
        return image, mask

    def random_rot90(image, mask, p=0.5):
        if random.random() < p:
            k = random.randint(1, 3)
            image = image.rotate(90 * k, expand=True)
            mask = mask.rotate(90 * k, expand=True)
            image = F.center_crop(image, (img_size, img_size))
            mask = F.center_crop(mask, (img_size, img_size))
        return image, mask

    def random_hflip(image, mask, p=0.5):
        if random.random() < p:
            image = F.hflip(image)
            mask = F.hflip(mask)
        return image, mask

    def random_vflip(image, mask, p=0.5):
        if random.random() < p:
            image = F.vflip(image)
            mask = F.vflip(mask)
        return image, mask

    image_to_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    def mask_to_tensor(mask):
        mask = np.array(mask, dtype=np.uint8)
        if mask_value_to_index is None:
            return torch.from_numpy(mask.astype(np.int64))

        mapped = np.zeros_like(mask, dtype=np.int64)
        for raw_value, class_index in mask_value_to_index.items():
            mapped[mask == raw_value] = class_index
        return torch.from_numpy(mapped)

    def train_transform(image, mask):
        image, mask = resize(image, mask)
        image, mask = random_rot90(image, mask)
        image, mask = random_hflip(image, mask)
        image, mask = random_vflip(image, mask)
        return image_to_tensor(image), mask_to_tensor(mask)

    def val_transform(image, mask):
        image, mask = resize(image, mask)
        return image_to_tensor(image), mask_to_tensor(mask)

    return train_transform, val_transform
