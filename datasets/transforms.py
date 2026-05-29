"""
Image augmentation pipelines (standard, clinical, evaluation) using Albumentations.

"""

import logging
import albumentations as A
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)

DEFAULT_IMAGE_SIZE = 128
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms_standard(image_size: int = DEFAULT_IMAGE_SIZE) -> A.Compose:
    """Standard training augmentations with flips, rotation, and strong colour jitter."""
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_train_transforms_clinical(image_size: int = DEFAULT_IMAGE_SIZE) -> A.Compose:
    """Milder training augmentations suitable for clinical imaging (less colour jitter)."""
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_eval_transforms(image_size: int = DEFAULT_IMAGE_SIZE) -> A.Compose:
    """Resize and normalise only – no augmentations for evaluation/inference."""
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_transforms(name: str, image_size: int = DEFAULT_IMAGE_SIZE, training: bool = True) -> A.Compose:
    """Return the training or evaluation transform pipeline by name."""
    if not training:
        logger.info(f"Using evaluation transforms (image_size={image_size})")
        return get_eval_transforms(image_size)

    if name == "standard":
        logger.info(f"Using standard training transforms (image_size={image_size})")
        return get_train_transforms_standard(image_size)

    if name == "clinical":
        logger.info(f"Using clinical training transforms (image_size={image_size})")
        return get_train_transforms_clinical(image_size)

    raise ValueError(f"Unknown transform pipeline: {name}. Choose from: standard, clinical")


if __name__ == "__main__":
    import numpy as np
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    sample_image = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)

    for name in ["standard", "clinical"]:
        transform = get_transforms(name, image_size=128, training=True)
        result = transform(image=sample_image)
        tensor = result["image"]
        print(f"{name}: input shape (256, 256, 3) -> output shape {tuple(tensor.shape)}")

    eval_transform = get_transforms("standard", image_size=128, training=False)
    result = eval_transform(image=sample_image)
    print(f"eval: input shape (256, 256, 3) -> output shape {tuple(result['image'].shape)}")