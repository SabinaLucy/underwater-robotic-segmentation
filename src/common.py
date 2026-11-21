from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("data/SUIM")
SIZE = 256
NUM_CLASSES = 8
CLASS_NAMES = ["water", "diver", "plant", "wreck", "robot", "reef", "fish", "sea-floor"]


def class_ids(mask_rgb):
    # SUIM codes use one bit per channel (R is the high bit), so thresholding
    # at 128 maps each pixel to a class id from 0 to 7.
    bits = (mask_rgb > 127).astype(np.uint8)
    return bits[..., 0] * 4 + bits[..., 1] * 2 + bits[..., 2]


def split_dirs(split):
    base = ROOT / ("TEST" if split == "test" else "train_val")
    return base / "images", base / "masks"


def load_image(split, stem):
    # One shared bicubic resize, the same one FUnIE-GAN's own test script uses,
    # so every enhancement condition starts from identical 256x256 pixels.
    img_dir, _ = split_dirs(split)
    img = Image.open(img_dir / f"{stem}.jpg").convert("RGB")
    return np.array(img.resize((SIZE, SIZE), Image.BICUBIC))


def load_mask(split, stem):
    # Nearest-neighbor keeps class ids intact; interpolation would invent classes.
    _, mask_dir = split_dirs(split)
    rgb = np.array(Image.open(mask_dir / f"{stem}.bmp").convert("RGB"))
    ids = Image.fromarray(class_ids(rgb), mode="L")
    return np.array(ids.resize((SIZE, SIZE), Image.NEAREST), dtype=np.int64)


def confusion_matrix(pred, target, n=NUM_CLASSES):
    idx = target.reshape(-1) * n + pred.reshape(-1)
    return np.bincount(idx, minlength=n * n).reshape(n, n)


def iou_per_class(cm):
    inter = np.diag(cm).astype(float)
    union = cm.sum(axis=0) + cm.sum(axis=1) - np.diag(cm)
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, np.nan)
    return iou


def miou(cm):
    # Classes absent from both prediction and ground truth are skipped, not counted as zero.
    return float(np.nanmean(iou_per_class(cm)))