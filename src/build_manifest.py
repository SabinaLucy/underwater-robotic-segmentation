import hashlib
import argparse
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path("data/SUIM")
SEED = 42
VAL_FRACTION = 0.15
AGREE_MIN = 0.90


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def load_mask(path):
    return np.array(Image.open(path).convert("RGB"))


def class_ids(mask_rgb):
    bits = (mask_rgb > 127).astype(np.uint8)
    return bits[..., 0] * 4 + bits[..., 1] * 2 + bits[..., 2]


def main():
    parser = argparse.ArgumentParser(description="Rebuild exploratory manifest; never overwrite Notebook 01 locked split")
    parser.add_argument("--allow-regenerate", action="store_true", help="Explicitly permit exploratory manifest generation")
    args = parser.parse_args()
    if not args.allow_regenerate:
        raise SystemExit("Refusing to regenerate a split by default. Use the locked results/tables/notebook_01/manifest.csv for research. Pass --allow-regenerate only for a separate exploratory rebuild.")
    img_dir = ROOT / "train_val" / "images"
    mask_dir = ROOT / "train_val" / "masks"
    stems = sorted(p.stem for p in img_dir.glob("*.jpg"))
    test_hashes = {md5(p) for p in (ROOT / "TEST" / "images").glob("*.jpg")}

    # Each image gets one reason, checked in this order, so counts per reason depend on the order and do not overlap.
    exclusions, hashes = {}, {}
    for stem in stems:
        img_path = img_dir / f"{stem}.jpg"
        mask = load_mask(mask_dir / f"{stem}.bmp")
        with Image.open(img_path) as im:
            size_ok = im.size == (mask.shape[1], mask.shape[0])
        hashes[stem] = md5(img_path)
        if hashes[stem] in test_hashes:
            exclusions[stem] = "image identical to a TEST image"
        elif not size_ok:
            exclusions[stem] = "mask size differs from image"
        elif not np.isin(mask, (0, 255)).all():
            exclusions[stem] = "mask has off-code pixel values"

    groups = defaultdict(list)
    for stem in stems:
        if stem not in exclusions:
            groups[hashes[stem]].append(stem)
    for members in groups.values():
        if len(members) < 2:
            continue
        ref = class_ids(load_mask(mask_dir / f"{members[0]}.bmp"))
        agree = [(class_ids(load_mask(mask_dir / f"{m}.bmp")) == ref).mean() for m in members[1:]]
        if min(agree) < AGREE_MIN:
            for m in members:
                exclusions[m] = "duplicate image with conflicting masks"
        else:
            for m in members[1:]:
                exclusions[m] = "duplicate image, other copy kept"

    kept = [s for s in stems if s not in exclusions]
    random.Random(SEED).shuffle(kept)
    n_val = round(len(kept) * VAL_FRACTION)
    rows = [{"split": "val", "stem": s} for s in kept[:n_val]]
    rows += [{"split": "train", "stem": s} for s in kept[n_val:]]
    rows += [{"split": "test", "stem": p.stem} for p in sorted((ROOT / "TEST" / "images").glob("*.jpg"))]

    out = Path("results/tables")
    out.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out / "manifest.csv", index=False)
    excl = pd.DataFrame(sorted(exclusions.items()), columns=["stem", "reason"])
    excl.to_csv(out / "exclusions.csv", index=False)
    print(manifest["split"].value_counts().to_string())
    print(excl["reason"].value_counts().to_string())
    print("total excluded:", len(excl), "of", len(stems))


if __name__ == "__main__":
    main()