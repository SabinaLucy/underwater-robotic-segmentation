"""SUIM dataset with a locked Notebook 01 manifest and joint spatial transforms."""
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from common import load_image, load_mask

DEFAULT_MANIFEST = Path('results/tables/notebook_01/manifest.csv')

class SuimDataset(Dataset):
    def __init__(self, split, manifest=DEFAULT_MANIFEST, limit=None, augment=False, transform=None):
        if split not in ('train', 'val', 'test'):
            raise ValueError(f'Unknown split: {split}')
        df = pd.read_csv(manifest)
        if not {'split', 'stem'}.issubset(df.columns):
            raise ValueError('Manifest requires split and stem columns')
        if df.duplicated('stem').any():
            raise ValueError('Duplicate image stems in locked manifest')
        self.stems = df.loc[df['split'] == split, 'stem'].tolist()
        if limit is not None:
            self.stems = self.stems[:limit]
        if not self.stems:
            raise ValueError(f'No entries for split={split}')
        self.split, self.augment, self.transform = split, augment, transform
        if augment and split != 'train':
            raise ValueError('Augmentation is permitted only for training')

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, i):
        stem = self.stems[i]
        img, mask = load_image(self.split, stem), load_mask(self.split, stem)
        if self.transform is not None:
            img = self.transform(img)
            if img.shape != (256, 256, 3) or img.dtype != np.uint8:
                raise ValueError('Enhancement must return 256x256 uint8 RGB image')
        if self.augment:
            if random.random() < 0.5:
                img, mask = np.fliplr(img), np.fliplr(mask)
            if random.random() < 0.25:
                img, mask = np.flipud(img), np.flipud(mask)
            if random.random() < 0.5:
                k = random.randint(1, 3)
                img, mask = np.rot90(img, k), np.rot90(mask, k)
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 127.5 - 1
        y = torch.from_numpy(np.ascontiguousarray(mask)).long()
        return x, y
