"""Reproducible SUIM U-Net training with validation selection and Drive-safe resume.

Run from project root, e.g.:
python src/train.py --out /content/drive/MyDrive/Underwater_Research/checkpoints/notebook_02/final
Tune only using train/val; never use test for model selection.
"""
import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from common import NUM_CLASSES, CLASS_NAMES, confusion_matrix, iou_per_class, miou
from dataset import SuimDataset, DEFAULT_MANIFEST
from unet import SmallUNet


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def worker_seed(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed); random.seed(seed)


def dice_loss(logits, target, smooth=1.0):
    probs = logits.softmax(1)
    target_one_hot = F.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).to(probs.dtype)
    dims = (0, 2, 3)
    intersection = (probs * target_one_hot).sum(dims)
    denominator = probs.sum(dims) + target_one_hot.sum(dims)
    present = target_one_hot.sum(dims) > 0
    dice = (2 * intersection + smooth) / (denominator + smooth)
    return (1 - dice[present]).mean() if present.any() else logits.sum() * 0


def criterion(logits, target, mode):
    ce = F.cross_entropy(logits, target)
    return ce if mode == 'ce' else ce + dice_loss(logits, target)


@torch.no_grad()
def evaluate(model, loader, device, loss_mode):
    model.eval()
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    total = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total += criterion(logits, y, loss_mode).item() * x.size(0)
        n += x.size(0)
        cm += confusion_matrix(logits.argmax(1).cpu().numpy(), y.cpu().numpy())
    return total / n, cm


def atomic_save(obj, path):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_history(path, rows):
    if not rows: return
    tmp = path.with_suffix('.csv.tmp')
    with open(tmp, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default=str(DEFAULT_MANIFEST))
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--loss', choices=['ce', 'ce_dice'], default='ce_dice')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--val-limit', type=int, default=None)
    ap.add_argument('--out', default='checkpoints/notebook_02/final')
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    if args.epochs < 1 or args.patience < 1: ap.error('epochs and patience must be positive')
    set_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    manifest = Path(args.manifest)
    if not manifest.exists(): raise FileNotFoundError(manifest)
    train_ds = SuimDataset('train', manifest, limit=args.limit, augment=True)
    val_ds = SuimDataset('val', manifest, limit=args.val_limit)
    generator = torch.Generator().manual_seed(args.seed)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, worker_init_fn=worker_seed, generator=generator)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, worker_init_fn=worker_seed)
    model = SmallUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)
    config = vars(args).copy()
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    start, best_score, best_epoch, stale, history = 1, -1.0, 0, 0, []
    if args.resume:
        ckpt_path = out / 'last.pth'
        if not ckpt_path.exists(): raise FileNotFoundError(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        keys = ('manifest', 'lr', 'weight_decay', 'loss', 'batch_size', 'seed', 'limit', 'val_limit')
        for key in keys:
            if ckpt['config'][key] != config[key]:
                raise ValueError(f'Resume config mismatch: {key}')
        model.load_state_dict(ckpt['model']); optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start = ckpt['epoch'] + 1
        best_score, best_epoch, stale = ckpt['best_score'], ckpt['best_epoch'], ckpt['stale']
        history = ckpt['history']
        random.setstate(ckpt['python_rng']); np.random.set_state(ckpt['numpy_rng'])
        torch.set_rng_state(ckpt['torch_rng'].detach().cpu().to(dtype=torch.uint8))
        if torch.cuda.is_available() and ckpt.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all([state.detach().cpu().to(dtype=torch.uint8) for state in ckpt['cuda_rng']])
        generator.set_state(ckpt['loader_rng'].detach().cpu().to(dtype=torch.uint8))
        print(f'Resumed from epoch {start - 1}')
    print(f'Device: {device}; train={len(train_ds)} val={len(val_ds)}; parameters={sum(p.numel() for p in model.parameters()):,}')
    for epoch in range(start, args.epochs + 1):
        model.train(); t0 = time.perf_counter(); running = n = 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y, args.loss)
            loss.backward(); optimizer.step()
            running += loss.item() * x.size(0); n += x.size(0)
        val_loss, cm = evaluate(model, val_dl, device, args.loss)
        score = miou(cm)
        improved = score > best_score + 1e-6
        if improved:
            best_score, best_epoch, stale = score, epoch, 0
        else:
            stale += 1
        scheduler.step(score)
        per_class = iou_per_class(cm)
        pixel_accuracy = float(np.trace(cm) / cm.sum()) if cm.sum() else float('nan')
        row = dict(epoch=epoch, pixel_accuracy=pixel_accuracy, train_loss=running/n, val_loss=val_loss,
                   val_miou=score, lr=optimizer.param_groups[0]['lr'],
                   seconds=time.perf_counter()-t0)
        row.update({f'iou_{name.replace("-", "_")}': float(v) for name, v in zip(CLASS_NAMES, per_class)})
        history.append(row)
        checkpoint = dict(epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
                          scheduler=scheduler.state_dict(), best_score=best_score, best_epoch=best_epoch,
                          stale=stale, history=history, config=config,
                          python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                          torch_rng=torch.get_rng_state(),
                          cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                          loader_rng=generator.get_state())
        if improved:
            atomic_save(checkpoint, out / 'best.pth')
        atomic_save(checkpoint, out / 'last.pth')
        write_history(out / 'history.csv', history)
        print(f'Epoch {epoch:02d}/{args.epochs:02d} | train_loss={row["train_loss"]:.4f} '
              f'val_loss={val_loss:.4f} | val_mIoU={score:.4f} | '
              f'pixel_acc={100*pixel_accuracy:.2f}% | best_mIoU={best_score:.4f} '
              f'(epoch {best_epoch}) | lr={row["lr"]:.2g} | {row["seconds"]:.0f}s', flush=True)
        if stale >= args.patience:
            print(f'Early stopping: no validation mIoU improvement for {stale} epochs')
            break
    if not (out / 'best.pth').exists(): raise RuntimeError('Best checkpoint missing')
    best_ckpt = torch.load(out / 'best.pth', map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['model'])
    best_loss, best_cm = evaluate(model, val_dl, device, args.loss)
    summary = dict(best_epoch=best_ckpt['epoch'], best_val_miou=miou(best_cm),
                   best_pixel_accuracy=float(np.trace(best_cm) / best_cm.sum()),
                   best_val_loss=best_loss, per_class_iou={name: (None if np.isnan(v) else float(v))
                   for name, v in zip(CLASS_NAMES, iou_per_class(best_cm))},
                   checkpoint=str(out / 'best.pth'))
    (out / 'best_validation.json').write_text(json.dumps(summary, indent=2))
    print('Best checkpoint validation:', json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
