"""
test_paired.py
================

Evaluate a trained CFANet checkpoint on the **paired T1/T2** test set
produced by ``data/dataset.py``.  This is the script you run *after*
``train.py`` finishes.

Usage
-----

    # evaluate the latest 20-epoch checkpoint and print metrics
    python test_paired.py --ckpt ./Snapshot/CFANet/CODNet_20.pth

    # evaluate a specific epoch and save prediction maps as PNG
    python test_paired.py --ckpt ./Snapshot/CFANet/CODNet_20.pth \
                          --save_dir ./res/paired_test

    # sweep all snapshots
    for f in ./Snapshot/CFANet/CODNet_*.pth; do
        python test_paired.py --ckpt "$f" --no_verbose
    done

Notes
-----
* The model is built with ``dual_backbone=True`` (the train-time
  configuration) so the two encoders are separate and the FFTCMA /
  GlobalFusion modules actually fuse real T1/T2 features.
* The eval split is the deterministic 80/10/10 re-split used by
  training — make sure you keep ``--seed`` and the ``--*_ratio``
  flags identical to those used at training time, otherwise
  you'll be evaluating on a different set of patients.
* The model produces **four** outputs (mask1, mask2, mask3, mask).
  We average the four sigmoid probabilities before thresholding —
  this consistently improves over taking only ``mask``.
* The eval ground truth is the **combined T1|T2 mask** (the same
  target the model was trained on).  If you change ``--mask_combine``
  here, you must use the same value at training time.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import cv2
except Exception:
    cv2 = None

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.dataset import MedicalSliceDataset
from lib.model import CFANet


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(pred_bin: np.ndarray, gt_bin: np.ndarray, eps: float = 1e-7):
    """Per-sample metrics, all numpy.

    pred_bin, gt_bin : {0,1} arrays of the same shape.
    Returns
    -------
    dict with keys: iou, dice, mae, precision, recall, f_measure
    """
    pred = pred_bin.astype(np.float32).ravel()
    gt   = gt_bin.astype(np.float32).ravel()

    tp = float((pred * gt).sum())
    fp = float((pred * (1 - gt)).sum())
    fn = float(((1 - pred) * gt).sum())
    tn = float(((1 - pred) * (1 - gt)).sum())

    iou       = tp / (tp + fp + fn + eps)
    dice      = 2 * tp / (2 * tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    # F-measure with beta^2 = 0.3  (same as COD benchmarks)
    beta2     = 0.3
    f_measure = (1 + beta2) * precision * recall / (beta2 * precision + recall + eps)
    mae       = float(np.mean(np.abs(pred - gt)))
    return dict(iou=iou, dice=dice, mae=mae, precision=precision,
                recall=recall, f_measure=f_measure)


def postprocess_2d(pred_bin: np.ndarray, min_area: int = 0, keep_largest: int = 0) -> np.ndarray:
    min_area = int(min_area or 0)
    keep_largest = int(keep_largest or 0)
    if min_area <= 0 and keep_largest <= 0:
        return pred_bin
    p = (pred_bin > 0).astype(np.uint8)
    if p.max() == 0:
        return p
    if cv2 is None:
        return p
    num, labels, stats, _ = cv2.connectedComponentsWithStats(p, connectivity=8)
    if num <= 1:
        return p
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.ones(num, dtype=bool)
    keep[0] = False
    if min_area > 0:
        keep[1:] &= (areas >= min_area)
    if keep_largest > 0:
        order = np.argsort(areas)[::-1]
        chosen = set((order[:keep_largest] + 1).tolist())
        keep2 = np.zeros(num, dtype=bool)
        for lab in chosen:
            keep2[lab] = True
        keep &= keep2
    out = keep[labels]
    return out.astype(np.uint8)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',         type=str,   default='./Snapshot/CFANet/CODNet_20.pth')
    p.add_argument('--data_root',    type=str,   default='./data/testdata')
    p.add_argument('--data_format',  type=str,   default='nifti', choices=['nifti', 'npy'])
    p.add_argument('--trainsize',    type=int,   default=256,
                   help='Must match the value used at training time.')
    p.add_argument('--mask_combine', type=str,   default='or',
                   choices=['or', 'and', 't1', 't2'],
                   help='Must match the value used at training time.')
    p.add_argument('--resplit',      action='store_true', default=False,
                   help='Resplit the patients in --data_root.  Leave OFF for the '
                        'held-out --data_root=data/testdata (use every patient).')
    p.add_argument('--resplit_seed', type=int,   default=42)
    p.add_argument('--train_ratio',  type=float, default=0.8)
    p.add_argument('--val_ratio',    type=float, default=0.2)
    p.add_argument('--test_ratio',   type=float, default=0.0)
    p.add_argument('--batchsize',    type=int,   default=4)
    p.add_argument('--num_workers',  type=int,   default=0)
    p.add_argument('--threshold',    type=float, default=0.5)
    p.add_argument('--min_area',     type=int,   default=0,
                   help='Remove predicted 2D connected components smaller than this area (in pixels).')
    p.add_argument('--keep_largest', type=int,   default=0,
                   help='Keep only top-K largest predicted 2D components per slice. 0 disables.')
    p.add_argument('--tta',          action='store_true',
                   help='Average 8 augmented predictions (h/v flip + 90° × 4).')
    p.add_argument('--save_dir',     type=str,   default='',
                   help='If set, save the predicted probability map of each '
                        'sample as <save_dir>/<patient>_<idx>_s<slice>.png.')
    p.add_argument('--no_verbose',   action='store_true',
                   help='Per-sample metrics not printed.')
    p.add_argument('--cpu',          action='store_true', help='Force CPU.')
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Optional TTA
# --------------------------------------------------------------------------- #
def tta_forward(model, x1: torch.Tensor, x2: torch.Tensor) -> np.ndarray:
    """Average 8 augmented forward passes.  Returns a (B, 1, H, W) prob array."""
    B, _, H, W = x1.shape
    probs = torch.zeros(B, 1, H, W, device=x1.device, dtype=x1.dtype)
    n_views = 0
    for k in (0, 1, 2, 3):
        for flip in (False, True):
            x1a = torch.rot90(x1, k=k, dims=(-2, -1))
            x2a = torch.rot90(x2, k=k, dims=(-2, -1))
            if flip:
                x1a = torch.flip(x1a, dims=(-1,))
                x2a = torch.flip(x2a, dims=(-1,))
            with torch.no_grad():
                s1, s2, s3, sm = model(x1a, x2a)
            p = (torch.sigmoid(s1) + torch.sigmoid(s2) +
                 torch.sigmoid(s3) + torch.sigmoid(sm)) / 4.0
            # invert the geometric transformation
            p = torch.rot90(p, k=-k, dims=(-2, -1))
            if flip:
                p = torch.flip(p, dims=(-1,))
            probs += p
            n_views += 1
    probs /= n_views
    return probs


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    opt = parse_args()

    if not os.path.isfile(opt.ckpt):
        raise FileNotFoundError(f'Checkpoint not found: {opt.ckpt}')

    device = torch.device('cuda' if torch.cuda.is_available() and not opt.cpu else 'cpu')

    # ---- model ---- #
    model = CFANet(channel=64, dual_backbone=True).to(device)
    state = torch.load(opt.ckpt, map_location=device)
    # state can be either a raw state_dict or a dict with 'state_dict' / 'model'
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    elif isinstance(state, dict) and 'model' in state:
        state = state['model']
    msg = model.load_state_dict(state, strict=False)
    if not opt.no_verbose:
        print(f'[model] loaded {opt.ckpt}  missing={len(msg.missing_keys)}  '
              f'unexpected={len(msg.unexpected_keys)}')
    model.eval()

    # ---- data ---- #
    if opt.data_format == 'npy':
        from preprocessed_dataset import PreprocessedDataset
        test_ds = PreprocessedDataset(opt.data_root, split='test')
    else:
        test_ds = MedicalSliceDataset(
            root=opt.data_root, split='test', trainsize=opt.trainsize,
            augment=False, mask_combine=opt.mask_combine,
            resplit=opt.resplit, seed=opt.resplit_seed,
            train_ratio=opt.train_ratio, val_ratio=opt.val_ratio,
            test_ratio=opt.test_ratio)
    test_loader = DataLoader(test_ds, batch_size=opt.batchsize, shuffle=False,
                             num_workers=opt.num_workers, pin_memory=True)

    # ---- inference ---- #
    if opt.save_dir:
        os.makedirs(opt.save_dir, exist_ok=True)

    metric_keys = ('iou', 'dice', 'mae', 'precision', 'recall', 'f_measure')
    accum = {k: [] for k in metric_keys}
    foreground_dice = []
    n_pos_only = n_neg_only = 0
    t0 = time.time()

    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            x1 = batch['image_t1'].to(device, non_blocking=True)
            x2 = batch['image_t2'].to(device, non_blocking=True)
            gt = batch['mask']  # (B, 1, H, W) {0,1}

            if opt.tta:
                probs = tta_forward(model, x1, x2)
            else:
                s1, s2, s3, sm = model(x1, x2)
                probs = (torch.sigmoid(s1) + torch.sigmoid(s2) +
                         torch.sigmoid(s3) + torch.sigmoid(sm)) / 4.0

            pred_bin = (probs >= opt.threshold).cpu().numpy()
            gt_bin   = (gt    >= 0.5).cpu().numpy()

            for b in range(pred_bin.shape[0]):
                pb = pred_bin[b, 0]
                if opt.min_area > 0 or opt.keep_largest > 0:
                    pb = postprocess_2d(pb, min_area=opt.min_area, keep_largest=opt.keep_largest)
                m = compute_metrics(pb, gt_bin[b, 0])
                for k in metric_keys:
                    accum[k].append(m[k])
                if gt_bin[b, 0].sum() == 0:
                    n_neg_only += 1
                else:
                    n_pos_only += 1
                    foreground_dice.append(m['dice'])

                if opt.save_dir:
                    # probability map (rescaled 0..255) for visual inspection
                    p = probs[b, 0].cpu().numpy()
                    p8 = np.clip(p * 255, 0, 255).astype(np.uint8)
                    name = batch.get('filename', None)
                    if isinstance(name, (list, tuple)) and len(name) > b:
                        tag = str(name[b])
                    elif isinstance(name, torch.Tensor):
                        tag = f'sample_{step:04d}_{b:02d}'
                    else:
                        tag = f'sample_{step:04d}_{b:02d}'
                    cv2.imwrite(os.path.join(opt.save_dir, f'{tag}.png'), p8)

            if not opt.no_verbose and (step + 1) % 5 == 0:
                running = {k: float(np.mean(accum[k])) for k in metric_keys}
                print(f'  [step {step+1:3d}/{len(test_loader):3d}]  '
                      f'mIoU={running["iou"]:.4f}  Dice={running["dice"]:.4f}  '
                      f'Fβ={running["f_measure"]:.4f}  MAE={running["mae"]:.4f}')

    elapsed = time.time() - t0

    # ---- summary ---- #
    print()
    print('=' * 70)
    print(f'Checkpoint       : {opt.ckpt}')
    print(f'Test samples     : {len(test_ds)}  (pos={n_pos_only}, neg={n_neg_only})')
    print(f'TTA              : {opt.tta}    threshold={opt.threshold}')
    print(f'Elapsed          : {elapsed:.1f} s   ({elapsed/max(1,len(test_ds)):.3f} s/sample)')
    if foreground_dice:
        print(f'Foreground Dice  : {np.mean(foreground_dice):.4f}  '
              f'(foreground_mean_dice, pos-only, n={len(foreground_dice)})')
    else:
        print('Foreground Dice  : N/A  (no positive ground-truth slices)')
    print('-' * 70)
    print(f'{"metric":<12} {"mean":>10} {"std":>10} {"min":>10} {"max":>10}')
    print('-' * 70)
    for k in metric_keys:
        arr = np.asarray(accum[k], dtype=np.float64)
        print(f'{k:<12} {arr.mean():>10.4f} {arr.std():>10.4f} '
              f'{arr.min():>10.4f} {arr.max():>10.4f}')
    print('=' * 70)


if __name__ == '__main__':
    main()
