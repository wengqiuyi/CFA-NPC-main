import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from lib.model import CFANet
from utils.dataloader import test_dataset


def compute_metrics(pred_bin, gt_bin, eps=1e-7):
    pred = pred_bin.astype(np.float32).ravel()
    gt = gt_bin.astype(np.float32).ravel()

    tp = float((pred * gt).sum())
    fp = float((pred * (1 - gt)).sum())
    fn = float(((1 - pred) * gt).sum())

    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    mae = float(np.mean(np.abs(pred - gt)))
    return {
        'iou': iou,
        'dice': dice,
        'mae': mae,
        'precision': precision,
        'recall': recall,
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Test CFANet on CVC-ClinicDB')
    parser.add_argument('--data_root', type=str, default='./CVC-ClinicDB/test')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--testsize', type=int, default=352)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--out_csv', type=str, default='',
                        help='Optional CSV path to save per-image metrics.')
    parser.add_argument('--cpu', action='store_true')
    return parser.parse_args()


def main():
    opt = parse_args()

    device = torch.device('cpu')
    if torch.cuda.is_available() and not opt.cpu:
        torch.cuda.set_device(opt.gpu)
        device = torch.device('cuda')

    model = CFANet(channel=64, dual_backbone=False).to(device)
    state = torch.load(opt.ckpt, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    elif isinstance(state, dict) and 'model' in state:
        state = state['model']
    msg = model.load_state_dict(state, strict=False)
    print(f'loaded {opt.ckpt}  missing={len(msg.missing_keys)}  unexpected={len(msg.unexpected_keys)}')
    model.eval()

    image_root = str(Path(opt.data_root) / 'images') + '/'
    gt_root = str(Path(opt.data_root) / 'masks') + '/'
    loader = test_dataset(image_root, gt_root, opt.testsize)

    if opt.save_dir:
        os.makedirs(opt.save_dir, exist_ok=True)

    metrics = {k: [] for k in ('iou', 'dice', 'mae', 'precision', 'recall')}
    per_image = []
    t0 = time.time()

    with torch.no_grad():
        for i in range(loader.size):
            image, gt, name = loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt = gt / (gt.max() + 1e-8)

            image = image.to(device)
            _, _, _, res = model(image)
            res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
            prob = torch.sigmoid(res).cpu().numpy().squeeze()
            pred_bin = (prob >= opt.threshold).astype(np.uint8)
            gt_bin = (gt >= 0.5).astype(np.uint8)

            m = compute_metrics(pred_bin, gt_bin)
            for k, v in m.items():
                metrics[k].append(v)
            per_image.append({
                'name': str(name),
                **{k: float(v) for k, v in m.items()},
            })

            if opt.save_dir:
                cv2.imwrite(str(Path(opt.save_dir) / name), (prob * 255).astype(np.uint8))

            print('[{:03d}/{:03d}] {} dice={:.4f} iou={:.4f} mae={:.4f}'.format(
                i + 1, loader.size, name, m['dice'], m['iou'], m['mae']))

    if opt.out_csv:
        out_path = Path(opt.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open('w', newline='', encoding='utf-8') as f:
            import csv
            w = csv.DictWriter(f, fieldnames=['name', 'dice', 'iou', 'mae', 'precision', 'recall'])
            w.writeheader()
            for r in sorted(per_image, key=lambda x: x['dice']):
                w.writerow(r)
        print(f'per_image_csv={out_path}')

    print('\n' + '=' * 69)
    print('CVC-ClinicDB test summary')
    print('=' * 69)
    for k, values in metrics.items():
        arr = np.asarray(values, dtype=np.float32)
        print('{:<10s} mean={:.4f} std={:.4f} min={:.4f} max={:.4f}'.format(
            k, float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max())))
    print('elapsed_sec={:.2f}'.format(time.time() - t0))


if __name__ == '__main__':
    main()
