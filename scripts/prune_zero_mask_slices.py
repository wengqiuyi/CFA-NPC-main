#!/usr/bin/env python3
import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description='Randomly move a portion of all-zero mask slices to backup.')
    p.add_argument('--dataset-root', required=True,
                   help='Root like /root/CFA-NPC-main-main/TrainDataset_spacing')
    p.add_argument('--split', default='train', choices=['train', 'val', 'test'],
                   help='Only this split will be pruned.')
    p.add_argument('--drop-ratio', type=float, default=0.5,
                   help='Fraction of all-zero mask samples to move away.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--mask-threshold', type=float, default=0.5)
    p.add_argument('--action', type=str, default='move', choices=['move', 'delete'],
                   help='move: move selected samples to backup (default). '
                        'delete: permanently delete selected samples (frees disk space).')
    p.add_argument('--backup-name', type=str, default='',
                   help='Optional backup folder name under dataset root.')
    return p.parse_args()


def main():
    opt = parse_args()
    dataset_root = Path(opt.dataset_root)
    split_root = dataset_root / opt.split
    masks_dir = split_root / 'masks'
    if not masks_dir.is_dir():
        raise FileNotFoundError(f'masks dir not found: {masks_dir}')

    sample_names = sorted(p.stem for p in masks_dir.glob('*.npy'))
    zero_names = []
    pos_names = []
    for name in sample_names:
        mask = np.load(masks_dir / f'{name}.npy', mmap_mode='r')
        if np.any(mask > opt.mask_threshold):
            pos_names.append(name)
        else:
            zero_names.append(name)

    rng = random.Random(opt.seed)
    zero_pick = list(zero_names)
    rng.shuffle(zero_pick)
    remove_count = int(round(len(zero_pick) * float(opt.drop_ratio)))
    remove_names = sorted(zero_pick[:remove_count])

    backup_root = None
    if opt.action == 'move':
        backup_name = opt.backup_name or f'_removed_zero_masks_{opt.split}_seed{opt.seed}_drop{int(round(opt.drop_ratio * 100))}'
        backup_root = dataset_root / backup_name / opt.split
        for sub in ('images_t1', 'images_t2', 'masks'):
            (backup_root / sub).mkdir(parents=True, exist_ok=True)

    for name in remove_names:
        for sub in ('images_t1', 'images_t2', 'masks'):
            src = split_root / sub / f'{name}.npy'
            if src.exists():
                if opt.action == 'delete':
                    src.unlink()
                else:
                    dst = backup_root / sub / src.name
                    shutil.move(str(src), str(dst))

    report = {
        'dataset_root': str(dataset_root),
        'split': opt.split,
        'seed': opt.seed,
        'drop_ratio': opt.drop_ratio,
        'mask_threshold': opt.mask_threshold,
        'action': opt.action,
        'total_samples_before': len(sample_names),
        'positive_samples_before': len(pos_names),
        'zero_mask_samples_before': len(zero_names),
        'removed_zero_mask_samples': len(remove_names),
        'remaining_samples_after': len(sample_names) - len(remove_names),
        'backup_root': str(backup_root) if backup_root is not None else '',
        'removed_files': remove_names,
    }
    report_path = (backup_root.parent / 'report.json') if backup_root is not None else (dataset_root / 'prune_report.json')
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
