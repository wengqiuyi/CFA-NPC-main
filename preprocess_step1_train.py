import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, '/root/CFANet-main')
from data.dataset import MedicalSliceDataset

print('=== Step 1: Processing Training Dataset ===')

# 创建训练数据集
train_dataset = MedicalSliceDataset(
    root='/root/CFANet-main/data/traindata',
    split='train',
    trainsize=256,
    augment=False,
    crop_size=0,
    resplit=True,
    seed=42,
    train_ratio=0.8,
    val_ratio=0.2,
    test_ratio=0.0,
    require_pair=True
)

# 保存训练数据
save_dir = Path('/root/CFANet-main/TrainDataset/train')
save_dir.mkdir(parents=True, exist_ok=True)
(save_dir / 'images_t1').mkdir(exist_ok=True)
(save_dir / 'images_t2').mkdir(exist_ok=True)
(save_dir / 'masks').mkdir(exist_ok=True)

print(f'Saving {len(train_dataset)} training samples...')
for idx in range(len(train_dataset)):
    sample = train_dataset[idx]
    filename = f'sample_{idx:06d}'
    np.save(save_dir / 'images_t1' / f'{filename}.npy', sample['image_t1'].numpy())
    np.save(save_dir / 'images_t2' / f'{filename}.npy', sample['image_t2'].numpy())
    np.save(save_dir / 'masks' / f'{filename}.npy', sample['mask'].numpy())
    if (idx + 1) % 50 == 0:
        print(f'  Saved {idx + 1}/{len(train_dataset)}')

print('Training dataset saved successfully!')
