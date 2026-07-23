import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, '/root/CFANet-main')
from data.dataset import MedicalSliceDataset

print('=== Step 3: Processing Test Dataset ===')

# 创建测试数据集
test_dataset = MedicalSliceDataset(
    root='/root/CFANet-main/data/testdata',
    split='test',
    trainsize=256,
    augment=False,
    crop_size=0,
    resplit=False,
    require_pair=True
)

# 保存测试数据
save_dir = Path('/root/CFANet-main/TestDataset/test')
save_dir.mkdir(parents=True, exist_ok=True)
(save_dir / 'images_t1').mkdir(exist_ok=True)
(save_dir / 'images_t2').mkdir(exist_ok=True)
(save_dir / 'masks').mkdir(exist_ok=True)

print(f'Saving {len(test_dataset)} test samples...')
for idx in range(len(test_dataset)):
    sample = test_dataset[idx]
    filename = f'sample_{idx:06d}'
    np.save(save_dir / 'images_t1' / f'{filename}.npy', sample['image_t1'].numpy())
    np.save(save_dir / 'images_t2' / f'{filename}.npy', sample['image_t2'].numpy())
    np.save(save_dir / 'masks' / f'{filename}.npy', sample['mask'].numpy())
    if (idx + 1) % 50 == 0:
        print(f'  Saved {idx + 1}/{len(test_dataset)}')

print('Test dataset saved successfully!')
