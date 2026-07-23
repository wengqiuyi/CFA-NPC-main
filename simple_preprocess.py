import os
import sys
import numpy as np
import torch
from pathlib import Path

# 导入 dataset 模块
sys.path.insert(0, '/root/CFANet-main')
from data.dataset import MedicalSliceDataset


def process_dataset(root, save_dir, split, resplit=True):
    print(f'Processing {split} dataset from {root}...')
    
    # 创建 dataset
    if split == 'test':
        dataset = MedicalSliceDataset(
            root=root,
            split='test',
            trainsize=256,
            augment=False,
            crop_size=0,
            resplit=False,
            require_pair=True
        )
    else:
        dataset = MedicalSliceDataset(
            root=root,
            split=split,
            trainsize=256,
            augment=False,
            crop_size=0,
            resplit=resplit,
            seed=42,
            train_ratio=0.8,
            val_ratio=0.2,
            test_ratio=0.0,
            require_pair=True
        )
    
    # 创建保存目录
    save_path = Path(save_dir) / split
    save_path.mkdir(parents=True, exist_ok=True)
    (save_path / 'images_t1').mkdir(exist_ok=True)
    (save_path / 'images_t2').mkdir(exist_ok=True)
    (save_path / 'masks').mkdir(exist_ok=True)
    
    # 保存样本
    print(f'Saving {len(dataset)} samples...')
    for idx in range(len(dataset)):
        if idx % 100 == 0:
            print(f'  Processed {idx}/{len(dataset)}')
        
        sample = dataset[idx]
        filename = f'sample_{idx:06d}'
        
        np.save(save_path / 'images_t1' / f'{filename}.npy', sample['image_t1'].numpy())
        np.save(save_path / 'images_t2' / f'{filename}.npy', sample['image_t2'].numpy())
        np.save(save_path / 'masks' / f'{filename}.npy', sample['mask'].numpy())
    
    print(f'{split} dataset saved to {save_path}')
    return len(dataset)


if __name__ == '__main__':
    # 处理训练和验证集
    traindata_root = '/root/CFANet-main/data/traindata'
    train_save_dir = '/root/CFANet-main/TrainDataset'
    process_dataset(traindata_root, train_save_dir, 'train', resplit=True)
    process_dataset(traindata_root, train_save_dir, 'val', resplit=True)
    
    # 处理测试集
    testdata_root = '/root/CFANet-main/data/testdata'
    test_save_dir = '/root/CFANet-main/TestDataset'
    process_dataset(testdata_root, test_save_dir, 'test', resplit=False)
    
    print('All done!')
