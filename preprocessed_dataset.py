"""
预处理后数据集的加载器
"""
import os
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset


class PreprocessedDataset(Dataset):
    """
    加载预处理后数据的 Dataset 类
    """
    def __init__(self, root_dir, split='train'):
        """
        Args:
            root_dir: 预处理数据的根目录
            split: 'train', 'val', 或 'test'
        """
        self.root_dir = Path(root_dir)
        self.split_dir = self.root_dir / split
        
        self.images_t1_dir = self.split_dir / 'images_t1'
        self.images_t2_dir = self.split_dir / 'images_t2'
        self.masks_dir = self.split_dir / 'masks'
        
        # 获取所有样本文件
        self.sample_files = sorted([f.stem for f in self.images_t1_dir.glob('*.npy')])
        self.is_positive = []
        for name in self.sample_files:
            mask = np.load(self.masks_dir / f'{name}.npy', mmap_mode='r')
            self.is_positive.append(bool(np.any(mask > 0.5)))
        
        n_pos = int(sum(self.is_positive))
        n_neg = int(len(self.sample_files) - n_pos)
        print(f'Loaded {split} dataset with {len(self.sample_files)} samples '
              f'(pos={n_pos}, neg={n_neg})')
    
    def __len__(self):
        return len(self.sample_files)

    @staticmethod
    def _ensure_25d_channels(arr):
        """
        Ensure image arrays are returned as 3xHxW.
        - new 2.5D exports already use 3xHxW
        - legacy exports may be HxW or 1xHxW
        """
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 2:
            return np.stack([arr, arr, arr], axis=0)
        if arr.ndim == 3 and arr.shape[0] == 1:
            return np.repeat(arr, 3, axis=0)
        if arr.ndim == 3 and arr.shape[0] == 3:
            return arr
        raise ValueError(f'Unsupported image array shape: {arr.shape}')
    
    def __getitem__(self, idx):
        filename = self.sample_files[idx]
        
        img_t1 = self._ensure_25d_channels(np.load(self.images_t1_dir / f'{filename}.npy'))
        img_t2 = self._ensure_25d_channels(np.load(self.images_t2_dir / f'{filename}.npy'))
        mask = np.asarray(np.load(self.masks_dir / f'{filename}.npy'), dtype=np.float32)
        if mask.ndim == 2:
            mask = mask[None, ...]
        elif mask.ndim != 3 or mask.shape[0] != 1:
            raise ValueError(f'Unsupported mask array shape: {mask.shape}')
        
        return {
            'image_t1': torch.from_numpy(img_t1),
            'image_t2': torch.from_numpy(img_t2),
            'mask': torch.from_numpy(mask),
            'seq': 'T1+T2',
            'filename': filename
        }

    def build_sample_weights(self, pos_weight=1.0):
        """Return per-sample weights for small-target positive-slice oversampling."""
        pos_weight = max(1.0, float(pos_weight))
        weights = [pos_weight if flag else 1.0 for flag in self.is_positive]
        return torch.tensor(weights, dtype=torch.double)


def main():
    """测试预处理数据加载"""
    print('Testing preprocessed dataset loading...')
    
    # 测试 TrainDataset
    train_dataset = PreprocessedDataset('/root/CFANet-main/TrainDataset', 'train')
    print(f'Train dataset size: {len(train_dataset)}')
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print(f'Sample - image_t1 shape: {sample["image_t1"].shape}')
        print(f'Sample - image_t2 shape: {sample["image_t2"].shape}')
        print(f'Sample - mask shape: {sample["mask"].shape}')
    
    print()
    
    # 测试 TestDataset
    test_dataset = PreprocessedDataset('/root/CFANet-main/TestDataset', 'test')
    print(f'Test dataset size: {len(test_dataset)}')
    if len(test_dataset) > 0:
        sample = test_dataset[0]
        print(f'Sample - image_t1 shape: {sample["image_t1"].shape}')


if __name__ == '__main__':
    main()
