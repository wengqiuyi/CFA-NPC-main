import os
import sys
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# 导入 dataset 模块
sys.path.insert(0, '/root/CFANet-main')
from data.dataset import MedicalSliceDataset


def save_preprocessed_data(dataset, save_dir, split_name):
    """
    将预处理后的数据保存到指定目录
    
    Args:
        dataset: MedicalSliceDataset 实例
        save_dir: 保存目录
        split_name: 数据集分割名称（'train', 'val', 'test'）
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 保存图像和掩码
    images_t1_dir = save_path / 'images_t1'
    images_t2_dir = save_path / 'images_t2'
    masks_dir = save_path / 'masks'
    
    images_t1_dir.mkdir(exist_ok=True)
    images_t2_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)
    
    # 保存数据清单
    manifest = []
    
    print(f'\n开始保存 {split_name} 数据集...')
    for idx in tqdm(range(len(dataset)), desc=f'保存 {split_name}'):
        sample = dataset[idx]
        
        img_t1 = sample['image_t1']
        img_t2 = sample['image_t2']
        mask = sample['mask']
        
        # 生成文件名
        filename = f'sample_{idx:06d}'
        
        # 保存为 numpy 文件
        np.save(images_t1_dir / f'{filename}.npy', img_t1.numpy())
        np.save(images_t2_dir / f'{filename}.npy', img_t2.numpy())
        np.save(masks_dir / f'{filename}.npy', mask.numpy())
        
        # 添加到清单
        manifest.append({
            'filename': filename,
            'has_mask': bool(mask.sum() > 0)
        })
    
    # 保存清单
    import csv
    manifest_path = save_path / 'manifest.csv'
    with open(manifest_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'has_mask'])
        writer.writeheader()
        writer.writerows(manifest)
    
    print(f'{split_name} 数据集保存完成！共 {len(dataset)} 个样本')
    print(f'保存位置: {save_path}')


def main():
    # 设置参数
    traindata_root = '/root/CFANet-main/data/traindata'
    testdata_root = '/root/CFANet-main/data/testdata'
    train_save_dir = '/root/CFANet-main/TrainDataset'
    test_save_dir = '/root/CFANet-main/TestDataset'
    
    # 1. 处理训练和验证集（不做数据增强）
    print('=' * 60)
    print('步骤 1/3: 准备训练数据集（不增强）')
    print('=' * 60)
    
    train_dataset = MedicalSliceDataset(
        root=traindata_root,
        split='train',
        trainsize=256,
        augment=False,  # 保存时不做数据增强
        crop_size=0,
        resplit=True,
        seed=42,
        train_ratio=0.8,
        val_ratio=0.2,
        test_ratio=0.0,
        require_pair=True
    )
    
    save_preprocessed_data(train_dataset, os.path.join(train_save_dir, 'train'), 'train')
    
    print('\n' + '=' * 60)
    print('步骤 2/3: 准备验证数据集（不增强）')
    print('=' * 60)
    
    val_dataset = MedicalSliceDataset(
        root=traindata_root,
        split='val',
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
    
    save_preprocessed_data(val_dataset, os.path.join(train_save_dir, 'val'), 'val')
    
    # 2. 处理测试集
    print('\n' + '=' * 60)
    print('步骤 3/3: 准备测试数据集')
    print('=' * 60)
    
    test_dataset = MedicalSliceDataset(
        root=testdata_root,
        split='test',
        trainsize=256,
        augment=False,
        crop_size=0,
        resplit=False,  # 测试集不重新分割
        require_pair=True
    )
    
    save_preprocessed_data(test_dataset, test_save_dir, 'test')
    
    print('\n' + '=' * 60)
    print('所有数据预处理和保存完成！')
    print('=' * 60)


if __name__ == '__main__':
    main()
