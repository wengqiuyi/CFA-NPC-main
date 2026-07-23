# 数据预处理和保存说明

## 概述

本项目已经更新了 `data/dataset.py` 文件，添加了数据预处理和保存功能。

## 使用方法

### 1. 预处理并保存数据

运行以下命令来处理并保存所有数据集：

```bash
cd /root/CFANet-main
python data/dataset.py save
```

这将执行以下操作：
- 处理训练数据（80%）并保存到 `/root/CFANet-main/TrainDataset/train`
- 处理验证数据（20%）并保存到 `/root/CFANet-main/TrainDataset/val`
- 处理测试数据并保存到 `/root/CFANet-main/TestDataset/test`

### 2. 数据结构

保存后的目录结构如下：

```
/root/CFANet-main/
├── TrainDataset/
│   ├── train/
│   │   ├── images_t1/    # T1 图像（.npy 格式）
│   │   ├── images_t2/    # T2 图像（.npy 格式）
│   │   └── masks/        # 掩码标签（.npy 格式）
│   └── val/
│       ├── images_t1/
│       ├── images_t2/
│       └── masks/
└── TestDataset/
    └── test/
        ├── images_t1/
        ├── images_t2/
        └── masks/
```

### 3. 加载预处理后的数据

使用 `preprocessed_dataset.py` 中的 `PreprocessedDataset` 类来加载数据：

```python
from preprocessed_dataset import PreprocessedDataset

# 加载训练集
train_dataset = PreprocessedDataset('/root/CFANet-main/TrainDataset', 'train')

# 加载验证集
val_dataset = PreprocessedDataset('/root/CFANet-main/TrainDataset', 'val')

# 加载测试集
test_dataset = PreprocessedDataset('/root/CFANet-main/TestDataset', 'test')

# 获取样本
sample = train_dataset[0]
img_t1 = sample['image_t1']  # torch.Tensor, shape (3, 256, 256)
img_t2 = sample['image_t2']  # torch.Tensor, shape (3, 256, 256)
mask = sample['mask']        # torch.Tensor, shape (1, 256, 256)
```

## 预处理步骤说明

数据预处理包括以下步骤（在 `MedicalSliceDataset` 中实现）：

1. **数据加载**：从 NIfTI 格式加载 T1 和 T2 图像及其对应的掩码
2. **数据清洗**：检查并修复 NaN 和 Inf 值
3. **归一化**：
   - 百分位截断（0.5% - 99.5%）
   - Z-score 标准化
4. **标签融合**：使用 OR 操作融合 T1 和 T2 掩码
5. **尺寸调整**：调整图像大小到 256x256
6. **格式转换**：转换为 PyTorch 张量，单通道扩展为 3 通道

## 注意事项

- 保存时不进行数据增强（`augment=False`）
- 训练和验证集使用固定随机种子（42）进行 80:20 分割
- 所有图像都调整为 256x256 大小
