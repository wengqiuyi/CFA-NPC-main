"""
3D MRI -> 2.5D 切片样本预处理 Pipeline
========================================

适用于头颈部淋巴结（NPC）等小目标分割任务的数据准备。
仅依赖 numpy + scipy.ndimage，不引入 PyTorch / TensorFlow。

核心流程
--------
  1. spacing-aware 重采样到各向同性（默认 1.0 x 1.0 x 1.0 mm）
  2. 非背景体素的 Z-score 强度标准化 → clip 到 [-3, 3]
  3. 以每个切片为中心，取 ±context 层做 2.5D 堆叠（边界用最近有效层填充）
  4. 对每个样本标注 is_positive（后续训练时可据此做加权采样 / 保留空样本）
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import zoom


def _resample_volume(
    volume: np.ndarray,
    original_spacing: Sequence[float],
    target_spacing: Sequence[float],
    order: int,
) -> np.ndarray:
    """重采样一个 3D 体积，由 zoom_factors = original_spacing / target_spacing 驱动。

    注意：
      当 original_spacing 与 target_spacing 完全相同时直接返回原数组，节省时间。
    """
    orig = np.asarray(original_spacing, dtype=np.float64)
    tgt = np.asarray(target_spacing, dtype=np.float64)
    if orig.shape != (3,) or tgt.shape != (3,):
        raise ValueError("spacing must be 3-tuple of (sx, sy, sz)")
    if np.allclose(orig, tgt, rtol=1e-4, atol=1e-6):
        return volume.astype(np.float32 if order >= 1 else volume.dtype, copy=False)
    factors = orig / tgt
    factors = np.clip(factors, 1e-3, 1e3)
    out = zoom(volume, zoom=factors, order=order, mode="nearest")
    if order >= 1:
        out = out.astype(np.float32, copy=False)
    return out


def preprocess_mri_to_25d(
    volume: np.ndarray,
    mask: np.ndarray,
    spacing: Sequence[float],
    modality: str = "T1",
    target_spacing: Sequence[float] = (1.0, 1.0, 1.0),
    context: int = 2,
) -> List[Dict[str, object]]:
    """将一个 3D 配对 MRI 体积 + 二值 mask 转换为 2.5D 切片样本列表。

    Parameters
    ----------
    volume : np.ndarray
        3D MRI 体积，形状 ``(D, H, W)``，dtype 为 float32 或 int16 等数字类型。
        D 为切片方向（通常是 z 轴 / 相位编码方向）。
    mask : np.ndarray
        与 volume 同形状的二值分割标签，值为 0（背景）或 1（前景 / 淋巴结）。
        dtype 不限，但会在内部二值化为 uint8。
    spacing : tuple of (sx, sy, sz)
        原始 MRI 在 x, y, z 三个方向的物理间距，单位 mm。
        注意: (sx, sy) 对应面内像素间距，sz 对应层厚（通常 sz >> sx/sy）。
    modality : {'T1', 'T2', 'T1CE'}, optional
        仅用于文档/可扩展性标记。当前所有模态使用相同的强度标准化策略（非背景
        体素 Z-score）。对于 T1CE 可以很容易地在后续扩展为不同百分位截断等。
    target_spacing : tuple of (tx, ty, tz), optional
        目标重采样间距。各向同性时为 (1.0, 1.0, 1.0)。
    context : int, optional
        2.5D 切片在中心切片两侧各自取的邻域层数。默认值 2 对应通道数 = 2*context+1 = 5。

    Returns
    -------
    List[Dict[str, object]]
        列表中的每个元素描述一个 2.5D 切片样本，包含如下键:

        - ``'image'``      : np.ndarray, ``(C, H, W)`` float32, C = 2*context+1。
        - ``'mask'``       : np.ndarray, ``(H, W)`` float32, 对应中心切片的 2D 标签。
        - ``'slice_idx'``  : int, 该样本在 **重采样后** 体积中的中心层索引。
        - ``'is_positive'``: bool, mask 中是否存在前景（>0.5 的像素）。
        - ``'spacing'``    : Tuple[float,float,float], 实际使用的 spacing（重采样后
                             一般等于 target_spacing，或跳过重采样时等于原 spacing）。

    Notes
    -----
    * **为什么对 mask 使用最近邻插值 (order=0) 而不是线性插值 (order=1)?**

      线性插值会把 0/1 二值标签变成 0 和 1 之间的连续灰度（"半前景"像素），
      这在语义分割训练中是不合法的：标签必须是离散的语义类别。即便之后再做阈值
      化，线性插值也会**伪造出原本不存在的假阳性像素**，或者把边缘本来锐利的小
      目标（头颈部淋巴结通常只有几十到几百个体素）侵蚀 / 扩张，进而让 Dice /
      IoU 指标虚高或虚低。最近邻插值虽然边缘不"光滑"，但严格保证标签空间是
      {0,1}，是分割任务重采样的 gold standard。

    * **为什么空样本 (is_positive=False) 被保留而不是直接删除?**

      1. 背景建模：在极端类别不平衡场景（淋巴结可能只占全身体积的 <0.01%），
         如果训练数据中只有"含病灶切片"，神经网络会对背景的正常解剖结构产生
         严重过拟合，测试时在正常组织上涌现大量假阳性（这也是之前 Mean Dice
         << Foreground Dice 的根本原因）。
      2. 负样本挖掘 / Hard Negative Mining：保留的空样本后续可以通过
         WeightedRandomSampler、focal loss 或课程学习被选择性地用更高权重或
         在 epoch 后期喂给模型，而不需要在预处理阶段做不可逆的删除。
      3. 评估一致性：在 inference 时我们需要对整张 3D 体积（含大量空切片）做
         逐切片预测，如果训练时从未见过空切片，模型在空切片上的输出阈值行为
         会非常不稳定。
    """
    # ------------------------------------------------------------------ basic sanity
    if volume.ndim != 3:
        raise ValueError(f"volume must be 3D (D,H,W), got shape {volume.shape}")
    if mask.shape != volume.shape:
        raise ValueError(f"mask shape {mask.shape} must match volume shape {volume.shape}")
    if context < 0:
        raise ValueError("context must be >= 0")
    modality_up = str(modality).upper()
    if modality_up not in ("T1", "T2", "T1CE"):
        raise ValueError(f"Unknown modality {modality}. Expected T1/T2/T1CE.")

    vol = volume.astype(np.float32, copy=False)
    msk = np.asarray(mask > 0.5, dtype=np.uint8)  # enforce binary

    # ------------------------------------------------------------------ 1) resample
    vol_iso = _resample_volume(vol, spacing, target_spacing, order=1)  # linear
    # Why mask -> order=0? See docstring above.
    msk_iso = _resample_volume(msk, spacing, target_spacing, order=0)
    msk_iso = np.asarray(msk_iso > 0.5, dtype=np.uint8)  # post-clamp binary after zoom
    actual_spacing: Tuple[float, float, float]
    if np.allclose(np.asarray(spacing, dtype=np.float64),
                   np.asarray(target_spacing, dtype=np.float64),
                   rtol=1e-4, atol=1e-6):
        actual_spacing = tuple(float(x) for x in spacing)  # type: ignore[assignment]
    else:
        actual_spacing = tuple(float(x) for x in target_spacing)  # type: ignore[assignment]

    D, H, W = vol_iso.shape

    # ------------------------------------------------------------------ 2) intensity norm (non-background voxels only)
    fg = vol_iso > 0.0
    if fg.any():
        mean = float(vol_iso[fg].mean())
        std = float(vol_iso[fg].std())
    else:
        mean, std = 0.0, 1.0
    normed = (vol_iso - mean) / (std + 1e-8)
    normed = np.clip(normed, -3.0, 3.0).astype(np.float32, copy=False)

    # ------------------------------------------------------------------ 3) 2.5D slice build (nearest valid slice padding at boundaries)
    samples: List[Dict[str, object]] = []
    for i in range(D):
        channel_idxs: List[int] = []
        for off in range(-context, context + 1):
            j = i + off
            # 最近有效层填充：越界时 clamp 到边界索引
            j_c = max(0, min(D - 1, j))
            channel_idxs.append(j_c)
        img_c = np.stack([normed[jj] for jj in channel_idxs], axis=0)  # (C, H, W)
        mask_c = msk_iso[i].astype(np.float32, copy=False)  # (H, W)
        is_pos = bool(mask_c.sum() > 0)
        samples.append(
            {
                "image": img_c,
                "mask": mask_c,
                "slice_idx": int(i),
                "is_positive": is_pos,
                "spacing": actual_spacing,
            }
        )
    return samples


# ---------------------------------------------------------------------------
# Example usage block (直接 `python mri_25d_pipeline.py` 可运行)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.RandomState(0)

    # 模拟一个典型头颈部 MRI：面内 0.5mm, 层厚 3mm, 体积 40x512x512
    sx, sy, sz = 0.5, 0.5, 3.0
    D, H, W = 40, 256, 256
    volume = rng.randn(D, H, W).astype(np.float32) * 100 + 500
    volume[volume < 400] = 0.0  # 模拟大量背景=0 的 MRI

    mask = np.zeros((D, H, W), dtype=np.uint8)
    # 模拟 2 处小淋巴结病灶（10x10 像素、跨 3 层）
    mask[15:18, 100:110, 120:130] = 1
    mask[25:27, 180:190, 80:90] = 1

    samples = preprocess_mri_to_25d(
        volume,
        mask,
        spacing=(sx, sy, sz),
        modality="T1",
        target_spacing=(1.0, 1.0, 1.0),
        context=2,
    )

    n_total = len(samples)
    n_pos = sum(1 for s in samples if s["is_positive"])
    print(f"[example] total slices after resample: {n_total}")
    print(f"[example] positive slices            : {n_pos}")
    print(f"[example] negative (empty) slices    : {n_total - n_pos}")
    print(f"[example] sample[0]['image'].shape   : {samples[0]['image'].shape}  (C,H,W)")
    print(f"[example] sample[0]['mask'].shape    : {samples[0]['mask'].shape}  (H,W)")
    print(f"[example] sample[0]['is_positive']   : {samples[0]['is_positive']}")
    print(f"[example] sample[0]['spacing']       : {samples[0]['spacing']}")
    first_pos = next((s for s in samples if s["is_positive"]), None)
    if first_pos is not None:
        print(f"[example] first positive @ slice_idx = {first_pos['slice_idx']}, "
              f"mask foreground px = {float(first_pos['mask'].sum()):.0f}")
