"""
preprocessed_dataset.py
=======================

PyTorch Dataset for the preprocessed .npy slices produced by
``scripts/preprocess_dataset_stratified.py`` (or compatible npy layouts).

Layout consumed
---------------
<root>/<split>/
  images_t1/sample_XXXXXX.npy   float32 shape (C, H, W), C usually = 3 (2.5D stack)
  images_t2/sample_XXXXXX.npy   float32 shape (C, H, W)
  masks/sample_XXXXXX.npy       float32 shape (1 or C, H, W) or (H, W)

Augmentation
------------
- ``augment=False`` (val/test): only optional central crop / no-op.
- ``augment=True``  (train): a dedicated **small-target aware** albumentations
  pipeline is applied, exactly matching the spec requested by the user for
  tiny head-and-neck lymph-node segmentation:

  1. Geometric transforms (image + mask, synchronised):
     - HorizontalFlip                      p=0.5
     - ShiftScaleRotate shift=0.1 / scale=0.1 / rotate=±15°  p=0.5
     - ElasticTransform  alpha=1 sigma=50 alpha_affine=50   p=0.3

  2. Intensity transforms (image only, NEVER touches mask):
     - RandomBrightnessContrast ±10%       p=0.3
     - GaussNoise  var_limit=(0.001, 0.01) p=0.3

  3. ROI-aware crop (last step, p=1.0):
     - if ``crop_size <= 0``  -> no crop, keep original resolution.
     - if mask has foreground -> crop around a randomly-chosen foreground pixel
       with probability ``ROI_CENTER_RATIO = 0.7``; remaining 30% -> pure
       random crop (avoids the model learning a "target always at the
       centre" position bias).
     - if mask is all zero    -> pure random crop (no crash).
     - if input H/W < crop_size the sample is first zero-padded to crop_size.

  See ``_build_albumentations_transform`` docstring for the "why" comments
  about each hyper-parameter choice for small targets.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import albumentations as A
    from albumentations import DualTransform  # type: ignore
    _ALB_OK = True
    _ALB_ERR: Optional[Exception] = None
except Exception as _e:
    _ALB_OK = False
    _ALB_ERR = _e
    A = None  # type: ignore
    DualTransform = object  # type: ignore


# ---------------------------------------------------------------------------
# Custom Albumentations transforms for ROI-aware crop + optional pad
# ---------------------------------------------------------------------------
if _ALB_OK:
    class _PadIfSmaller(DualTransform):
        def __init__(self, height: int, width: int, pad_value: float = 0.0,
                     mask_pad_value: int = 0, always_apply: bool = True, p: float = 1.0):
            super().__init__(always_apply, p)
            self.height = int(height)
            self.width = int(width)
            self.pad_value = float(pad_value)
            self.mask_pad_value = int(mask_pad_value)

        def apply(self, img: np.ndarray, **params) -> np.ndarray:
            H, W = img.shape[:2]
            ph, pw = max(0, self.height - H), max(0, self.width - W)
            if ph == 0 and pw == 0:
                return img
            pt, pb = ph // 2, ph - ph // 2
            pl, pr = pw // 2, pw - pw // 2
            if img.ndim == 2:
                return np.pad(img, ((pt, pb), (pl, pr)), mode="constant",
                              constant_values=self.pad_value)
            return np.pad(img, ((pt, pb), (pl, pr), (0, 0)), mode="constant",
                          constant_values=self.pad_value)

        def apply_to_mask(self, img: np.ndarray, **params) -> np.ndarray:
            H, W = img.shape[:2]
            ph, pw = max(0, self.height - H), max(0, self.width - W)
            if ph == 0 and pw == 0:
                return img
            pt, pb = ph // 2, ph - ph // 2
            pl, pr = pw // 2, pw - pw // 2
            if img.ndim == 2:
                return np.pad(img, ((pt, pb), (pl, pr)), mode="constant",
                              constant_values=self.mask_pad_value)
            return np.pad(img, ((pt, pb), (pl, pr), (0, 0)), mode="constant",
                          constant_values=self.mask_pad_value)

        def apply_to_bbox(self, bbox, **params):  # pragma: no cover
            raise NotImplementedError

        def apply_to_keypoint(self, keypoint, **params):  # pragma: no cover
            return keypoint

        def get_transform_init_args_names(self):
            return ("height", "width", "pad_value", "mask_pad_value")


    class _RoiAwareRandomCrop(DualTransform):
        """Foreground-centred random crop with a fallback for empty masks."""

        def __init__(self, height: int, width: int, roi_center_ratio: float = 0.7,
                     always_apply: bool = False, p: float = 1.0):
            super().__init__(always_apply, p)
            if height <= 0 or width <= 0:
                raise ValueError("height/width must be > 0")
            if not (0.0 <= roi_center_ratio <= 1.0):
                raise ValueError("roi_center_ratio must lie in [0,1]")
            self.height = int(height)
            self.width = int(width)
            self.roi_center_ratio = float(roi_center_ratio)

        @property
        def targets_as_params(self):
            return ["image", "mask"]

        def get_params_dependent_on_targets(self, params: Dict) -> Dict:
            H, W = params["image"].shape[:2]
            ch, cw = self.height, self.width
            y_max = max(H - ch, 0)
            x_max = max(W - cw, 0)
            mask = params.get("mask")
            has_fg = False
            if mask is not None:
                m = np.asarray(mask)
                has_fg = bool(m.size and m.sum() > 0.5)

            use_roi = has_fg and (float(np.random.rand()) < self.roi_center_ratio)
            if use_roi:
                ys, xs = np.where(np.asarray(mask) > 0.5)
                if ys.size == 0:
                    use_roi = False
                else:
                    k = int(np.random.randint(0, ys.size))
                    cy, cx = int(ys[k]), int(xs[k])
                    jy = int(np.random.randint(-max(1, ch // 4), max(1, ch // 4) + 1))
                    jx = int(np.random.randint(-max(1, cw // 4), max(1, cw // 4) + 1))
                    y0 = int(np.clip(cy - ch // 2 + jy, 0, y_max))
                    x0 = int(np.clip(cx - cw // 2 + jx, 0, x_max))
            if not use_roi:
                y0 = 0 if y_max <= 0 else int(np.random.randint(0, y_max + 1))
                x0 = 0 if x_max <= 0 else int(np.random.randint(0, x_max + 1))
            return {"x_min": x0, "x_max": x0 + cw, "y_min": y0, "y_max": y0 + ch}

        def apply(self, img, x_min=0, y_min=0, x_max=0, y_max=0, **params):
            return img[y_min:y_max, x_min:x_max]

        def apply_to_mask(self, img, x_min=0, y_min=0, x_max=0, y_max=0, **params):
            return img[y_min:y_max, x_min:x_max]

        def apply_to_bbox(self, bbox, **params):  # pragma: no cover
            raise NotImplementedError

        def apply_to_keypoint(self, kp, x_min=0, y_min=0, **params):  # pragma: no cover
            x, y, a, s = kp
            return (x - x_min, y - y_min, a, s)

        def get_transform_init_args_names(self):
            return ("height", "width", "roi_center_ratio")


def _build_albumentations_transform(crop_size: int, seed: Optional[int] = None,
                                    roi_center_ratio: float = 0.7
                                    ) -> Optional["A.Compose"]:
    """Build the train-time small-target aware augmentation pipeline.

    Design notes (matching user's spec exactly)
    --------------------------------------------

    1. Why ``ElasticTransform(alpha=1, sigma=50, alpha_affine=50)`` is
       "gentle" for tiny targets?

       - ``alpha`` scales the *magnitude* of the random displacement field.
       - ``sigma`` controls its *spatial smoothness* (higher = more
         low-frequency, less jagged).

       With alpha=1 / sigma=50, each voxel shifts by much less than one
       pixel locally, so a small lymph-node blob of ~20x20 pixels is
       smoothly stretched / squeezed instead of torn.

       If we instead used ``alpha=100 / sigma=50``, the local displacement
       would be ~100x larger: the ~20 pixel target could be split across
       several displaced regions, or even moved entirely out of the later
       256x256 crop window, which makes Dice / IoU meaningless on that
       sample (equivalent to label corruption).

    2. Why keep ``1 - roi_center_ratio = 0.3`` pure random crops instead of
       always centering the crop on the foreground?

       100% ROI-centred cropping teaches the network that the target
       *always* lives near the image centre -> a strong **position bias**.
       At inference, head-and-neck lymph nodes can live anywhere from the
       high jugular chain down to the supraclavicular fossa (edges of the
       field-of-view are common); a position-biased model would under-detect
       them catastrophically. Keeping 30% random crops is a cheap regulariser
       that forces the decoder to generalise spatially.

    3. Why ``GaussNoise var_limit=(0.001, 0.01)`` works well after z-score?

       Images flowing into this transform have already been normalised via
       a non-background-voxel z-score and clipped to [-3,3], so 1 std dev
       in the image corresponds numerically to ~1.0.

       - sqrt(0.001) ~ 0.032 std  -> faint thermal noise (low-noise MRI).
       - sqrt(0.01)  ~ 0.10  std  -> clearly visible noise (fast / noisy
         acquisition, typical for T2/STIR sequences used to spot nodes).

       Lymph-node contrast after z-score is typically in the range
       0.5 ~ 1.5 std above muscle. Raising ``var_limit`` to 0.1 (0.316 std)
       or more would start drowning the lesion signal under noise ->
       equivalent to random erasing on the positive class, which makes early
       training diverge (foreground loss explodes).
    """
    if not _ALB_OK:
        raise RuntimeError(
            "albumentations is required for training augmentations. Install it "
            f"via `pip install albumentations`. Import err: {_ALB_ERR!r}"
        )
    if crop_size <= 0:
        # Apply geometry + intensity but skip any cropping.
        crop_steps: List = []
    else:
        crop_steps = [
            _PadIfSmaller(height=crop_size, width=crop_size,
                          pad_value=0.0, mask_pad_value=0),
            _RoiAwareRandomCrop(height=crop_size, width=crop_size,
                                roi_center_ratio=roi_center_ratio, p=1.0),
        ]
    if seed is not None:
        try:
            A.set_seed(int(seed))
        except Exception:
            pass
        np.random.seed(int(seed))

    geometric = [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            interpolation=1,       # bilinear for image; mask uses mask_interpolation=0 (nearest)
            border_mode=0,         # cv2.BORDER_CONSTANT = 0
            fill=0.0,
            fill_mask=0,
            p=0.5,
        ),
        # alpha=1, sigma=50  -> mild smooth displacement (see docstring above).
        # alpha_affine / value / mask_value are dropped in albumentations 2.x,
        # use 'fill' / 'fill_mask' instead.
        A.ElasticTransform(
            alpha=1.0,
            sigma=50.0,
            interpolation=1,
            border_mode=0,
            fill=0.0,
            fill_mask=0,
            p=0.3,
        ),
    ]
    intensity = [
        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            brightness_by_max=False,
            p=0.3,
        ),
        # albumentations 2.x: GaussNoise takes 'std_range' (not variance).
        # User requested var_limit=(0.001, 0.01) -> sqrt -> (0.0316, 0.10).
        A.GaussNoise(std_range=(0.0316, 0.100), mean_range=(0.0, 0.0), p=0.3),
    ]
    return A.Compose(geometric + intensity + crop_steps,
                     additional_targets={"mask": "mask"})


def _list_npy_files(folder: Path) -> List[Path]:
    return sorted([p for p in folder.glob("*.npy") if p.is_file()])


def _intersect_names(t1_files: List[Path], t2_files: List[Path],
                     m_files: List[Path]) -> List[str]:
    s1 = {p.stem for p in t1_files}
    s2 = {p.stem for p in t2_files}
    sm = {p.stem for p in m_files}
    return sorted(s1 & s2 & sm)


def _find_patient_volumes(root: Path, patient_id: str):
    """Locate cached per-patient 3-D volumes saved by preprocess_dataset_stratified.

    Returns (t1_vol_path or None, t2_vol_path or None, mask_vol_path or None).
    The vols are saved under ``<root>/<split>/_volumes/<patient_id>_t1.npy`` etc.
    """
    vol_dir = root / "_volumes"
    if not vol_dir.is_dir():
        # Try <root>/<split>/_volumes (root = root/<split> layout)
        return (None, None, None)
    t1 = vol_dir / f"{patient_id}_t1.npy"
    t2 = vol_dir / f"{patient_id}_t2.npy"
    mk = vol_dir / f"{patient_id}_mask.npy"
    return (
        str(t1) if t1.is_file() else None,
        str(t2) if t2.is_file() else None,
        str(mk) if mk.is_file() else None,
    )


# ========================================================================= #
# Strategy A — Foreground-aware (ROI) Patch Cropping  (pure numpy, no torch,
# no albumentations dependency).  Directly matches the user's spec.
# ========================================================================= #
def roi_aware_crop(
    volume: np.ndarray,
    mask: np.ndarray,
    patch_size: Tuple[int, int] = (256, 256),
    ratio: float = 0.7,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """小目标专用的 ROI 感知裁剪（策略 A）。 **不要纯随机裁剪** —— 淋巴结极小，
    纯随机裁剪几乎一定会把目标裁掉。

    Parameters
    ----------
    volume : np.ndarray
        2.5D 切片图像，形状 ``(C, H, W)`` 或 ``(H, W)``。T1/T2 可分别调用两次本
        函数，或在同 ``rng`` + 同 mask 顺序下调用两次保证裁剪区域完全一致。
    mask : np.ndarray
        对应中心层的二维分割标签，形状 ``(H, W)``，值 0/1。
    patch_size : (int, int)
        (patch_h, patch_w)，通常 (256, 256)。
    ratio : float
        当 mask 中存在前景像素时，**以目标为中心进行裁剪** 的概率（默认 0.7，
        即 70% ROI 裁剪 / 30% 纯随机裁剪，后者用于维持背景建模能力）。
    rng : numpy RandomState, optional
        外部随机状态；当 T1/T2 要共享完全相同的裁剪坐标时传入同一个 rng。

    Returns
    -------
    (cropped_volume, cropped_mask, meta)
        cropped_volume: 形状 ``(C, patch_h, patch_w)`` 或 ``(patch_h, patch_w)``
        cropped_mask  : 形状 ``(patch_h, patch_w)``
        meta          : ``{'kind': 'roi'|'random', 'y0': int, 'x0': int,
                           'y1': int, 'x1': int, 'center_used': (cy, cx) or None}``

    Notes
    -----
    * 小目标为什么不能用普通 RandomCrop？
      设淋巴结占 15×15 = 225 px，原图 512×512 = 262144 px。
      若完全随机取 256×256 裁剪框，则裁剪框 **完全不覆盖淋巴结** 的概率约为
      ``(1 - (256+15)/512)^2 ≈ (1-0.529)^2 ≈ 0.22``，加上被边缘裁断的情况，
      实际 batch 里"目标近乎消失"的比例轻松 > 50%，等效于把正样本当负样本
      训练，foreground Dice 永远上不去。

    * 为什么仍要保留 1-ratio = 30% 的纯随机裁剪？
      100% 把目标硬塞到裁剪中心会教会模型"病灶必须在图片正中央"——产生严重的
      **位置偏置 (position bias)**，推理时边缘病灶（如颌下、锁骨上淋巴结正
      好靠近扫描 FOV 边缘）的召回率会暴跌。保留少量随机裁剪是低成本正则化。
    """
    ratio = min(max(float(ratio), 0.0), 1.0)
    ph, pw = int(patch_size[0]), int(patch_size[1])
    if ph <= 0 or pw <= 0:
        raise ValueError("patch_size must be (h, w) positive ints")
    if rng is None:
        rng = np.random  # type: ignore[assignment]

    # Normalise input shapes (volume may be 2D or CHW; mask must be 2D)
    m = np.asarray(mask)
    if m.ndim != 2:
        raise ValueError(f"mask must be (H,W), got shape {m.shape}")
    H, W = m.shape
    if ph > H or pw > W:
        # zero-pad up to patch_size first (mirroring PadIfSmaller behaviour above)
        pad_h_t = max(0, (ph - H) // 2)
        pad_h_b = max(0, (ph - H) - pad_h_t)
        pad_w_l = max(0, (pw - W) // 2)
        pad_w_r = max(0, (pw - W) - pad_w_l)
        if volume.ndim == 2:
            v_padded = np.pad(np.asarray(volume),
                              ((pad_h_t, pad_h_b), (pad_w_l, pad_w_r)),
                              mode="constant", constant_values=0.0)
        else:
            v_padded = np.pad(np.asarray(volume),
                              ((0, 0), (pad_h_t, pad_h_b), (pad_w_l, pad_w_r)),
                              mode="constant", constant_values=0.0)
        m_padded = np.pad(m, ((pad_h_t, pad_h_b), (pad_w_l, pad_w_r)),
                          mode="constant", constant_values=0)
        y_off = pad_h_t
        x_off = pad_w_l
        H2, W2 = m_padded.shape
    else:
        v_padded = np.asarray(volume)
        m_padded = m
        y_off = 0
        x_off = 0
        H2, W2 = H, W

    y_max = max(H2 - ph, 0)
    x_max = max(W2 - pw, 0)
    has_fg = bool(m_padded.sum() > 0.5)
    use_roi = has_fg and (float(rng.rand()) < ratio)  # type: ignore[union-attr]

    center_used: Optional[Tuple[int, int]] = None
    if use_roi:
        coords = np.argwhere(m_padded > 0.5)
        # 从所有前景像素中随机挑一个作为裁剪锚点（而不是 bbox 中心，能更好地
        # 覆盖病灶边缘、增加多样性，同时避免 ROI 只覆盖病灶中心的过拟合）。
        pick = int(rng.randint(0, len(coords)))  # type: ignore[union-attr]
        cy, cx = int(coords[pick, 0]), int(coords[pick, 1])
        center_used = (cy, cx)
        # 在 patch_h/2 附近引入 ± patch/4 的抖动，避免 ROI 永远居中
        jitter_y = int(rng.randint(-max(1, ph // 4), max(1, ph // 4) + 1))  # type: ignore[union-attr]
        jitter_x = int(rng.randint(-max(1, pw // 4), max(1, pw // 4) + 1))  # type: ignore[union-attr]
        y0 = int(np.clip(cy - ph // 2 + jitter_y, 0, y_max))
        x0 = int(np.clip(cx - pw // 2 + jitter_x, 0, x_max))
        kind = "roi"
    else:
        y0 = 0 if y_max <= 0 else int(rng.randint(0, y_max + 1))  # type: ignore[union-attr]
        x0 = 0 if x_max <= 0 else int(rng.randint(0, x_max + 1))  # type: ignore[union-attr]
        kind = "random"
    y1, x1 = y0 + ph, x0 + pw

    if v_padded.ndim == 2:
        v_crop = v_padded[y0:y1, x0:x1]
    else:
        v_crop = v_padded[:, y0:y1, x0:x1]
    m_crop = m_padded[y0:y1, x0:x1]
    meta = {
        "kind": kind,
        "y0": y0 - y_off, "x0": x0 - x_off,
        "y1": y1 - y_off, "x1": x1 - x_off,
        "center_used": center_used,
    }
    return v_crop, m_crop, meta


# ========================================================================= #
# Strategy B — Oversample tiny-positive slices via area-inverse weights
# ========================================================================= #
def build_area_inverse_weights(
    fg_areas: Sequence[int],
    pos_sample_weight: float = 1.0,
    tiny_threshold: int = 400,       # <400 px (~20×20) 算作 "极小"
    tiny_weight_boost: float = 3.0,  # 极小目标额外 *3
) -> List[float]:
    """按目标像素面积反比例生成样本权重（策略 B 的 Oversampling 核心）。

    普通 ``WeightedRandomSampler(pos_sample_weight=2)`` 只给所有正样本一个固定
    倍数，但对 "10x10 淋巴结" 和 "80x80 大肿块" 一视同仁，实际上小目标才是
    真正欠采样、最容易在随机梯度里"消失"的类别。本函数实现：

        w_i = 1,                                  if area_i == 0 (空切片)
            = pos_sample_weight * (median_area / area_i) ** 0.5
                                               * (tiny_weight_boost if area_i < tiny)
          else

    平方根压制避免"只有 1 px 亮斑"的脏样本获得 1000 倍离谱权重（Dice 噪声样
    本过度拟合）。

    Parameters
    ----------
    fg_areas : list[int]
        每个样本 mask 中值为 1 的像素个数，长度 = N 样本。
    pos_sample_weight : float
        正样本相对负样本的基础权重倍数（等价于旧版 build_sample_weights）。
    tiny_threshold : int
        "极小病灶" 的像素数阈值，默认 400（~20×20 px，对应 ~1-3mm 的
        0.5mm 面内分辨率淋巴结）。
    tiny_weight_boost : float
        对 area < tiny_threshold 的样本额外乘的权重倍数。
    """
    areas = np.asarray(list(fg_areas), dtype=np.int64)
    pos_mask = areas > 0
    weights = np.ones(areas.shape[0], dtype=np.float32)
    if pos_mask.any():
        base_w = max(1.0, float(pos_sample_weight))
        med = float(np.median(areas[pos_mask]))
        inv = np.sqrt(med / np.maximum(areas[pos_mask].astype(np.float32), 1.0))
        # 压制 sqrt 比例的极端倍率，上下限到 [0.5, 4]
        inv = np.clip(inv, 0.5, 4.0)
        tiny_boost = np.where(areas[pos_mask] < int(tiny_threshold),
                              float(tiny_weight_boost), 1.0).astype(np.float32)
        weights[pos_mask] = base_w * inv * tiny_boost
    return weights.tolist()


class PreprocessedDataset(Dataset):
    """Loads the paired T1/T2 npy slices produced by preprocess_*.py scripts.

    Supports two input modes:

    1. **Legacy 2.5D (k_slice == 3, default)**: loads per-slice ``*.npy`` of
       shape ``(C,H,W)`` directly.  This is the path used by the existing
       ``TrainDataset_strat`` / ``TestDataset_strat`` directories.

    2. **Pseudo-3D (k_slice > 3, e.g. 9 or 15)**: loads a stack of ``k_slice``
       contiguous 2-D slices centred on the original sample's slice index.  If
       the per-patient 3-D volume cache ``<root>/_volumes/<patient>_t1.npy``
       exists it is used in preference; otherwise the stack is built from
       neighbouring ``*.npy`` files (0-padded at the boundaries of the
       volume).  This path forces the network to model *inter-slice
       continuity* that is lost when each slice is trained independently.
    """

    def __init__(self, root, split: str = "train", augment: bool = False,
                 seed: int = 42, crop_size: int = 0,
                 k_slice: int = 3,
                 use_roi_crop: bool = True) -> None:
        if int(k_slice) % 2 != 1 or int(k_slice) <= 0:
            raise ValueError(f"k_slice must be an odd positive integer, got {k_slice!r}")
        self.k_slice = int(k_slice)

        root = Path(root)
        # Try both <root>/<split>/... and <root>/... layouts (test sets are
        # sometimes saved without a 'test/' sub-directory).
        candidate = root / split
        base = candidate if candidate.is_dir() else root
        for sub in ("images_t1", "images_t2", "masks"):
            if not (base / sub).is_dir():
                raise FileNotFoundError(
                    f"Cannot find '{sub}' under either {candidate} or {root}. "
                    "Check your --data_root / split arguments."
                )

        t1_files = _list_npy_files(base / "images_t1")
        t2_files = _list_npy_files(base / "images_t2")
        m_files  = _list_npy_files(base / "masks")
        common = _intersect_names(t1_files, t2_files, m_files)
        if not common:
            raise RuntimeError(f"No overlapping sample names under {base}. "
                               "Is this a preprocessed npy dataset directory?")

        self.base = base
        self.t1_dir = base / "images_t1"
        self.t2_dir = base / "images_t2"
        self.m_dir  = base / "masks"
        self.names  = common
        self.split  = split
        self.augment = bool(augment)
        self.crop_size = int(crop_size)
        self.use_roi_crop = bool(use_roi_crop)
        self.seed = int(seed)
        self._rng = np.random.RandomState(self.seed)

        # --------------------------------------------------------------- masks
        self._is_positive: List[bool] = []
        self._fg_areas: List[int] = []
        for n in self.names:
            m = np.asarray(np.load(self.m_dir / f"{n}.npy", mmap_mode="r"))
            s = int((m > 0.5).sum())
            self._is_positive.append(s > 0)
            self._fg_areas.append(s)

        # ------------------------------------------------- per-sample metadata
        # Try to load the slice records produced by preprocess_dataset_stratified,
        # which map each sample name → (patient_id, slice_idx_in_volume).
        self._slice_records: Dict[str, Dict[str, object]] = {}
        rec_path = (root / "preprocess_report.json") if (root / "preprocess_report.json").is_file() \
            else (root.parent / "preprocess_report.json") if (root.parent / "preprocess_report.json").is_file() \
            else None
        if rec_path is not None:
            try:
                report = json.load(open(rec_path))
                # Preprocess reports are organised as {split_name: {slice_records: [...]}}
                for split_key, split_val in report.items():
                    if isinstance(split_val, dict) and "slice_records" in split_val:
                        for rec in split_val["slice_records"]:
                            if isinstance(rec, dict) and "filename" in rec:
                                self._slice_records[str(rec["filename"])] = rec
            except Exception:
                self._slice_records = {}

        # ------------------------------------------------- per-patient ordered
        # index (k_slice fallback): map name → (patient_id, pos_in_patient) and
        # patient_id → [(slice_idx, filename), ...] sorted by slice_idx.
        # This is the CRITICAL FIX that prevents _build_stack_from_neighbours
        # from borrowing slices from a DIFFERENT patient at boundaries (which
        # is what caused k=9 test Dice to drop 5.5% from 0.709 → 0.655 in
        # epoch150 — 232/398 test samples had cross-patient neighbours!).
        self._pid_sorted: Dict[object, List[Tuple[int, str]]] = {}
        self._name_to_pidpos: Dict[str, Tuple[object, int]] = {}
        if self._slice_records:
            import collections as _C
            tmp: dict = _C.defaultdict(list)
            for name, rec in self._slice_records.items():
                pid = rec.get('patient_id')
                sli = rec.get('slice_idx')
                if pid is None or not isinstance(sli, int):
                    continue
                tmp[pid].append((int(sli), str(name)))
            for pid, lst in tmp.items():
                lst.sort(key=lambda t: t[0])
                self._pid_sorted[pid] = lst
                for pos, (_s, fname) in enumerate(lst):
                    self._name_to_pidpos[fname] = (pid, pos)

        # ------------------------------------------------- volume cache (k>3)
        # If <base>/_volumes exists, maps patient_id → (t1_vol_path, t2_vol_path, m_vol_path).
        self._vol_cache: Dict[str, Tuple[Optional[str], Optional[str], Optional[str]]] = {}

        # ------------------------------------------------------ augmentation
        self.transform: Optional[A.Compose] = None
        roi_ratio = 0.7 if self.use_roi_crop else 0.0
        if self.augment and _ALB_OK:
            self.transform = _build_albumentations_transform(
                crop_size=self.crop_size, seed=self.seed, roi_center_ratio=roi_ratio,
            )
        elif self.augment:
            import warnings
            warnings.warn(
                "albumentations not importable; augment=True will be a NO-OP. "
                f"Import error: {_ALB_ERR!r}"
            )

    # ---------------------------------------------------------------- helpers
    def __len__(self) -> int:
        return len(self.names)

    def _get_patient_volume(self, patient_id: str):
        if patient_id not in self._vol_cache:
            self._vol_cache[patient_id] = _find_patient_volumes(self.base, patient_id)
        return self._vol_cache[patient_id]

    def _build_stack_from_vol(self, vol: Optional[np.ndarray],
                              center_slice: int, k: int) -> np.ndarray:
        """Return (k, H, W) float32 stack centred on ``center_slice``; zero-pad."""
        if vol is None:
            # We cannot build the stack — caller will fall back to per-slice npy.
            return None  # type: ignore[return-value]
        v = np.asarray(vol, dtype=np.float32)
        if v.ndim == 4 and v.shape[0] == 1:
            v = v[0]
        D = v.shape[0]
        half = k // 2
        lo = center_slice - half
        hi = center_slice + half + 1
        if lo < 0 or hi > D:
            pad_lo = max(0, -lo)
            pad_hi = max(0, hi - D)
            if v.ndim == 3:
                v = np.pad(v, ((pad_lo, pad_hi), (0, 0), (0, 0)),
                           mode="constant", constant_values=0.0)
            else:
                raise ValueError(f"Unexpected volume shape {v.shape} (need D,H,W)")
            lo += pad_lo
            hi += pad_lo
        return np.ascontiguousarray(v[lo:hi]).astype(np.float32, copy=False)

    def _build_stack_from_neighbours(self, t1_or_t2_dir: Path, sample_idx: int,
                                     center_name: str, k: int,
                                     orig_c: int = 3) -> np.ndarray:
        """Fallback: build a k-slice stack from single-slice npy files.

        The existing per-slice *.npy already store a 3-channel 2.5D stack, so
        we take the **middle channel** of each neighbouring slice as the
        intensity of that z-position in the new pseudo-3D stack.

        **Critical ordering fix**: neighbours are selected from the **same
        patient** using the preprocess_report (patient_id + slice_idx) index.
        Previously we just shifted the **global** ``sample_idx`` by ±half,
        which borrowed slices from the neighbouring *patient* at boundary
        indices (232/398 test samples → cross-patient context → k=9 Dice
        collapsed by 5.5%).  When the per-patient index is unavailable we
        gracefully fall back to the old global-offset behaviour with a
        one-shot warning.
        """
        half = k // 2
        center_hw = None
        channels: List[np.ndarray] = []
        order: List[Optional[str]] = []

        # ------- NEW: build order from per-patient slice list --------------
        if self._name_to_pidpos and center_name in self._name_to_pidpos:
            pid, cpos = self._name_to_pidpos[center_name]
            plist = self._pid_sorted.get(pid, [])
            for rel in range(-half, half + 1):
                j = cpos + rel
                if 0 <= j < len(plist):
                    order.append(plist[j][1])
                else:
                    order.append(None)  # same-patient out-of-range → zero-pad
        else:
            # Legacy fall-back (no preprocess_report available): use global
            # sample_idx offset; warn once per dataset instance.
            if not getattr(self, '_warned_global_neighbours', False):
                import warnings
                warnings.warn(
                    "preprocess_report slice_records missing: k_slice>3 will "
                    "use global sample_idx offsets, which leaks cross-patient "
                    "context at patient boundaries and degrades accuracy.")
                self._warned_global_neighbours = True
            N = len(self.names)
            for rel in range(-half, half + 1):
                j = sample_idx + rel
                if 0 <= j < N:
                    order.append(self.names[j])
                else:
                    order.append(None)
        # -------------------------------------------------------------------

        for name in order:
            if name is not None:
                arr = np.load(t1_or_t2_dir / f"{name}.npy")
                if arr.ndim == 3:
                    mid = arr[arr.shape[0] // 2]
                elif arr.ndim == 2:
                    mid = arr
                else:
                    raise ValueError(f"unexpected npy shape {arr.shape} for {name}")
            else:
                if center_hw is None:
                    ref = np.load(t1_or_t2_dir / f"{center_name}.npy")
                    center_hw = ref.shape[-2:]
                mid = np.zeros(center_hw, dtype=np.float32)
            if center_hw is None:
                center_hw = mid.shape
            channels.append(np.asarray(mid, dtype=np.float32))
        stack = np.stack(channels, axis=0)  # (k, H, W)
        return stack

    def _load_k_slice(self, idx: int, modality: str):
        """Return (k_slice, H, W) stack for the requested modality ('t1'/'t2').

        Returns
        -------
        (stack, mask_centre) where stack is (k,H,W) float32 and mask_centre is
        (H,W) uint8.  If k_slice==3 the legacy per-slice npy content is
        returned unchanged.
        """
        name = self.names[idx]
        dir_npy = self.t1_dir if modality == "t1" else self.t2_dir
        if self.k_slice == 3:
            legacy = self._ensure_chw(np.load(dir_npy / f"{name}.npy"))
            # canonical C,H,W with C=3 for all legacy
            if legacy.shape[0] != 3:
                if legacy.ndim == 3 and legacy.shape[0] > 3:
                    legacy = legacy[:3]
                elif legacy.ndim == 2:
                    legacy = np.stack([legacy] * 3, axis=0).astype(np.float32)
            return legacy.astype(np.float32, copy=False)

        # k_slice > 3 path ------------------------------------------------
        rec = self._slice_records.get(name, {})
        patient_id = rec.get("patient_id")
        center_slice = rec.get("slice_idx") if isinstance(rec.get("slice_idx"), int) else None
        k = self.k_slice

        # Preferred path: per-patient volume cache -------------------------
        t1p, t2p, mp = (None, None, None)
        if patient_id is not None:
            t1p, t2p, mp = self._get_patient_volume(str(patient_id))
        vol_path = t1p if modality == "t1" else t2p
        if vol_path is not None and center_slice is not None:
            vol = np.load(vol_path, mmap_mode="r")
            return self._build_stack_from_vol(vol, int(center_slice), k)

        # Fallback path: concatenate middle channels of neighbouring 2.5D npy
        return self._build_stack_from_neighbours(dir_npy, idx, name, k)

    def build_sample_weights(self, pos_sample_weight: float,
                             mode: str = "area_inverse",
                             tiny_threshold: int = 400,
                             tiny_weight_boost: float = 3.0) -> List[float]:
        """生成 WeightedRandomSampler 的权重数组。

        Parameters
        ----------
        pos_sample_weight : float
            正样本的基础权重倍数（负样本始终 = 1.0）。
        mode : {'binary', 'area_inverse'}
            * ``binary``      —— 旧行为：正样本一律乘以 pos_sample_weight。
            * ``area_inverse`` —— **策略 B：面积反比例加权**，默认推荐。
        tiny_threshold, tiny_weight_boost
            仅在 ``area_inverse`` 模式生效，给 < tiny_threshold 像素的极小
            病灶额外乘 tiny_weight_boost。
        """
        if mode == "binary":
            w_pos = max(1.0, float(pos_sample_weight))
            return [w_pos if p else 1.0 for p in self._is_positive]
        if mode == "area_inverse":
            return build_area_inverse_weights(
                self._fg_areas,
                pos_sample_weight=pos_sample_weight,
                tiny_threshold=tiny_threshold,
                tiny_weight_boost=tiny_weight_boost,
            )
        raise ValueError(f"Unknown mode {mode!r}, use 'binary' or 'area_inverse'")

    @staticmethod
    def _ensure_chw(arr: np.ndarray, dtype=np.float32) -> np.ndarray:
        a = np.asarray(arr)
        if a.ndim == 2:
            a = a[None, ...]
        elif a.ndim == 3 and a.shape[-1] in (1, 3, 5) and a.shape[0] > a.shape[-1]:
            # heuristic: accept (H,W,C) input and transpose to (C,H,W)
            a = np.ascontiguousarray(np.transpose(a, (2, 0, 1)))
        return a.astype(dtype, copy=False)

    @staticmethod
    def _ensure_hw_mask(arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr, dtype=np.float32)
        while a.ndim > 2:
            if a.shape[0] in (1, 3, 5) and all(a.shape[0] == a.shape[k] for k in range(a.ndim) if False):
                pass
            if a.shape[0] == 1:
                a = a[0]
            elif a.shape[-1] == 1:
                a = a[..., 0]
            else:
                a = a.max(axis=0) if a.shape[0] <= 5 else a
        return np.asarray(a > 0.5, dtype=np.uint8)

    # ----------------------------------------------------------------- core
    def __getitem__(self, idx: int):
        name = self.names[idx]
        # Load T1/T2 with the k_slice-aware loader.  k=3 returns the legacy
        # 3-channel 2.5D npy unchanged; k>3 returns a (k, H, W) pseudo-3D stack
        # from the per-patient volume cache when available.
        t1 = self._load_k_slice(idx, "t1")  # (k_slice, H, W) float32
        t2 = self._load_k_slice(idx, "t2")  # (k_slice, H, W) float32
        # Mask is always 2-D for the current centroid slice (same as training
        # target) — k_slice adds context, but the supervision remains on the
        # same central slice so metrics are directly comparable with the
        # k=3 2.5D baseline.
        m_raw = np.load(self.m_dir / f"{name}.npy")
        mask_hw = self._ensure_hw_mask(m_raw)  # (H,W) uint8 {0,1}

        if self.transform is not None:
            # Pack T1 and T2 along the channel axis so they receive the SAME
            # geometric transform (otherwise T1 & T2 would be independently
            # shifted / rotated / cropped).
            C1, H, W = t1.shape
            C2 = t2.shape[0]
            img_stacked = np.concatenate([t1, t2], axis=0)  # (C1+C2, H, W)
            img_hwc = np.ascontiguousarray(
                np.transpose(img_stacked, (1, 2, 0))
            ).astype(np.float32, copy=False)
            out = self.transform(image=img_hwc, mask=mask_hw)
            aug_hwc = out["image"]
            aug_mask = out["mask"]
            aug_stacked = np.ascontiguousarray(np.transpose(aug_hwc, (2, 0, 1)))
            t1_aug = aug_stacked[:C1].astype(np.float32, copy=False)
            t2_aug = aug_stacked[C1:C1 + C2].astype(np.float32, copy=False)
            if aug_mask.ndim == 3:
                aug_mask = aug_mask[..., 0]
            m_aug = np.asarray(aug_mask > 0.5, dtype=np.float32)
            m_aug = m_aug[None, ...]  # (1, H, W)
        elif self.crop_size > 0:
            # ---------------------------------------------------------------
            # 用户显式要求的 "不要纯随机裁剪！" 纯 numpy 路径：
            # augment=False 但 crop_size > 0 时，仍然用 roi_aware_crop
            # 执行"70% ROI / 30% 随机"的策略，保证小目标不会因 random crop
            # 被全部从 train batch 中消除。T1/T2 共享 rng 以保证裁剪
            # 坐标完全一致。
            #
            # --use_roi_crop=False 时，把 ratio=0 强制传进去，等价于纯
            # 随机裁剪（避免破坏 API 契约，同时让用户旧命令保留默认
            # True = ROI-aware 行为不变）。
            # ---------------------------------------------------------------
            cs = int(self.crop_size)
            roi_ratio = 0.7 if self.use_roi_crop else 0.0
            # Build a per-sample derived RandomState (seeded by global seed +
            # idx) so the crop is deterministic for a given sample index when
            # seed is fixed.  Matches typical torchvision semantics.
            seed_base = int(self._rng.randint(0, (1 << 30)))
            seed_s = ((seed_base ^ (idx * 2654435761)) & 0xFFFFFFFF)
            srng = np.random.RandomState(seed_s)
            t1_c, m_half, _meta = roi_aware_crop(
                t1, mask_hw, patch_size=(cs, cs), ratio=roi_ratio, rng=srng
            )
            srng2 = np.random.RandomState(seed_s)  # same seed -> same crop coords
            t2_c, m_final, _meta2 = roi_aware_crop(
                t2, mask_hw, patch_size=(cs, cs), ratio=roi_ratio, rng=srng2
            )
            m_aug = np.asarray(m_final > 0.5, dtype=np.float32)[None, ...]
            t1_aug = t1_c.astype(np.float32, copy=False)
            t2_aug = t2_c.astype(np.float32, copy=False)
        else:
            t1_aug = t1
            t2_aug = t2
            m_aug = mask_hw.astype(np.float32, copy=False)[None, ...]  # (1,H,W)

        return {
            "image_t1": torch.from_numpy(t1_aug),
            "image_t2": torch.from_numpy(t2_aug),
            "mask":     torch.from_numpy(m_aug),
            "is_positive": torch.tensor(bool(self._is_positive[idx])),
            "fg_area": torch.tensor(int(self._fg_areas[idx])),
            "filename": name,
        }


# ---------------------------------------------------------------- get_train_transform (explicit public helper, matches user spec)
def get_train_transform(image_height: int = 256, image_width: int = 256,
                        roi_center_ratio: float = 0.7,
                        seed: Optional[int] = None):
    """Public helper that constructs an albumentations Compose exactly as
    specified in the user's task:  input image=(H,W,C) / mask=(H,W), returns
    the augmented dict with identical shapes.

    The preprocessed_dataset internal ``_build_albumentations_transform``
    uses the same recipe with ``crop_size = image_height`` (assumes square
    crops, which is standard for 2.5D segmentation).

    Use this standalone helper when you want to augment e.g. a 5-channel
    2.5D stack produced by ``scripts/mri_25d_pipeline.py`` before passing
    it into a custom training loop that does not use PreprocessedDataset.
    """
    if image_height != image_width:
        import warnings
        warnings.warn(
            "get_train_transform assumes square crops in the current ROI crop "
            f"implementation; got height={image_height} width={image_width}. "
            "The behaviour is still valid but padding uses 'image_height' "
            "and 'image_width' independently."
        )
    if not _ALB_OK:
        raise RuntimeError(
            f"albumentations required but unavailable. err={_ALB_ERR!r}"
        )
    cs = max(int(image_height), int(image_width))
    tf = _build_albumentations_transform(
        crop_size=cs, seed=seed, roi_center_ratio=float(roi_center_ratio),
    )
    # Rebuild with possibly asymmetric crop via a post-hoc crop when sizes differ.
    if image_height != image_width:
        # Use the symmetric helper + extra asymmetric final crop via CenterCrop
        # is simpler than refactoring the custom ROI class.
        extra = A.Compose(
            [A.CenterCrop(height=image_height, width=image_width, p=1.0)],
            additional_targets={"mask": "mask"},
        )

        class _Chained:
            def __init__(self, aug, post):
                self.aug = aug
                self.post = post

            def __call__(self, *args, **kwargs):
                out = self.aug(*args, **kwargs)
                if "image" in out and "mask" in out:
                    return self.post(image=out["image"], mask=out["mask"])
                return out

        return _Chained(tf, extra)
    return tf
