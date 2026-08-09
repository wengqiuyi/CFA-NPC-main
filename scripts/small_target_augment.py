"""
针对小目标（头颈部淋巴结）的 albumentations 数据增强 Pipeline
===========================================================

核心设计目标：在目标像素占比 <<1% 的极端类别不平衡场景下，**不把小目标"弄丢"**
同时仍然保持足够的数据多样性。增强 pipeline 分三部分：

1. 几何增强 (HorizontalFlip / ShiftScaleRotate / ElasticTransform)
   - 同时作用于 image 和 mask，使用 albumentations 的同步变换机制
2. 强度增强 (RandomBrightnessContrast / GaussNoise)
   - 仅作用于 image（albumentations 自动保证不污染 mask）
3. 小目标感知的 RandomCrop
   - 如果有目标：以 roi_center_ratio (默认 70%) 的概率以某个前景像素为中心裁剪
   - 其余 30% 概率或无目标样本：纯随机裁剪（保持背景建模能力）
   - 尺寸不足先 pad 到 crop_size
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import albumentations as A
    _ALB_OK = True
    _ALB_ERR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    _ALB_OK = False
    _ALB_ERR = _e
    A = None  # type: ignore


class RoiAwareRandomCrop(A.DualTransform):
    """自定义小目标感知裁剪：70% 概率以目标为中心裁剪，30% 概率纯随机裁剪。

    这是一个 ``DualTransform`` 子类，因此 image 和 mask（以及 additional_targets
    里的所有空间目标）会同步被裁剪。
    """

    def __init__(
        self,
        height: int,
        width: int,
        roi_center_ratio: float = 0.7,
        always_apply: bool = False,
        p: float = 1.0,
    ) -> None:
        super().__init__(always_apply, p)
        if height <= 0 or width <= 0:
            raise ValueError("height/width must be positive")
        if not (0.0 <= roi_center_ratio <= 1.0):
            raise ValueError("roi_center_ratio must be within [0, 1]")
        self.height = int(height)
        self.width = int(width)
        self.roi_center_ratio = float(roi_center_ratio)

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        img = params["image"]
        H, W = img.shape[:2]
        ch = self.height
        cw = self.width
        mask = params.get("mask", None)

        # 如果原图 < crop_size，先在 transform 外 pad。这里给出合理的 clamp 行为。
        y_max = max(H - ch, 0)
        x_max = max(W - cw, 0)

        has_fg = False
        if mask is not None:
            m = np.asarray(mask)
            has_fg = bool(m.sum() > 0.5) if m.size else False

        use_roi = has_fg and (float(np.random.rand()) < self.roi_center_ratio)
        if use_roi and mask is not None:
            m = np.asarray(mask)
            # 取所有前景像素坐标，在其中随机挑一个作为裁剪"锚点"
            ys, xs = np.where(m > 0.5)
            if ys.size == 0:
                use_roi = False
            else:
                pick = int(np.random.randint(0, ys.size))
                cy = int(ys[pick])
                cx = int(xs[pick])
                # 裁剪框左上角相对于中心在 [-ch/2, +ch/2] 均匀抖动
                jy = int(np.random.randint(-max(1, ch // 4), max(1, ch // 4) + 1))
                jx = int(np.random.randint(-max(1, cw // 4), max(1, cw // 4) + 1))
                y0 = cy - ch // 2 + jy
                x0 = cx - cw // 2 + jx
                # 边界保护，禁止裁剪框越界
                y0 = int(np.clip(y0, 0, y_max))
                x0 = int(np.clip(x0, 0, x_max))
        if not use_roi:
            y0 = int(np.random.randint(0, y_max + 1)) if y_max > 0 else 0
            x0 = int(np.random.randint(0, x_max + 1)) if x_max > 0 else 0

        return {"x_min": x0, "x_max": x0 + cw, "y_min": y0, "y_max": y0 + ch}

    @property
    def targets_as_params(self):
        return ["image", "mask"]

    def apply(self, img: np.ndarray, x_min: int = 0, y_min: int = 0,
              x_max: int = 0, y_max: int = 0, **params: Any) -> np.ndarray:
        return img[y_min:y_max, x_min:x_max]

    def apply_to_mask(self, img: np.ndarray, x_min: int = 0, y_min: int = 0,
                      x_max: int = 0, y_max: int = 0, **params: Any) -> np.ndarray:
        return img[y_min:y_max, x_min:x_max]

    def apply_to_bbox(self, bbox, x_min: int = 0, y_min: int = 0,
                      x_max: int = 0, y_max: int = 0, **params: Any):
        # 不需要 bbox，留 stub
        raise NotImplementedError

    def apply_to_keypoint(self, keypoint, x_min: int = 0, y_min: int = 0,
                          x_max: int = 0, y_max: int = 0, **params: Any):
        x, y, a, s = keypoint
        return (x - x_min, y - y_min, a, s)

    def get_transform_init_args_names(self):
        return ("height", "width", "roi_center_ratio")


class PadIfSmaller(A.DualTransform):
    """如果图像尺寸小于 (h, w) 则用 0 pad 到目标尺寸；否则不做改动。"""

    def __init__(self, height: int, width: int, pad_value: float = 0.0,
                 mask_pad_value: int = 0, always_apply: bool = True, p: float = 1.0):
        super().__init__(always_apply, p)
        self.height = int(height)
        self.width = int(width)
        self.pad_value = float(pad_value)
        self.mask_pad_value = int(mask_pad_value)

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        H, W = img.shape[:2]
        pad_h = max(0, self.height - H)
        pad_w = max(0, self.width - W)
        if pad_h == 0 and pad_w == 0:
            return img
        pad_b = pad_h // 2
        pad_t = pad_h - pad_b
        pad_r = pad_w // 2
        pad_l = pad_w - pad_r
        if img.ndim == 2:
            return np.pad(img, ((pad_t, pad_b), (pad_l, pad_r)),
                          mode="constant", constant_values=self.pad_value)
        return np.pad(img, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)),
                      mode="constant", constant_values=self.pad_value)

    def apply_to_mask(self, img: np.ndarray, **params: Any) -> np.ndarray:
        H, W = img.shape[:2]
        pad_h = max(0, self.height - H)
        pad_w = max(0, self.width - W)
        if pad_h == 0 and pad_w == 0:
            return img
        pad_b = pad_h // 2
        pad_t = pad_h - pad_b
        pad_r = pad_w // 2
        pad_l = pad_w - pad_r
        if img.ndim == 2:
            return np.pad(img, ((pad_t, pad_b), (pad_l, pad_r)),
                          mode="constant", constant_values=self.mask_pad_value)
        return np.pad(img, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)),
                      mode="constant", constant_values=self.mask_pad_value)

    def apply_to_bbox(self, bbox, **params):  # stub
        raise NotImplementedError

    def apply_to_keypoint(self, keypoint, **params):  # stub
        return keypoint

    def get_transform_init_args_names(self):
        return ("height", "width", "pad_value", "mask_pad_value")


def get_train_transform(
    image_height: int = 256,
    image_width: int = 256,
    roi_center_ratio: float = 0.7,
    seed: Optional[int] = None,
) -> "A.Compose":
    """构建针对小目标（头颈部淋巴结）优化的训练数据增强 pipeline。

    Parameters
    ----------
    image_height, image_width : int
        最终输出裁剪尺寸，默认 256x256。
    roi_center_ratio : float
        当样本包含前景目标时，以 **目标区域为中心** 进行裁剪的概率。
        其余概率退化为纯随机裁剪，保证模型仍然看到"病灶不在中心"的样本。

        * 为什么针对小目标必须做 ROI 感知裁剪？
          对于淋巴结这种在 512x512 原图上仅占 10~30 像素边长的极小目标，
          直接做纯 256x256 的 RandomCrop，裁剪后不包含任何前景像素的概率接近
          95%，等效于把所有正样本"变成"了负样本。ROI 感知裁剪通过把裁剪框锚定
          在前景像素上，保证了小目标在训练 mini-batch 中不会被淹没。
    seed : int, optional
        用于确定性测试的固定随机种子。

    Returns
    -------
    A.Compose
        可直接调用的 albumentations Compose 对象。
        输入约定:
            image = (H, W, C) float32    (注意通道在后！)
            mask  = (H, W)    float32 / uint8
        返回同样结构的 dict。

    Notes
    -----
    * **为什么 ElasticTransform 参数 alpha=1, sigma=50, alpha_affine=50 对小目标是"温和"的？**

      ElasticTransform 的形变幅度由 ``alpha / sigma`` 共同控制。
      - ``alpha`` 决定位移场的整体强度（越大越扭曲）
      - ``sigma`` 决定位移场的空间尺度（越大越平滑，不会出现高频抖动）

      取 alpha=1 / sigma=50 时：位移场低频、幅度 <1px/每格，等效于把整个解剖
      结构"轻轻拉伸/挤压"，淋巴结虽然小，但是形变在空间上平滑且幅度小，不会
      出现把淋巴结撕裂成多块或直接位移到裁剪框外的情况。
      如果把 alpha 改成 100（sigma=50 不变），局部形变幅度会是 100 倍，淋巴结
      这种只有几十像素的小目标很可能被"揉碎"或翻转到标签边界外，令 Dice 指标
      严重失真。

    * **为什么必须保留 30% 的纯随机裁剪，而不是 100% ROI 裁剪？**

      100% ROI 裁剪会产生**位置偏差 (position bias)**：模型会学到"目标永远在
      中心附近"，一旦推理时病灶出现在切片边缘（淋巴结本来就常见于颌面、颈动脉
      鞘附近），召回率会暴跌。保留一定比例纯随机裁剪，本质是做正则化，强制模型
      在**任何空间位置**都能识别病灶形态。

    * **为什么 GaussNoise 的 var_limit 设置在 (0.001, 0.01)？**

      进入这个 pipeline 的图像已经做过基于非背景体素的 Z-score 标准化并 clip
      到 [-3, 3]，因此 1 std 对应约为 1.0 的值。
      - sqrt(0.001) ≈ 0.032 std — 几乎不可见的背景热噪声，模拟低噪声 MRI
      - sqrt(0.01)  ≈ 0.1   std — 明显但不至于淹没病灶的噪声，模拟高 b 值或
        梯度线圈切换时的采集噪声。
      如果把 var_limit 调到 0.1 以上 (std>0.3)，小目标本身的对比度（淋巴结在
      T2 上通常只比周围肌肉亮 0.5~1.5 std）就会被噪声彻底淹没，等同于做随机
      擦除的强负增强，训练初期会让 loss 不收敛。
    """
    if not _ALB_OK:
        raise RuntimeError(
            "albumentations is required for this function. Install via `pip install albumentations`. "
            f"Import error: {_ALB_ERR!r}"
        )
    if seed is not None:
        try:
            A.set_seed(int(seed))
        except Exception:
            pass
        np.random.seed(int(seed))

    # ------------------------------------------------------------------ geometric (affect both image & mask)
    #
    # Albumentations 2.x compatibility note:
    #   - ShiftScaleRotate dropped 'value' / 'mask_value' -> use 'fill' / 'fill_mask'.
    #   - ElasticTransform dropped 'alpha_affine' / 'value' / 'mask_value' ->
    #     'fill' / 'fill_mask'.  'alpha_affine' is a legacy alias and triggers a
    #     warning in 2.x so we simply omit it; alpha=1 / sigma=50 still gives
    #     the same mild displacement field that the user's spec requested.
    #   - GaussNoise in 2.x uses 'std_range' (square root of variance) instead
    #     of 'var_limit'.  The user-specified var_limit=(0.001, 0.01) maps to
    #     std_range=(sqrt(0.001)=0.0316, sqrt(0.01)=0.1).
    geometric = [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            interpolation=1,         # bilinear for image; mask uses nearest via mask_interpolation
            border_mode=0,           # cv2.BORDER_CONSTANT = 0
            fill=0.0,
            fill_mask=0,
            p=0.5,
        ),
        # 温和的弹性形变。参数含义见 docstring。
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

    # ------------------------------------------------------------------ intensity (image-only, safe by design in albumentations)
    intensity = [
        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            brightness_by_max=False,
            p=0.3,
        ),
        A.GaussNoise(std_range=(0.0316, 0.1000), mean_range=(0.0, 0.0), p=0.3),
    ]

    # ------------------------------------------------------------------ size handling + ROI-aware crop
    size_steps = [
        PadIfSmaller(height=image_height, width=image_width, pad_value=0.0, mask_pad_value=0),
        RoiAwareRandomCrop(
            height=image_height,
            width=image_width,
            roi_center_ratio=roi_center_ratio,
            p=1.0,
        ),
    ]

    return A.Compose(
        geometric + intensity + size_steps,
        additional_targets={"mask": "mask"},
    )


# 便捷包装：接受 (C,H,W) image，内部 CHW->HWC 再 HWC->CHW，符合用户约定的接口
def apply_train_transform(
    transform: "A.Compose",
    image_chw: np.ndarray,
    mask_hw: np.ndarray,
) -> Dict[str, np.ndarray]:
    """对 (C, H, W) 多通道 2.5D 切片 + (H, W) mask 调用 albumentations。

    albumentations 的约定是 image=(H, W, C), mask=(H, W)。这里提供通道维度的
    双向转换，避免用户自己写错。
    """
    if image_chw.ndim != 3:
        raise ValueError(f"image must be (C,H,W), got shape {image_chw.shape}")
    if mask_hw.ndim != 2:
        raise ValueError(f"mask must be (H,W), got shape {mask_hw.shape}")
    C, H_img, W_img = image_chw.shape
    H_msk, W_msk = mask_hw.shape
    if (H_img, W_img) != (H_msk, W_msk):
        raise ValueError(
            f"image HW ({H_img},{W_img}) does not match mask HW ({H_msk},{W_msk})"
        )

    image_hwc = np.transpose(image_chw, (1, 2, 0)).astype(np.float32, copy=False)
    mask = np.asarray(mask_hw)
    if mask.dtype not in (np.uint8, np.int32, np.int64):
        mask = (mask > 0.5).astype(np.uint8)

    out = transform(image=image_hwc, mask=mask)
    aug_img_hwc: np.ndarray = out["image"]
    aug_mask: np.ndarray = out["mask"]
    aug_img_chw = np.ascontiguousarray(
        np.transpose(aug_img_hwc, (2, 0, 1))
    ).astype(np.float32, copy=False)
    if aug_mask.ndim == 3:
        aug_mask = aug_mask[..., 0]
    aug_mask = np.asarray(aug_mask, dtype=mask_hw.dtype)
    return {"image": aug_img_chw, "mask": aug_mask}


if __name__ == "__main__":  # pragma: no cover
    # 生成假数据: 5 通道 2.5D 切片 + 中心只有 10x10 小目标的 mask
    rng = np.random.RandomState(42)
    C, H, W = 5, 512, 512
    image = (rng.randn(C, H, W).astype(np.float32) * 0.8).clip(-3.0, 3.0)
    mask = np.zeros((H, W), dtype=np.uint8)
    yy, xx = np.mgrid[H // 2 - 5:H // 2 + 5, W // 2 - 5:W // 2 + 5]
    mask[yy, xx] = 1
    # 在病灶周围加一点 T2 "亮信号" 让假数据更真实
    image[2, yy, xx] += 1.2

    tfm = get_train_transform(
        image_height=256, image_width=256, roi_center_ratio=0.7, seed=123
    )

    before_fg_px = int(mask.sum())
    augmented = apply_train_transform(tfm, image, mask)
    after_fg_px = int((augmented["mask"] > 0.5).sum())

    print(f"before shape: image={image.shape}, mask={mask.shape}, fg_pixels={before_fg_px}")
    print(f"after  shape: image={augmented['image'].shape}, mask={augmented['mask'].shape}, "
          f"fg_pixels={after_fg_px}")

    # matplotlib 可视化（若安装则显示对比图）
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(9, 9))

        def _im(ax, arr, title, is_mask=False):
            if arr.ndim == 3:
                ax.imshow(arr[2], cmap="gray", vmin=-3, vmax=3)
            else:
                ax.imshow(arr, cmap="gray" if is_mask else "viridis",
                          vmin=0, vmax=1)
            ax.set_title(title)
            ax.axis("off")

        _im(axes[0, 0], image, "Before: 2.5D center slice (ch2)")
        _im(axes[0, 1], mask, "Before: mask", is_mask=True)
        _im(axes[1, 0], augmented["image"], "After: 2.5D center slice (ch2)")
        _im(axes[1, 1], augmented["mask"], "After: mask", is_mask=True)

        out_png = "/tmp/small_target_aug_demo.png"
        fig.tight_layout()
        fig.savefig(out_png, dpi=100)
        plt.close(fig)
        print(f"visualization saved -> {out_png}")
    except Exception as _e:
        print(f"(matplotlib not installed or headless env, skip plotting. err={_e})")

    # 另外测一个全 0 mask 的负样本，保证不会在 ROI 裁剪逻辑里报错
    empty_mask = np.zeros((H, W), dtype=np.uint8)
    out_neg = apply_train_transform(tfm, image, empty_mask)
    print(f"negative (empty mask) sample: after shape={out_neg['image'].shape}, "
          f"fg_pixels={int((out_neg['mask']>0.5).sum())}")
