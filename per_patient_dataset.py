"""
PerPatientDataset
=================

PyTorch Dataset that consumes the per-patient preprocessed volumes produced
by ``scripts/preprocess_per_patient.py`` and applies the remaining online
steps of the spec:

  Step 3  2.5D slice stacking (context=2 -> 5 channels per modality).
          The stack is ALWAYS built from the SAME patient's volume, with
          nearest-valid-slice padding at the patient's Z boundaries.  This
          is the structural guarantee the user asked for: no slice from
          patient B can ever appear as context for patient A.
  Step 5  Small-target ROI-aware patch crop (256x256, 70% ROI / 30% random).
  Step 6  Data augmentation (geometric: Flip/ShiftScaleRotate/Elastic;
          intensity: Brightness/Contrast/GaussNoise).  T1 and T2 share the
          exact same geometric transform + crop (they are packed along the
          channel axis before passing to albumentations).
  Step 4  Per-sample weights (1.0 / 0.5 / 0.1 for positive / edge / bg)
          are exposed via ``sample_weights`` for a WeightedRandomSampler.

Layout consumed
---------------
<root>/
  manifest.json
  <split>/<patient_id>/
      t1.npy       float16 (D, H, W)   resampled + z-scored + clipped
      t2.npy       float16 (D, H, W)
      mask.npy     uint8  (D, H, W)    combined T1|T2 mask
      slices.json  [{slice_idx, kind, weight}, ...]

Output of __getitem__
---------------------
dict(
  image_t1    Tensor (5, H, W) float32     # k_slice = 2*context+1
  image_t2    Tensor (5, H, W) float32
  mask        Tensor (1, H, W) float32     # binary {0,1}, centre slice
  weight      Tensor scalar float32        # 1.0 / 0.5 / 0.1
  is_positive Tensor scalar bool
  fg_area     Tensor scalar int
  patient_id  str
  slice_idx   int
)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# Reuse the existing, spec-matched albumentations pipeline + ROI crop helper
# so augmentation is byte-identical to the original training path.
from preprocessed_dataset import (
    _build_albumentations_transform,
    roi_aware_crop,
)
try:
    import albumentations as A
    _ALB_OK = True
except Exception:  # pragma: no cover
    _ALB_OK = False
    A = None  # type: ignore


class PerPatientDataset(Dataset):
    def __init__(self, root, split: str = "train",
                 augment: bool = False,
                 crop_size: int = 256,
                 context: int = 2,
                 seed: int = 42,
                 roi_center_ratio: float = 0.7,
                 positive_only: bool = False,
                 exclude_pids=None) -> None:
        self.root = Path(root)
        self.split = str(split)
        self.augment = bool(augment)
        self.crop_size = int(crop_size)
        self.context = int(context)
        self.k_slice = 2 * self.context + 1
        self.positive_only = bool(positive_only)
        self.seed = int(seed)
        self.roi_center_ratio = float(roi_center_ratio)
        # ``exclude_pids``: optional set/list of patient ids to drop entirely
        # (e.g. known-bad / out-of-distribution cases that would skew metrics).
        self.exclude_pids = set(str(p) for p in (exclude_pids or []))

        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"manifest.json not found under {self.root}. "
                "Run scripts/preprocess_per_patient.py first.")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        split_pids = self.manifest.get("splits", {}).get(self.split, [])
        if not split_pids:
            raise RuntimeError(f"No patients in split '{self.split}' under {self.root}")

        # ---- enumerate samples (patient_id, slice_idx, weight, kind) ----
        # ``positive_only=True`` keeps ONLY slices whose centre mask has
        # foreground (kind == 'positive'), skipping the empty-mask 'edge' and
        # 'background' slices. Used for test-time inference where there is
        # nothing to score on empty masks.
        self.samples: List[Tuple[str, int, float, str]] = []
        for pid in split_pids:
            if pid in self.exclude_pids:
                continue
            sj = self.root / self.split / pid / "slices.json"
            if not sj.is_file():
                continue
            for rec in json.loads(sj.read_text(encoding="utf-8")):
                kind = str(rec["kind"])
                if self.positive_only and kind != "positive":
                    continue
                self.samples.append(
                    (pid, int(rec["slice_idx"]),
                     float(rec["weight"]), kind)
                )
        if not self.samples:
            raise RuntimeError(
                f"No samples enumerated for split '{self.split}'"
                + (" (positive_only=True)" if self.positive_only else "") + ".")

        # ---- per-patient volume memmap cache (filled lazily per worker) ----
        self._vol_cache: Dict[str, Tuple[object, object, object]] = {}

        # ---- transforms ----
        # IMPORTANT: the custom ``_RoiAwareRandomCrop`` albumentations transform
        # is silently a NO-OP in albumentations >= 2.x (its get_params_*
        # contract changed), so images larger than crop_size were never cropped
        # -> batch size mismatch. We therefore use albumentations ONLY for the
        # geometric + intensity augmentation (crop_size=0 -> no crop steps) and
        # perform the ROI-aware crop ourselves in numpy via the well-tested
        # ``roi_aware_crop`` (which always returns exactly (C, ph, pw)).
        self.transform = None
        if self.augment and _ALB_OK:
            self.transform = _build_albumentations_transform(
                crop_size=0, seed=self.seed,
                roi_center_ratio=self.roi_center_ratio,
            )
        elif self.augment and not _ALB_OK:
            import warnings
            warnings.warn("albumentations not available; augment=True will be a NO-OP.")
        # Val/test uses a deterministic numpy centre pad+crop (no randomness)
        # -> reproducible lesion-level evaluation. See ``_center_pad_crop``.

    # ------------------------------------------------------------------ helpers
    def __len__(self) -> int:
        return len(self.samples)

    @property
    def sample_weights(self) -> List[float]:
        """Per-sample weights for WeightedRandomSampler (Step 4)."""
        return [w for (_p, _s, w, _k) in self.samples]

    def _get_volumes(self, pid: str):
        if pid not in self._vol_cache:
            pdir = self.root / self.split / pid
            t1 = np.load(pdir / "t1.npy", mmap_mode="r")
            t2 = np.load(pdir / "t2.npy", mmap_mode="r")
            mk = np.load(pdir / "mask.npy", mmap_mode="r")
            self._vol_cache[pid] = (t1, t2, mk)
        return self._vol_cache[pid]

    def _build_25d_stack(self, vol, center: int) -> np.ndarray:
        """Step 3: build a (k_slice, H, W) float32 stack centred on ``center``.

        Uses nearest-valid-slice padding at the patient's Z boundaries
        (clamps the index to [0, D-1]) so every channel is real anatomy
        from THIS patient -- never zero, never another patient.  This is
        the standard 2.5D boundary behaviour (matches mri_25d_pipeline.py).
        """
        v = np.asarray(vol, dtype=np.float32)
        D = v.shape[0]
        idxs = []
        for off in range(-self.context, self.context + 1):
            j = center + off
            idxs.append(max(0, min(D - 1, j)))
        return np.stack([v[i] for i in idxs], axis=0)  # (k, H, W)

    @staticmethod
    def _center_pad_crop(vol: np.ndarray, mask: np.ndarray,
                         ph: int, pw: int) -> Tuple[np.ndarray, np.ndarray]:
        """Deterministic centre pad (to >= ph,pw) then centre crop to (ph, pw).

        Guarantees the output is exactly ``(C, ph, pw)`` / ``(ph, pw)`` with no
        randomness -- used for val/test so lesion-level evaluation is
        reproducible.  Larger inputs are centre-cropped; smaller inputs are
        zero-padded (centre) first.
        """
        m = np.asarray(mask)
        H, W = m.shape
        pad_h, pad_w = max(0, ph - H), max(0, pw - W)
        if pad_h or pad_w:
            pt, pb = pad_h // 2, pad_h - pad_h // 2
            pl, pr = pad_w // 2, pad_w - pad_w // 2
            if vol.ndim == 3:
                vol = np.pad(vol, ((0, 0), (pt, pb), (pl, pr)),
                             mode="constant", constant_values=0.0)
            else:
                vol = np.pad(vol, ((pt, pb), (pl, pr)),
                             mode="constant", constant_values=0.0)
            m = np.pad(m, ((pt, pb), (pl, pr)),
                       mode="constant", constant_values=0)
        H, W = m.shape
        y0 = max(0, (H - ph) // 2)
        x0 = max(0, (W - pw) // 2)
        if vol.ndim == 3:
            vol = vol[:, y0:y0 + ph, x0:x0 + pw]
        else:
            vol = vol[y0:y0 + ph, x0:x0 + pw]
        m = m[y0:y0 + ph, x0:x0 + pw]
        return vol, m

    # ----------------------------------------------------------------- core
    def __getitem__(self, idx: int):
        pid, slice_idx, weight, kind = self.samples[idx]
        t1_vol, t2_vol, mask_vol = self._get_volumes(pid)

        # Step 3: 2.5D stacks (k_slice, H, W) from THIS patient only.
        t1_stack = self._build_25d_stack(t1_vol, slice_idx)
        t2_stack = self._build_25d_stack(t2_vol, slice_idx)
        center_mask = np.asarray(mask_vol[slice_idx], dtype=np.uint8)  # (H, W)

        # foreground stats for monitoring / area-inverse weighting fallback
        fg_area = int((center_mask > 0).sum())
        is_pos = fg_area > 0

        C1 = t1_stack.shape[0]  # k_slice (=5 for context=2)
        # Pack T1+T2 along the channel axis so both modalities receive the
        # IDENTICAL geometric transform AND the identical crop coordinates.
        img_stacked = np.concatenate([t1_stack, t2_stack], axis=0)  # (2*k, H, W)

        if self.transform is not None:
            # ---- train: Step 6 augmentation (geometric + intensity, NO crop) ----
            img_hwc = np.ascontiguousarray(
                np.transpose(img_stacked, (1, 2, 0))
            ).astype(np.float32, copy=False)
            out = self.transform(image=img_hwc, mask=center_mask)
            aug_hwc = out["image"]
            aug_mask = out["mask"]
            aug_stacked = np.ascontiguousarray(
                np.transpose(aug_hwc, (2, 0, 1))).astype(np.float32, copy=False)
            if aug_mask.ndim == 3:
                aug_mask = aug_mask[..., 0]
            aug_mask = (np.asarray(aug_mask) > 0.5).astype(np.uint8)
            # ---- Step 5: ROI-aware patch crop 256x256 (70% ROI / 30% random) ----
            # Done in numpy (not albumentations) because the custom
            # _RoiAwareRandomCrop is a silent no-op in albumentations 2.x.
            # roi_aware_crop always returns exactly (C, ph, pw).
            aug_stacked, aug_mask, _ = roi_aware_crop(
                aug_stacked, aug_mask,
                patch_size=(self.crop_size, self.crop_size),
                ratio=self.roi_center_ratio,
            )
            t1_aug = aug_stacked[:C1]
            t2_aug = aug_stacked[C1:]
            m_aug = aug_mask.astype(np.float32)[None, ...]
        else:
            # ---- val/test: deterministic centre pad + crop to crop_size ----
            aug_stacked, aug_mask = self._center_pad_crop(
                img_stacked, center_mask, self.crop_size, self.crop_size)
            t1_aug = aug_stacked[:C1].astype(np.float32, copy=False)
            t2_aug = aug_stacked[C1:].astype(np.float32, copy=False)
            m_aug = (aug_mask > 0.5).astype(np.float32)[None, ...]

        # NOTE: use .clone() (not bare torch.from_numpy) so each returned tensor
        # owns a *resizable* PyTorch storage. With num_workers>0 PyTorch's
        # default_collate worker-branch does
        #   elem._typed_storage()._new_shared(...); elem.new(storage).resize_(...)
        # which raises "Trying to resize storage that is not resizable" when
        # `elem` is a from_numpy tensor (numpy-backed, non-resizable). Cloning
        # breaks the numpy-storage link and lets collate write straight into
        # shared memory.
        return {
            "image_t1": torch.from_numpy(np.ascontiguousarray(t1_aug)).clone(),
            "image_t2": torch.from_numpy(np.ascontiguousarray(t2_aug)).clone(),
            "mask": torch.from_numpy(np.ascontiguousarray(m_aug)).clone(),
            "weight": torch.tensor(float(weight), dtype=torch.float32),
            "is_positive": torch.tensor(bool(is_pos)),
            "fg_area": torch.tensor(int(fg_area)),
            "patient_id": pid,
            "slice_idx": int(slice_idx),
        }
