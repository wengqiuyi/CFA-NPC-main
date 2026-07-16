"""
data/dataset.py
================

**Paired** slice-level dataset loader for the **NIfTI 3D-MRI T1/T2**
cross-modal segmentation task.

The manifest contains one row per *file*, with ``seq ∈ {T1, T2}``.
T1 and T2 of the same patient are stored as **two separate files**
(e.g. ``case0298_T1_0.nii.gz`` and ``case0299_T2_0.nii.gz``) — they
share the same ``patient`` and ``idx_in_patient`` but have different
``case_id`` values.

This loader:

1.  Groups rows by ``(patient, idx_in_patient)`` and keeps only groups
    that have **both** a T1 and a T2 file.
2.  Reads the T1 and T2 masks once to determine the depth ``D`` and
    the per-slice target.  The two masks are combined with a
    configurable operator (default = OR, since empirically the same
    lesion is annotated with slight disagreement between modalities).
3.  Expands every (T1, T2) volume pair to ``D`` independent
    ``(slice_t1, slice_t2, mask)`` training samples.
4.  In ``__getitem__`` the T1 and T2 slices are normalised and
    augmented **independently in intensity** (different MRI contrasts
    need different intensity jitter) but **identically in geometry**
    (a horizontal flip must flip both modalities together).
5.  Outputs ``image_t1``, ``image_t2`` and ``mask`` so the
    ``dual_backbone`` CFANet can be trained on real cross-modal pairs.
"""

import csv
import os
import random
from collections import OrderedDict

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as TF
from torch.utils.data import Dataset


# --------------------------------------------------------------------------- #
# Volume-level LRU cache (keyed on a single .nii.gz path)
# --------------------------------------------------------------------------- #
class _VolumeCache:
    """Tiny LRU cache for the most-recently-used 3-D volumes.

    A single volume is ~16 MB (16 × 512 × 512 float32).  Caching the
    last few volumes avoids re-reading the same .nii.gz from disk
    when a DataLoader worker iterates over neighbouring slice indices
    of the same volume.
    """

    def __init__(self, capacity: int = 8):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, path: str):
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]
        arr = sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)
        self.cache[path] = arr
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return arr

    def get_many(self, paths):
        return [self.get(p) for p in paths]


# One cache per process is plenty (4 paired volumes × 2 mods + mask ≈ 96 MB)
_VOLUME_CACHE = _VolumeCache(capacity=4)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _percentile_clip(arr: np.ndarray, low=0.5, high=99.5) -> np.ndarray:
    """Clip extreme intensities (typical for MRI)."""
    lo = np.percentile(arr, low)
    hi = np.percentile(arr, high)
    return np.clip(arr, lo, hi)


def _zscore(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-image z-score normalisation."""
    mean = arr.mean()
    std  = arr.std()
    if std < eps:
        std = eps
    return (arr - mean) / std


def _resize_2d(img: np.ndarray, size: int, mode: str) -> np.ndarray:
    """Resize a 2-D numpy array to ``(size, size)``."""
    t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)   # (1,1,H,W)
    if mode == 'bilinear':
        out = TF.interpolate(t, size=(size, size), mode='bilinear', align_corners=False)
    else:
        out = TF.interpolate(t, size=(size, size), mode='nearest')
    return out.squeeze(0).squeeze(0).numpy()


def _random_crop(img1, img2, msk, crop_h, crop_w):
    """Random spatial crop with the same crop box for img1/img2/msk.

    The T1, T2 and mask MUST be cropped together (same anatomical
    region).  Pads with image mean / mask 0 if the requested crop is
    larger than the input.
    """
    H, W = img1.shape[-2:]
    pad_h = max(0, crop_h - H)
    pad_w = max(0, crop_w - W)
    if pad_h or pad_w:
        img1 = np.pad(img1, ((0, pad_h), (0, pad_w)), mode='constant',
                      constant_values=float(img1.mean()))
        img2 = np.pad(img2, ((0, pad_h), (0, pad_w)), mode='constant',
                      constant_values=float(img2.mean()))
        msk  = np.pad(msk,  ((0, pad_h), (0, pad_w)), mode='constant',
                      constant_values=0.0)
        H, W = img1.shape
    top  = random.randint(0, H - crop_h)
    left = random.randint(0, W - crop_w)
    return (img1[top:top + crop_h, left:left + crop_w].copy(),
            img2[top:top + crop_h, left:left + crop_w].copy(),
            msk [top:top + crop_h, left:left + crop_w].copy())


def _gamma_jitter(arr: np.ndarray, gamma_range=(0.7, 1.4), eps: float = 1e-6) -> np.ndarray:
    """Per-modality gamma correction (random contrast)."""
    g = random.uniform(*gamma_range)
    a = arr - arr.min() + eps
    a = a / (a.max() + eps)
    a = np.power(a, 1.0 / g)
    a = (a - a.mean()) / (a.std() + eps)
    return a


def _brightness_contrast(arr: np.ndarray, b_range=(-0.1, 0.1), c_range=(0.8, 1.2)):
    """Per-modality brightness (additive) and contrast (multiplicative) jitter."""
    return arr * random.uniform(*c_range) + random.uniform(*b_range)


def _combine_masks(t1_m: np.ndarray, t2_m: np.ndarray, mode: str) -> np.ndarray:
    """Combine T1 and T2 binary masks for the same anatomical slice.

    mode
    ----
    'or'  : union — used by default, captures any pixel annotated
            in either modality (most permissive)
    'and' : intersection — very conservative
    't1'  : use T1 mask only
    't2'  : use T2 mask only
    """
    if mode == 'or':
        return ((t1_m > 0.5) | (t2_m > 0.5)).astype(np.float32)
    if mode == 'and':
        return ((t1_m > 0.5) & (t2_m > 0.5)).astype(np.float32)
    if mode == 't1':
        return (t1_m > 0.5).astype(np.float32)
    if mode == 't2':
        return (t2_m > 0.5).astype(np.float32)
    raise ValueError(f"Unknown mask_combine mode: {mode}")


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class MedicalSliceDataset(Dataset):
    """
    Args
    ----
    root          : path to the manifest.csv parent directory
    split         : 'train' | 'val' | 'test'
    trainsize     : square resize target (default 256)
    augment       : whether to apply random augmentation
    crop_size     : if > 0, random-crop to (crop_size, crop_size) at
                    native resolution BEFORE resize.  Try 384 for
                    small-target MRI segmentation.
    mask_combine  : 'or' | 'and' | 't1' | 't2' — how to merge the T1
                    and T2 mask of the same slice.  Default 'or'.
    require_pair  : if True (default), only yield groups that have
                    BOTH T1 and T2 files for the same (patient, idx).
                    If False, fall back to single-modality samples
                    (degenerate, only used for debugging).
    resplit       : if True (default), ignore the split column in
                    manifest.csv and **re-split all paired groups**
                    deterministically by (train_ratio, val_ratio,
                    test_ratio).  The original manifest split was made
                    independently per modality so it puts T1 val
                    and T2 val into completely different patients —
                    this is incompatible with the dual-input task.
    seed          : random seed used for re-splitting.
    train_ratio   : fraction of paired groups used for training.
    val_ratio     : fraction used for validation.
    test_ratio    : fraction used for testing.  The remainder is
                    dropped (use 0.8 / 0.1 / 0.1 by default).
    """

    def __init__(self,
                 root: str,
                 split: str = 'train',
                 trainsize: int = 256,
                 augment: bool = True,
                 crop_size: int = 0,
                 mask_combine: str = 'or',
                 require_pair: bool = True,
                 resplit: bool = True,
                 seed: int = 42,
                 train_ratio: float = 0.8,
                 val_ratio: float = 0.1,
                 test_ratio: float = 0.1):
        super().__init__()
        self.root        = root
        self.split       = split
        self.trainsize   = trainsize
        self.augment     = augment
        self.crop_size   = int(crop_size) if crop_size else 0
        self.mask_combine = mask_combine
        self.require_pair = require_pair
        self.resplit      = resplit
        self.seed         = seed

        manifest = os.path.join(root, 'manifest.csv')
        # ------------------------------------------------------------------
        # 1) read manifest, normalise path separator (Windows -> POSIX)
        # ------------------------------------------------------------------
        rows = []
        with open(manifest, 'r', newline='') as f:
            reader = csv.DictReader(
                f,
                fieldnames=['case_id', 'patient', 'seq', 'idx_in_patient',
                            'image', 'mask', 'image_shape', 'mask_shape', 'split'])
            for row in reader:
                img_rel = row['image'].strip().replace('\\', '/')
                msk_rel = row['mask'].strip().replace('\\', '/')
                rows.append({
                    'case_id'      : row['case_id'].strip(),
                    'patient'      : row['patient'].strip(),
                    'seq'          : row['seq'].strip().upper(),
                    'idx_in_patient': row['idx_in_patient'].strip(),
                    'image'        : os.path.join(root, img_rel),
                    'mask'         : os.path.join(root, msk_rel),
                    'split'        : row['split'].strip().lower(),
                })
        if not rows:
            raise RuntimeError(f'Empty manifest {manifest}')

        # ------------------------------------------------------------------
        # 2) group by (patient, idx_in_patient, split) and build pairs
        # ------------------------------------------------------------------
        groups = {}   # (patient, idx, split) -> {'T1': row, 'T2': row}
        for r in rows:
            key = (r['patient'], r['idx_in_patient'], r['split'])
            slot = groups.setdefault(key, {})
            if r['seq'] in ('T1', 'T2'):
                slot[r['seq']] = r

        if self.resplit:
            # --------------------------------------------------------------
            # Rebuild the split from scratch, treating every (patient, idx)
            # that has BOTH modalities as one indivisible unit.  This is
            # necessary because the original manifest split is made
            # independently per modality, so the T1 val set and the T2
            # val set come from completely different patients and the
            # dual-input task would have nothing to evaluate on.
            # --------------------------------------------------------------
            pair_keys = set()    # (patient, idx) that have BOTH
            for (pat, idx, _sp), slot in groups.items():
                if 'T1' in slot and 'T2' in slot:
                    pair_keys.add((pat, idx))

            pair_keys = sorted(pair_keys)
            rng = random.Random(self.seed)
            rng.shuffle(pair_keys)

            n_total = len(pair_keys)
            n_train = int(round(n_total * train_ratio))
            n_val   = int(round(n_total * val_ratio))
            n_test  = int(round(n_total * test_ratio))
            # ensure at least 1 group per non-empty split
            if train_ratio > 0: n_train = max(n_train, 1)
            if val_ratio   > 0: n_val   = max(n_val,   1)
            if test_ratio  > 0: n_test  = max(n_test,  1)
            # cap at n_total
            over = max(0, n_train + n_val + n_test - n_total)
            n_train -= min(over, n_train)

            train_keys = pair_keys[:n_train]
            val_keys   = pair_keys[n_train:n_train + n_val]
            test_keys  = pair_keys[n_train + n_val:n_train + n_val + n_test]

            if split == 'train':
                allowed_pairs = set(train_keys)
            elif split == 'val':
                allowed_pairs = set(val_keys)
            elif split == 'test':
                allowed_pairs = set(test_keys)
            else:
                raise ValueError(f"Unknown split={split}")
            self._split_sizes = (n_train, n_val, n_test, n_total)

            # rebuild groups: keep only the (pat, idx) for this split,
            # regardless of the original `split` column.
            new_groups = {}
            for (pat, idx, _sp), slot in groups.items():
                if (pat, idx) not in allowed_pairs:
                    continue
                if 'T1' not in slot or 'T2' not in slot:
                    continue
                new_groups[(pat, idx, split.lower())] = slot
            groups = new_groups
        else:
            # legacy: use the manifest's split column directly
            groups = {k: v for k, v in groups.items() if k[2] == split.lower()}

        if not groups:
            raise RuntimeError(f'No paired samples for split={split}')

        # ------------------------------------------------------------------
        # 3) build sample list, reading each mask pair ONCE to learn
        #    the depth D, the per-slice target (combined mask > 0.5),
        #    and to flag a healthy depth match between T1 & T2.
        # ------------------------------------------------------------------
        all_samples = []   # (t1_img, t2_img, t1_msk, t2_msk, slice_idx, has_target)
        n_skipped_no_pair = 0
        n_skipped_depth   = 0
        for (patient, idx, _sp), slot in groups.items():
            if self.require_pair and not ('T1' in slot and 'T2' in slot):
                n_skipped_no_pair += 1
                continue
            t1 = slot.get('T1')
            t2 = slot.get('T2')
            try:
                t1_m = sitk.GetArrayFromImage(sitk.ReadImage(t1['mask'])) if t1 else None
                t2_m = sitk.GetArrayFromImage(sitk.ReadImage(t2['mask'])) if t2 else None
            except Exception as e:
                print(f'[WARN] failed to read mask for {patient}/{idx}: {e}, skipping')
                continue

            # squeeze to (D, H, W)
            def _to_3d(a):
                while a is not None and a.ndim > 3:
                    a = np.take(a, 0, axis=0)
                if a is not None and a.ndim == 2:
                    a = a[None]
                return a

            t1_m = _to_3d(t1_m)
            t2_m = _to_3d(t2_m)

            # depth sanity check
            D1 = t1_m.shape[0] if t1_m is not None else None
            D2 = t2_m.shape[0] if t2_m is not None else None
            if t1_m is not None and t2_m is not None and D1 != D2:
                print(f'[WARN] {patient}/{idx} T1 depth {D1} != T2 depth {D2}, skipping')
                n_skipped_depth += 1
                continue
            D = D1 if D1 is not None else D2
            # build per-slice combined mask target
            for s in range(D):
                m_t1 = (t1_m[s] > 0.5) if t1_m is not None else np.zeros_like(t2_m[s], dtype=bool)
                m_t2 = (t2_m[s] > 0.5) if t2_m is not None else np.zeros_like(t1_m[s], dtype=bool)
                if self.mask_combine == 'or':
                    has = bool((m_t1 | m_t2).any())
                elif self.mask_combine == 'and':
                    has = bool((m_t1 & m_t2).any())
                elif self.mask_combine == 't1':
                    has = bool(m_t1.any())
                else:  # 't2'
                    has = bool(m_t2.any())
                all_samples.append((
                    t1['image'] if t1 else None,
                    t2['image'] if t2 else None,
                    t1['mask']  if t1 else None,
                    t2['mask']  if t2 else None,
                    s,
                    has,
                ))

        if not all_samples:
            raise RuntimeError(
                f'No samples for split={split} '
                f'(skipped_no_pair={n_skipped_no_pair}, skipped_depth={n_skipped_depth})')

        # No filtering — every slice of every paired group is yielded.
        self.samples = all_samples

        # ------------------------------------------------------------------
        # log
        # ------------------------------------------------------------------
        n_total = len(self.samples)
        n_pos   = sum(1 for s in self.samples if s[5])
        n_paired_vol = len(groups)
        extra = ''
        if self.resplit and hasattr(self, '_split_sizes'):
            nt, nv, nte, _ = self._split_sizes
            extra = f'  re-split=[train:{nt}/val:{nv}/test:{nte}]'
        print(f'[MedicalSliceDataset] split={split:<5}  '
              f'paired_volumes={n_paired_vol}  '
              f'samples={n_total} (pos={n_pos}, neg={n_total - n_pos})  '
              f'crop={self.crop_size or "off"}  augment={self.augment}  '
              f'mask_combine={mask_combine}  trainsize={trainsize}'
              f'{extra}')

    def __len__(self):
        return len(self.samples)

    # ------------------------------------------------------------------ #
    def __getitem__(self, idx):
        t1_path, t2_path, t1_msk_path, t2_msk_path, slice_idx, _has = self.samples[idx]

        # ---- 1. load volumes from cache (lazy) ----
        t1_vol = _VOLUME_CACHE.get(t1_path) if t1_path else None
        t2_vol = _VOLUME_CACHE.get(t2_path) if t2_path else None
        t1m_vol = _VOLUME_CACHE.get(t1_msk_path) if t1_msk_path else None
        t2m_vol = _VOLUME_CACHE.get(t2_msk_path) if t2_msk_path else None

        def _slice(arr, s):
            if arr is None: return None
            if arr.ndim == 2: return arr
            D = arr.shape[0]
            return arr[min(s, D - 1)]

        img1 = _slice(t1_vol,  slice_idx)
        img2 = _slice(t2_vol,  slice_idx)
        msk1 = _slice(t1m_vol, slice_idx)
        msk2 = _slice(t2m_vol, slice_idx)

        # ---- 2. per-modality normalisation (T1 and T2 have very
        #         different intensity distributions → must do this
        #         independently) ----
        img1 = _percentile_clip(img1)
        img1 = _zscore(img1)
        img2 = _percentile_clip(img2)
        img2 = _zscore(img2)
        msk  = _combine_masks(msk1 if msk1 is not None else np.zeros_like(img1),
                              msk2 if msk2 is not None else np.zeros_like(img2),
                              self.mask_combine)

        # ---- 3. augmentations (train only) ----
        if self.augment and self.split == 'train':
            # 3.1 random crop — SAME box for T1/T2/mask
            if self.crop_size > 0 and (img1.shape[0] >= self.crop_size and
                                       img1.shape[1] >= self.crop_size):
                img1, img2, msk = _random_crop(img1, img2, msk,
                                               self.crop_size, self.crop_size)

            # 3.2 horizontal flip — SAME for T1/T2/mask
            if random.random() < 0.5:
                img1 = img1[:, ::-1].copy()
                img2 = img2[:, ::-1].copy()
                msk  = msk [:, ::-1].copy()
            # 3.3 vertical flip
            if random.random() < 0.5:
                img1 = img1[::-1, :].copy()
                img2 = img2[::-1, :].copy()
                msk  = msk [::-1, :].copy()
            # 3.4 random 90-deg rotation
            k = random.randint(0, 3)
            if k:
                img1 = np.rot90(img1, k=k).copy()
                img2 = np.rot90(img2, k=k).copy()
                msk  = np.rot90(msk,  k=k).copy()

            # 3.5 intensity jitter — INDEPENDENT for T1 and T2
            #     (T1 and T2 are different physical contrasts)
            if random.random() < 0.5:
                img1 = _gamma_jitter(img1)
            if random.random() < 0.5:
                img2 = _gamma_jitter(img2)
            if random.random() < 0.5:
                img1 = _brightness_contrast(img1)
            if random.random() < 0.5:
                img2 = _brightness_contrast(img2)
            # 3.6 light gaussian noise — applied to BOTH (same scanner)
            if random.random() < 0.2:
                sigma = 0.02
                img1 = img1 + np.random.normal(0, sigma, img1.shape).astype(np.float32)
                img2 = img2 + np.random.normal(0, sigma, img2.shape).astype(np.float32)

        # ---- 4. resize to fixed trainsize ----
        img1_r = _resize_2d(img1, self.trainsize, 'bilinear')
        img2_r = _resize_2d(img2, self.trainsize, 'bilinear')
        msk_r  = _resize_2d(msk,  self.trainsize, 'nearest')

        # ---- 5. to tensor, tile to 3 channels for the Res2Net backbone ----
        img1_t = torch.from_numpy(img1_r).float().unsqueeze(0).expand(3, -1, -1).contiguous()
        img2_t = torch.from_numpy(img2_r).float().unsqueeze(0).expand(3, -1, -1).contiguous()
        msk_t  = torch.from_numpy(msk_r).float().unsqueeze(0)

        return {
            'image_t1': img1_t,
            'image_t2': img2_t,
            'mask'    : msk_t,
            'seq'     : 'T1+T2',
        }


# --------------------------------------------------------------------------- #
# Test
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    print('--- train (paired T1+T2, every slice, no filtering) ---')
    ds = MedicalSliceDataset(root='./TrainDataset', split='train',
                             trainsize=256, augment=True, crop_size=384)
    print('Train samples:', len(ds))
    sample = ds[0]
    print('image_t1:', sample['image_t1'].shape, sample['image_t1'].dtype,
          'min/max:', float(sample['image_t1'].min()), float(sample['image_t1'].max()))
    print('image_t2:', sample['image_t2'].shape, sample['image_t2'].dtype,
          'min/max:', float(sample['image_t2'].min()), float(sample['image_t2'].max()))
    print('mask    :', sample['mask'].shape,  sample['mask'].dtype,
          'pos_frac:', float(sample['mask'].mean()))

    print('\n--- val ---')
    val_ds = MedicalSliceDataset(root='./TrainDataset', split='val',
                                trainsize=256, augment=False)
    print('Val samples:', len(val_ds))

    print('\n--- test ---')
    test_ds = MedicalSliceDataset(root='./TrainDataset', split='test',
                                 trainsize=256, augment=False)
    print('Test samples:', len(test_ds))
