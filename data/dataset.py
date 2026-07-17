"""
data/dataset.py
================

Paired T1/T2 slice-level dataset for the **per-patient directory**
layout that the user actually uploaded:

    data/
    ├── traindata/
    │   └── <patient_id>/
    │       ├── t1/
    │       │   ├── <T1 image>.nii.gz        (the actual T1 scan)
    │       │   └── T1-node-label_*.nii.gz   (the T1 mask)
    │       └── t2/
    │           ├── <T2 image>.nii.gz        (the actual T2 scan)
    │           └── T2-node-label_*.nii.gz   (the T2 mask)
    └── testdata/
        └── <patient_id>/ ...

Each (image, mask) pair is a 3-D volume.  The mask is identified by
the substring ``label`` (case-insensitive); the image is the other
``.nii.gz`` file.  When a patient has more than one (image, mask)
pair in a modality (e.g. consecutive slice ranges of the same
series) the loader matches image ↔ mask by their trailing
``_<idx>`` suffix and **only keeps (T1, T2) suffix matches that
exist in both modalities**.

The two masks for the same slice are combined with a configurable
operator (default = OR) so the model sees a single target.
"""

import os
import random
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as TF
from torch.utils.data import Dataset


# --------------------------------------------------------------------------- #
# Volume-level LRU cache (keyed on a single .nii.gz path)
# --------------------------------------------------------------------------- #
class _VolumeCache:
    """Tiny LRU cache for the most-recently-used 3-D volumes."""

    def __init__(self, capacity: int = 8):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, path):
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]
        arr = sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)
        self.cache[path] = arr
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return arr


# Module-level cache (per worker process)
_VOLUME_CACHE = _VolumeCache(capacity=8)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
_LABEL_PAT = re.compile(r'_(\d+)$')   # trailing _<digits>


def _split_idx_from_name(stem: str) -> str:
    """Extract the trailing _<idx> from a filename stem, or '' if absent.

    Examples
    --------
    'T1-node-label_5'         -> '_5'
    'T1-node-label'           -> ''
    '5 OAx T1 FSE_3'          -> '_3'
    '5 OAx T1 FSE'            -> ''
    """
    m = _LABEL_PAT.search(stem)
    return f'_{m.group(1)}' if m else ''


def _is_mask(fname: str) -> bool:
    """Mask file heuristic: filename contains the word 'label' (or 'mask')."""
    low = fname.lower()
    return ('label' in low) or ('mask' in low)


def _pair_with_suffix(files):
    """Group a list of (path, stem) by their trailing _<idx> suffix.

    Returns
    -------
    dict : suffix -> (image_path, mask_path)
           Only suffix groups that have BOTH an image and a mask are kept.
    """
    by_suffix = {}
    for path, stem in files:
        sfx = _split_idx_from_name(stem)
        slot = by_suffix.setdefault(sfx, [None, None])   # [img, msk]
        if _is_mask(path.name):
            slot[1] = path
        else:
            slot[0] = path
    return {sfx: (img, msk) for sfx, (img, msk) in by_suffix.items()
            if img is not None and msk is not None}


def _percentile_clip(arr, low=0.5, high=99.5):
    lo = np.percentile(arr, low)
    hi = np.percentile(arr, high)
    return np.clip(arr, lo, hi)


def _zscore(arr, eps=1e-6):
    mean = arr.mean()
    std = arr.std()
    if std < eps:
        std = eps
    return (arr - mean) / std


def _resize_2d(img, size, mode):
    t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
    if mode == 'bilinear':
        out = TF.interpolate(t, size=(size, size), mode='bilinear', align_corners=False)
    else:
        out = TF.interpolate(t, size=(size, size), mode='nearest')
    return out.squeeze(0).squeeze(0).numpy()


def _random_crop(img1, img2, msk, crop_h, crop_w):
    H, W = img1.shape[-2:]
    pad_h = max(0, crop_h - H)
    pad_w = max(0, crop_w - W)
    if pad_h or pad_w:
        img1 = np.pad(img1, ((0, pad_h), (0, pad_w)), mode='constant',
                      constant_values=float(img1.mean()))
        img2 = np.pad(img2, ((0, pad_h), (0, pad_w)), mode='constant',
                      constant_values=float(img2.mean()))
        msk = np.pad(msk, ((0, pad_h), (0, pad_w)), mode='constant',
                     constant_values=0.0)
        H, W = img1.shape
    top = random.randint(0, H - crop_h)
    left = random.randint(0, W - crop_w)
    return (img1[top:top + crop_h, left:left + crop_w].copy(),
            img2[top:top + crop_h, left:left + crop_w].copy(),
            msk[top:top + crop_h, left:left + crop_w].copy())


def _gamma_jitter(arr, gamma_range=(0.7, 1.4), eps=1e-6):
    g = random.uniform(*gamma_range)
    a = arr - arr.min() + eps
    a = a / (a.max() + eps)
    a = np.power(a, 1.0 / g)
    return (a - a.mean()) / (a.std() + eps)


def _brightness_contrast(arr, b_range=(-0.1, 0.1), c_range=(0.8, 1.2)):
    return arr * random.uniform(*c_range) + random.uniform(*b_range)


def _combine_masks(t1_m, t2_m, mode):
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
    """Per-patient paired T1/T2 slice dataset.

    Args
    ----
    root          : path to either ``data/traindata`` or ``data/testdata``
    split         : 'train' | 'val' | 'test'  (only used for logging and
                    for the resplit logic)
    trainsize     : square resize target (default 256)
    augment       : whether to apply random augmentation
    crop_size     : if > 0, random-crop to (crop_size, crop_size) at native
                    resolution BEFORE resize
    mask_combine  : 'or' | 'and' | 't1' | 't2'  (default 'or')
    resplit       : if True, split paired patients deterministically
                    (used for traindata → train/val).  Set False for
                    testdata (every patient is test).
    seed          : random seed for the deterministic re-split
    train_ratio   : fraction used for training
    val_ratio     : fraction used for validation
    test_ratio    : fraction used for testing (usually 0 for traindata)
    require_pair  : if True, only yield patients that have a complete
                    T1+T2 pair (image AND mask, with matching suffix)
    """

    def __init__(self,
                 root: str,
                 split: str = 'train',
                 trainsize: int = 256,
                 augment: bool = True,
                 crop_size: int = 0,
                 mask_combine: str = 'or',
                 resplit: bool = True,
                 seed: int = 42,
                 train_ratio: float = 0.8,
                 val_ratio: float = 0.2,
                 test_ratio: float = 0.0,
                 require_pair: bool = True):
        super().__init__()
        self.root         = str(root)
        self.split        = split
        self.trainsize    = trainsize
        self.augment      = augment
        self.crop_size    = int(crop_size) if crop_size else 0
        self.mask_combine = mask_combine
        self.resplit      = resplit
        self.seed         = seed
        self.require_pair = require_pair

        # --------------------------------------------------------------
        # 1) enumerate patients, pair (T1 img, T1 msk) and (T2 img, T2 msk)
        #    by trailing _<idx> suffix
        # --------------------------------------------------------------
        root_path = Path(self.root)
        if not root_path.is_dir():
            raise FileNotFoundError(f'root does not exist: {root_path}')

        all_patients = []   # (patient_id, [(t1_img, t1_msk, t2_img, t2_msk), ...])
        n_skip_dir     = 0
        n_skip_incompl = 0
        n_skip_nomatch = 0
        n_skip_depth   = 0

        for pdir in sorted(p for p in root_path.iterdir() if p.is_dir()):
            t1d = pdir / 't1'
            t2d = pdir / 't2'
            if not t1d.is_dir() or not t2d.is_dir():
                n_skip_dir += 1
                continue

            t1_files = [(f, f.stem) for f in t1d.iterdir() if f.suffix == '.gz']
            t2_files = [(f, f.stem) for f in t2d.iterdir() if f.suffix == '.gz']

            t1_pairs = _pair_with_suffix(t1_files)   # suffix -> (img, msk)
            t2_pairs = _pair_with_suffix(t2_files)
            if not t1_pairs or not t2_pairs:
                n_skip_incompl += 1
                continue

            # keep only suffixes that exist in BOTH modalities
            common = set(t1_pairs) & set(t2_pairs)
            if not common:
                n_skip_nomatch += 1
                continue

            # build (T1_img, T1_msk, T2_img, T2_msk) per suffix
            series = []
            for sfx in sorted(common):
                t1_img, t1_msk = t1_pairs[sfx]
                t2_img, t2_msk = t2_pairs[sfx]
                # read masks ONCE to learn depth and combined-target
                try:
                    t1m = sitk.GetArrayFromImage(sitk.ReadImage(str(t1_msk)))
                    t2m = sitk.GetArrayFromImage(sitk.ReadImage(str(t2_msk)))
                except Exception as e:
                    print(f'[WARN] {pdir.name}/{sfx}: mask read failed: {e}, skip')
                    continue
                # squeeze to (D, H, W)
                while t1m.ndim > 3: t1m = np.take(t1m, 0, axis=0)
                while t2m.ndim > 3: t2m = np.take(t2m, 0, axis=0)
                if t1m.ndim == 2: t1m = t1m[None]
                if t2m.ndim == 2: t2m = t2m[None]
                D1, D2 = t1m.shape[0], t2m.shape[0]
                if D1 != D2:
                    n_skip_depth += 1
                    print(f'[WARN] {pdir.name}/{sfx}: T1 depth {D1} != T2 depth {D2}, skip')
                    continue
                D = D1
                # pre-compute per-slice target presence using OR
                for s in range(D):
                    m_t1 = (t1m[s] > 0.5) if t1m is not None else None
                    m_t2 = (t2m[s] > 0.5) if t2m is not None else None
                    if self.mask_combine == 'or':
                        has = bool((m_t1 | m_t2).any())
                    elif self.mask_combine == 'and':
                        has = bool((m_t1 & m_t2).any())
                    elif self.mask_combine == 't1':
                        has = bool(m_t1.any())
                    else:  # 't2'
                        has = bool(m_t2.any())
                    series.append((
                        str(t1_img), str(t1_msk),
                        str(t2_img), str(t2_msk),
                        s, has,
                    ))

            if series:
                all_patients.append((pdir.name, series))

        if not all_patients:
            raise RuntimeError(
                f'No samples in {self.root} '
                f'(skipped: dir={n_skip_dir}, incompl={n_skip_incompl}, '
                f'no_match={n_skip_nomatch}, depth={n_skip_depth})')

        # --------------------------------------------------------------
        # 2) resplit (deterministic, per-patient)
        # --------------------------------------------------------------
        patient_ids = [p[0] for p in all_patients]
        if self.resplit:
            rng = random.Random(self.seed)
            ids = list(patient_ids)
            rng.shuffle(ids)
            n_total = len(ids)
            n_train = int(round(n_total * train_ratio))
            n_val   = int(round(n_total * val_ratio))
            n_test  = int(round(n_total * test_ratio))
            if train_ratio > 0: n_train = max(n_train, 1)
            if val_ratio   > 0: n_val   = max(n_val,   1)
            if test_ratio  > 0: n_test  = max(n_test,  1)
            over = max(0, n_train + n_val + n_test - n_total)
            n_train -= min(over, n_train)
            train_ids = set(ids[:n_train])
            val_ids   = set(ids[n_train:n_train + n_val])
            test_ids  = set(ids[n_train + n_val:n_train + n_val + n_test])
            if split == 'train':
                allowed = train_ids
            elif split == 'val':
                allowed = val_ids
            elif split == 'test':
                allowed = test_ids
            else:
                raise ValueError(f"Unknown split: {split}")
            self._split_sizes = (n_train, n_val, n_test, n_total)
        else:
            allowed = set(patient_ids)
            self._split_sizes = (0, 0, 0, len(patient_ids))

        # --------------------------------------------------------------
        # 3) build flat sample list (one entry per 2-D slice)
        # --------------------------------------------------------------
        all_samples = []
        for pid, series in all_patients:
            if pid not in allowed:
                continue
            all_samples.extend(series)
        if not all_samples:
            raise RuntimeError(
                f'split={split} got 0 samples '
                f'(root={self.root}, total patients={len(patient_ids)})')

        self.samples = all_samples

        # log
        n_total = len(self.samples)
        n_pos   = sum(1 for s in self.samples if s[5])
        n_pat   = sum(1 for pid, _ in all_patients if pid in allowed)
        extra = ''
        if self.resplit and hasattr(self, '_split_sizes'):
            nt, nv, nte, _ = self._split_sizes
            extra = f'  re-split=[train:{nt}/val:{nv}/test:{nte}]'
        print(f'[MedicalSliceDataset] root={self.root}  split={split:<5}  '
              f'patients={n_pat}  '
              f'samples={n_total} (pos={n_pos}, neg={n_total - n_pos})  '
              f'crop={self.crop_size or "off"}  augment={self.augment}  '
              f'mask_combine={mask_combine}  trainsize={trainsize}'
              f'{extra}')

    # ------------------------------------------------------------------ #
    def __len__(self):
        return len(self.samples)

    # ------------------------------------------------------------------ #
    def __getitem__(self, idx):
        t1_img_path, t1_msk_path, t2_img_path, t2_msk_path, s, _has = self.samples[idx]

        t1_vol = _VOLUME_CACHE.get(t1_img_path)
        t2_vol = _VOLUME_CACHE.get(t2_img_path)
        t1m_vol = _VOLUME_CACHE.get(t1_msk_path)
        t2m_vol = _VOLUME_CACHE.get(t2_msk_path)

        def _slice(arr, s):
            if arr is None: return None
            if arr.ndim == 2: return arr
            D = arr.shape[0]
            return arr[min(s, D - 1)]

        img1 = _slice(t1_vol, s)
        img2 = _slice(t2_vol, s)
        msk1 = _slice(t1m_vol, s)
        msk2 = _slice(t2m_vol, s)

        # per-modality normalisation
        img1 = _percentile_clip(img1); img1 = _zscore(img1)
        img2 = _percentile_clip(img2); img2 = _zscore(img2)
        msk  = _combine_masks(msk1 if msk1 is not None else np.zeros_like(img1),
                              msk2 if msk2 is not None else np.zeros_like(img2),
                              self.mask_combine)

        # augment
        if self.augment and self.split == 'train':
            if self.crop_size > 0 and (img1.shape[0] >= self.crop_size and
                                       img1.shape[1] >= self.crop_size):
                img1, img2, msk = _random_crop(img1, img2, msk,
                                               self.crop_size, self.crop_size)
            if random.random() < 0.5:
                img1 = img1[:, ::-1].copy(); img2 = img2[:, ::-1].copy(); msk = msk[:, ::-1].copy()
            if random.random() < 0.5:
                img1 = img1[::-1, :].copy(); img2 = img2[::-1, :].copy(); msk = msk[::-1, :].copy()
            k = random.randint(0, 3)
            if k:
                img1 = np.rot90(img1, k=k).copy()
                img2 = np.rot90(img2, k=k).copy()
                msk  = np.rot90(msk,  k=k).copy()
            # intensity augmentations — reduced probability (was 0.5/0.5/0.2)
            # to prevent per-batch loss spikes
            if random.random() < 0.30: img1 = _gamma_jitter(img1)
            if random.random() < 0.30: img2 = _gamma_jitter(img2)
            if random.random() < 0.30: img1 = _brightness_contrast(img1)
            if random.random() < 0.30: img2 = _brightness_contrast(img2)
            if random.random() < 0.10:
                sigma = 0.02
                img1 = img1 + np.random.normal(0, sigma, img1.shape).astype(np.float32)
                img2 = img2 + np.random.normal(0, sigma, img2.shape).astype(np.float32)

        # resize
        img1_r = _resize_2d(img1, self.trainsize, 'bilinear')
        img2_r = _resize_2d(img2, self.trainsize, 'bilinear')
        msk_r  = _resize_2d(msk,  self.trainsize, 'nearest')

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
    print('--- traindata / train (with crop) ---')
    train_ds = MedicalSliceDataset(
        root='./data/traindata', split='train',
        trainsize=256, augment=True, crop_size=384,
        train_ratio=0.8, val_ratio=0.2, test_ratio=0.0,
    )
    s = train_ds[0]
    print('image_t1:', s['image_t1'].shape, s['image_t1'].dtype,
          'min/max:', float(s['image_t1'].min()), float(s['image_t1'].max()))
    print('image_t2:', s['image_t2'].shape, s['image_t2'].dtype,
          'min/max:', float(s['image_t2'].min()), float(s['image_t2'].max()))
    print('mask    :', s['mask'].shape,    s['mask'].dtype,
          'pos_frac:', float(s['mask'].mean()))

    print('\n--- traindata / val ---')
    val_ds = MedicalSliceDataset(
        root='./data/traindata', split='val',
        trainsize=256, augment=False,
        train_ratio=0.8, val_ratio=0.2, test_ratio=0.0,
    )
    print('val samples:', len(val_ds))

    print('\n--- testdata / test (resplit=False, all 30 patients) ---')
    test_ds = MedicalSliceDataset(
        root='./data/testdata', split='test',
        trainsize=256, augment=False,
        resplit=False,
    )
    print('test samples:', len(test_ds))
