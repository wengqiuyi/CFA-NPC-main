"""
Stratified preprocessing for the 3D paired MRI dataset under /root/CFA-NPC-main-main/dataset.

What it does
------------
1) Reads original 3-D NIfTI volumes under:
     <raw_train_root>/<patient>/t1/*.nii.gz
     <raw_train_root>/<patient>/t2/*.nii.gz
     <raw_test_root>/<patient>/t1/*.nii.gz
     <raw_test_root>/<patient>/t2/*.nii.gz

   Mask heuristics (same as historical code): any file whose name contains
   "label" or "mask" is treated as a segmentation mask, the rest is image.

2) For each patient builds one sample list with the following slice policies
   applied on a *per-volume* (per patient) basis:
     - positive slices            : keep all
     - context negatives          : any empty slice within ±`--context_window`
                                    of any positive slice -> keep all
     - far negatives              : empty slices further than context_window
                                    from any positive slice:
                                      * random keep with `--far_keep_p`
                                        OR
                                      * keep one slice every `--far_keep_N`

3) Saves 2.5D stacks (3 slices) as preprocessed `.npy` with the standard
   layout that train.py / test_paired.py consume:
     <train_out>/train/{images_t1,images_t2,masks}
     <train_out>/val/{images_t1,images_t2,masks}
     <test_out>/test/{images_t1,images_t2,masks}

   Each `images_*/*.npy` has shape (3, trainsize, trainsize) float32 and is
   already normalised with the same pipeline used in data/dataset.py:
     NaN/Inf fix -> 0.5~99.5 % clip -> per-slice z-score.

4) Writes a human-readable JSON report (kept / removed counts, split sizes,
   per-patient slice decisions) under:
     <train_out>/preprocess_report.json
     <test_out>/preprocess_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Minimal, self-contained re-implementation of the historical helpers that
# used to live in data/dataset.py. Keeping them here avoids a hard dependency
# on a module that was deleted / relocated.
# ---------------------------------------------------------------------------
_LAB_PAT = re.compile(r"_(\d+)$")


def _split_suffix(stem: str) -> str:
    m = _LAB_PAT.search(stem)
    return f"_{m.group(1)}" if m else ""


def _is_mask(fname: str) -> bool:
    low = fname.lower()
    return ("label" in low) or ("mask" in low)


def _pair_files(folder: Path) -> Dict[str, Tuple[Optional[Path], Optional[Path]]]:
    """Return {key: (image_path, mask_path)} keeping only paired groups.

    Matching strategy (relaxed because image & mask suffix digits can differ):
      1. Strict suffix match on trailing _<N> (original behaviour).
      2. If nothing matched and there is exactly 1 image + 1 mask in the
         folder, pair them directly under the key "" (most common case for
         the NPC dataset where image has _4 and mask has _5).
      3. Otherwise pair images[i] with masks[i] by positional order, using
         the image's trailing suffix as key.
    """
    imgs: List[Path] = []
    msks: List[Path] = []
    for f in sorted(folder.glob("*.nii.gz")):
        try:
            if not f.exists() or not f.is_file():
                continue
        except OSError:
            continue
        if _is_mask(f.name):
            msks.append(f)
        else:
            imgs.append(f)

    strict: Dict[str, List[Optional[Path]]] = {}
    all_files: List[Path] = list(sorted(folder.glob("*.nii.gz")))
    for f in all_files:
        try:
            if not f.exists() or not f.is_file():
                continue
        except OSError:
            continue
        sfx = _split_suffix(f.stem.replace(".nii", ""))
        slot = strict.setdefault(sfx, [None, None])
        if _is_mask(f.name):
            slot[1] = f
        else:
            slot[0] = f
    matched = {k: (v[0], v[1]) for k, v in strict.items() if v[0] is not None and v[1] is not None}
    if matched:
        return matched

    if len(imgs) == 1 and len(msks) == 1:
        return {"": (imgs[0], msks[0])}

    n = min(len(imgs), len(msks))
    out: Dict[str, Tuple[Optional[Path], Optional[Path]]] = {}
    for i in range(n):
        sfx = _split_suffix(imgs[i].stem.replace(".nii", ""))
        key = sfx if sfx else f"_{i}"
        out[key] = (imgs[i], msks[i])
    return out


def _find_patient_pairs(root: Path) -> Optional[Tuple[Path, Path, Path, Path]]:
    """Return (t1_img, t1_msk, t2_img, t2_msk) for a patient folder or None."""
    pairs = {}
    for mod in ("t1", "t2"):
        mod_dir = root / mod
        if not mod_dir.is_dir():
            return None
        pairs[mod] = _pair_files(mod_dir)
    if not pairs["t1"] or not pairs["t2"]:
        return None

    common = sorted(set(pairs["t1"].keys()) & set(pairs["t2"].keys()))
    if common:
        sfx = common[-1]
        t1i, t1m = pairs["t1"][sfx]
        t2i, t2m = pairs["t2"][sfx]
        if None not in (t1i, t1m, t2i, t2m):
            return t1i, t1m, t2i, t2m  # type: ignore[return-value]

    if len(pairs["t1"]) == 1 and len(pairs["t2"]) == 1:
        t1i, t1m = next(iter(pairs["t1"].values()))
        t2i, t2m = next(iter(pairs["t2"].values()))
        if None not in (t1i, t1m, t2i, t2m):
            return t1i, t1m, t2i, t2m  # type: ignore[return-value]
    return None


def _load_volume(path: Path, is_mask: bool) -> np.ndarray:
    img = sitk.ReadImage(str(path))
    if is_mask:
        img = sitk.Cast(img > 0, sitk.sitkFloat32)
    else:
        img = sitk.Cast(img, sitk.sitkFloat32)
    arr = sitk.GetArrayFromImage(img)
    return np.asarray(arr, dtype=np.float32)


# --- intensity preprocessing ------------------------------------------------
def _check_and_fix(arr: np.ndarray, name: str) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    nan_mask = np.isnan(arr)
    inf_mask = np.isinf(arr)
    if np.any(nan_mask) or np.any(inf_mask):
        finite_mask = np.isfinite(arr)
        if np.any(finite_mask):
            mean_val = float(np.mean(arr[finite_mask]))
            min_val = float(np.min(arr[finite_mask]))
            max_val = float(np.max(arr[finite_mask]))
            arr[nan_mask] = mean_val
            arr[np.isposinf(arr)] = max_val
            arr[np.isneginf(arr)] = min_val
        else:
            arr[:] = 0.0
    return arr


def _percentile_clip(arr: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    lo = float(np.percentile(arr, low))
    hi = float(np.percentile(arr, high))
    return np.clip(arr, lo, hi)


def _zscore(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = float(arr.mean())
    std = float(arr.std())
    if std < eps:
        std = eps
    return (arr - mean) / std


def _normalize_stack(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        arr = arr[None, ...]
    out: List[np.ndarray] = []
    for i in range(arr.shape[0]):
        ch = _check_and_fix(arr[i], "slice")
        ch = _percentile_clip(ch)
        ch = _zscore(ch)
        out.append(ch.astype(np.float32, copy=False))
    return np.stack(out, axis=0)


def _stack_neighbors(vol: np.ndarray, s: int) -> np.ndarray:
    if vol.ndim == 2:
        base = vol.astype(np.float32, copy=False)
        return np.stack([base, base, base], axis=0)
    D = vol.shape[0]
    idx = [max(0, min(D - 1, s - 1)), max(0, min(D - 1, s)), max(0, min(D - 1, s + 1))]
    return np.stack([vol[i].astype(np.float32, copy=False) for i in idx], axis=0)


def _resize(arr: np.ndarray, size: int, mode: str) -> np.ndarray:
    t = torch.from_numpy(arr).float()
    if arr.ndim == 2:
        t = t.unsqueeze(0).unsqueeze(0)
        out = F.interpolate(
            t,
            size=(size, size),
            mode=mode,
            align_corners=False if mode == "bilinear" else None,
        )
        out_np: np.ndarray = out.squeeze(0).squeeze(0).numpy()
        return out_np
    t = t.unsqueeze(0)
    out = F.interpolate(
        t,
        size=(size, size),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )
    return out.squeeze(0).numpy()


# --- slice selection policies ----------------------------------------------
def _slice_strategy(
    mask_vol: np.ndarray,
    context_window: int,
    far_keep_p: Optional[float],
    far_keep_N: Optional[int],
    rng: random.Random,
) -> Dict[str, object]:
    """Per-volume slice keep / drop decisions.

    Returns
    -------
    {
      "keep":             np.ndarray[bool] of shape (D,)
      "positive":         np.ndarray[bool]
      "context":          np.ndarray[bool]   (negative but within context_window)
      "far_kept_ratio":   float              (kept / total far negatives)
      "kept_pos":         int
      "kept_context":     int
      "kept_far":         int
      "removed_far":      int
    }
    """
    if mask_vol.ndim == 2:
        mask_vol = mask_vol[None, ...]
    D = mask_vol.shape[0]
    positive = np.asarray([bool(np.any(mask_vol[s] > 0.5)) for s in range(D)])

    pos_idx = np.where(positive)[0]
    context = np.zeros(D, dtype=bool)
    if pos_idx.size > 0:
        lo = max(0, int(pos_idx.min()) - context_window)
        hi = min(D - 1, int(pos_idx.max()) + context_window)
        context[lo : hi + 1] = True
    context = context & (~positive)

    keep = np.zeros(D, dtype=bool)
    keep[positive] = True
    keep[context] = True

    far_neg_mask = (~positive) & (~context)
    far_idx_all = np.where(far_neg_mask)[0].tolist()
    kept_far = []
    if far_idx_all:
        if far_keep_N is not None and int(far_keep_N) >= 1:
            kept_far = [far_idx_all[i] for i in range(0, len(far_idx_all), int(far_keep_N))]
        elif far_keep_p is not None:
            p = min(max(float(far_keep_p), 0.0), 1.0)
            kept_far = [i for i in far_idx_all if rng.random() < p]
        else:
            # default: stratified keep ~15%
            p = 0.15
            kept_far = [i for i in far_idx_all if rng.random() < p]
    for i in kept_far:
        keep[i] = True

    kept_far_cnt = int(sum(1 for i in kept_far))
    removed_far_cnt = int(len(far_idx_all) - kept_far_cnt)
    far_kept_ratio = (kept_far_cnt / len(far_idx_all)) if far_idx_all else 1.0

    return {
        "keep": keep,
        "positive": positive,
        "context": context,
        "far_kept_ratio": float(far_kept_ratio),
        "kept_pos": int(positive.sum()),
        "kept_context": int(context.sum()),
        "kept_far": int(kept_far_cnt),
        "removed_far": int(removed_far_cnt),
    }


# --- dataset assembly -------------------------------------------------------
def _enumerate_patients(root: Path) -> List[Path]:
    out: List[Path] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        out.append(p)
    return out


def patient_level_split(
    patients: List[Path],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[Path], List[Path]]:
    rng = random.Random(seed)
    shuffled = list(patients)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    over = max(0, n_train + n_val - n)
    n_train -= min(over, n_train)
    train_pats = shuffled[:n_train]
    val_pats = shuffled[n_train : n_train + n_val]
    return train_pats, val_pats


def _save_split(
    out_dir: Path,
    split: str,
    patient_entries: List[Tuple[Path, Dict[str, object]]],
    mask_combine: str,
    trainsize: int,
) -> Tuple[int, int, int, List[Dict[str, object]]]:
    split_dir = out_dir / split
    t1_dir = split_dir / "images_t1"
    t2_dir = split_dir / "images_t2"
    m_dir = split_dir / "masks"
    # Cache per-patient 3-D volumes (resized to trainsize) under a shared
    # _volumes folder so preprocessed_dataset.PreprocessedDataset can load
    # k_slice > 3 pseudo-3D stacks *without* reading through every per-slice
    # npy.  Volume cache layout:
    #   <out_dir>/<split>/_volumes/<patient>_t1.npy   shape (D, trainsize, trainsize)
    #   <out_dir>/<split>/_volumes/<patient>_t2.npy   shape (D, trainsize, trainsize)
    #   <out_dir>/<split>/_volumes/<patient>_mask.npy shape (D, trainsize, trainsize) float32 {0,1}
    vol_dir = split_dir / "_volumes"
    for d in (t1_dir, t2_dir, m_dir, vol_dir):
        d.mkdir(parents=True, exist_ok=True)

    pos_cnt = 0
    ctx_cnt = 0
    far_cnt = 0
    slice_records: List[Dict[str, object]] = []
    sample_idx = 0

    for patient_dir, vol_info in patient_entries:
        t1_vol = vol_info["t1_vol"]  # type: ignore[index]
        t2_vol = vol_info["t2_vol"]  # type: ignore[index]
        m1_vol = vol_info["t1m_vol"]  # type: ignore[index]
        m2_vol = vol_info["t2m_vol"]  # type: ignore[index]
        keep = np.asarray(vol_info["keep"])  # type: ignore[index]
        pos = np.asarray(vol_info["positive"])  # type: ignore[index]
        ctx = np.asarray(vol_info["context"])  # type: ignore[index]

        pid = patient_dir.name
        per_patient_record: Dict[str, object] = {
            "patient_id": pid,
            "depth": int(len(keep)),
            "kept_count": int(keep.sum()),
            "kept_positive": int(np.count_nonzero(keep & pos)),
            "kept_context": int(np.count_nonzero(keep & ctx)),
            "kept_far_negative": int(np.count_nonzero(keep & ~pos & ~ctx)),
        }

        # ------------------------------------------------------------------
        # Cache the full resized volume for this patient so k_slice > 3
        # loader can index directly into the z-axis.
        # ------------------------------------------------------------------
        D = int(t1_vol.shape[0])
        # Compute combined OR mask volume (identical rule to per-slice above)
        if mask_combine == "or":
            comb = ((np.asarray(m1_vol) > 0.5) | (np.asarray(m2_vol) > 0.5)).astype(np.float32)
        elif mask_combine == "and":
            comb = ((np.asarray(m1_vol) > 0.5) & (np.asarray(m2_vol) > 0.5)).astype(np.float32)
        elif mask_combine == "t1":
            comb = (np.asarray(m1_vol) > 0.5).astype(np.float32)
        else:  # "t2"
            comb = (np.asarray(m2_vol) > 0.5).astype(np.float32)
        # Resize each z-slice of the full volume to (trainsize, trainsize)
        def _resize_vol(v, mode):
            v = np.asarray(v, dtype=np.float32)
            if v.ndim == 2:
                v = v[None, ...]
            out = np.stack([_resize(v[s], trainsize, mode) for s in range(v.shape[0])], axis=0)
            return out.astype(np.float32, copy=False)
        t1_full = _normalize_stack_per_slice(_resize_vol(t1_vol, "bilinear"))
        t2_full = _normalize_stack_per_slice(_resize_vol(t2_vol, "bilinear"))
        mk_full = (_resize_vol(comb, "nearest") > 0.5).astype(np.float32)
        # Trim all volumes to the same D, just in case (they shouldn't differ
        # after _align_z, but stay defensive).
        D_common = min(D, t1_full.shape[0], t2_full.shape[0], mk_full.shape[0])
        t1_full = t1_full[:D_common]
        t2_full = t2_full[:D_common]
        mk_full = mk_full[:D_common]
        np.save(vol_dir / f"{pid}_t1.npy", t1_full)
        np.save(vol_dir / f"{pid}_t2.npy", t2_full)
        np.save(vol_dir / f"{pid}_mask.npy", mk_full)

        # ---------- per-slice legacy 2.5D files (unchanged) -------------
        for s in range(len(keep)):
            if not bool(keep[s]):
                continue
            img1 = _normalize_stack(_stack_neighbors(t1_vol, s))
            img2 = _normalize_stack(_stack_neighbors(t2_vol, s))
            ms1 = m1_vol[s] if m1_vol.ndim == 3 else m1_vol
            ms2 = m2_vol[s] if m2_vol.ndim == 3 else m2_vol
            ms1 = _check_and_fix(np.asarray(ms1, dtype=np.float32), "m1")
            ms2 = _check_and_fix(np.asarray(ms2, dtype=np.float32), "m2")

            if mask_combine == "or":
                mask = np.asarray((ms1 > 0.5) | (ms2 > 0.5), dtype=np.float32)
            elif mask_combine == "and":
                mask = np.asarray((ms1 > 0.5) & (ms2 > 0.5), dtype=np.float32)
            elif mask_combine == "t1":
                mask = np.asarray(ms1 > 0.5, dtype=np.float32)
            elif mask_combine == "t2":
                mask = np.asarray(ms2 > 0.5, dtype=np.float32)
            else:
                raise ValueError(f"Unknown mask_combine: {mask_combine}")

            img1_r = _resize(img1, trainsize, "bilinear")
            img2_r = _resize(img2, trainsize, "bilinear")
            mask_r = _resize(mask, trainsize, "nearest")
            mask_r = np.asarray(mask_r[None, ...] if mask_r.ndim == 2 else mask_r, dtype=np.float32)

            filename = f"sample_{sample_idx:06d}"
            np.save(t1_dir / f"{filename}.npy", img1_r.astype(np.float32))
            np.save(t2_dir / f"{filename}.npy", img2_r.astype(np.float32))
            np.save(m_dir / f"{filename}.npy", mask_r.astype(np.float32))

            kind = "positive" if bool(pos[s]) else ("context" if bool(ctx[s]) else "far_negative")
            if kind == "positive":
                pos_cnt += 1
            elif kind == "context":
                ctx_cnt += 1
            else:
                far_cnt += 1

            slice_records.append(
                {
                    "filename": filename,
                    "patient_id": pid,
                    "slice_idx": int(s),
                    "kind": kind,
                }
            )
            sample_idx += 1

        vol_info["per_patient_record"] = per_patient_record

    return pos_cnt, ctx_cnt, far_cnt, slice_records


def _normalize_stack_per_slice(vol: np.ndarray) -> np.ndarray:
    """Per-slice z-score normalize a 3-D volume (D, H, W).

    This matches the per-slice normalisation used for legacy 2.5D stacks,
    applied per z-slice so the cached full volume has identical statistics
    to the independent 2.5D slice npy files.
    """
    v = np.asarray(vol, dtype=np.float32)
    if v.ndim != 3:
        raise ValueError(f"_normalize_stack_per_slice expects (D,H,W), got {v.shape}")
    out = np.empty_like(v)
    for s in range(v.shape[0]):
        sl = v[s]
        fg = sl > sl.mean()
        if fg.sum() < 32:
            # fallback: robust global clip instead of foreground mask
            mu = float(sl.mean()); sd = float(sl.std()) if sl.std() > 1e-6 else 1.0
        else:
            mu = float(sl[fg].mean()); sd = float(sl[fg].std()) if sl[fg].std() > 1e-6 else 1.0
        out[s] = np.clip((sl - mu) / sd, -3.0, 3.0).astype(np.float32, copy=False)
    return out


def build_and_save(
    raw_train_root: Path,
    raw_test_root: Path,
    train_out: Path,
    test_out: Path,
    trainsize: int,
    mask_combine: str,
    context_window: int,
    far_keep_p: Optional[float],
    far_keep_N: Optional[int],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> None:
    rng = random.Random(seed)
    np.random.seed(seed)

    for d in (train_out, test_out):
        d.mkdir(parents=True, exist_ok=True)

    # ---------------------- train root into train / val --------------------
    all_train_patients = _enumerate_patients(raw_train_root)
    train_pats, val_pats = patient_level_split(all_train_patients, train_ratio, val_ratio, seed)

    def _load_patients(patient_dirs: List[Path]) -> List[Tuple[Path, Dict[str, object]]]:
        entries: List[Tuple[Path, Dict[str, object]]] = []
        skipped: List[str] = []
        for pdir in patient_dirs:
            pair = _find_patient_pairs(pdir)
            if pair is None:
                skipped.append(pdir.name)
                continue
            t1i, t1m, t2i, t2m = pair
            try:
                t1_vol = _load_volume(t1i, is_mask=False)
                t2_vol = _load_volume(t2i, is_mask=False)
                t1m_vol = _load_volume(t1m, is_mask=True)
                t2m_vol = _load_volume(t2m, is_mask=True)
            except Exception as e:
                skipped.append(f"{pdir.name}: {e}")
                continue

            # ------------------------------------------------------------------
            # 不同模态在 DICOM 重建成 NIfTI 时 Z 维度层数可能不严格一致
            # （例如 T1=16 层、T2=13 层）；对图像和 mask 一起沿 Z 对齐到
            # 两者的 min(D_t1, D_t2)，前端中心裁剪，后端 0-pad，保证 | &
            # 操作时 shape 完全匹配。
            # ------------------------------------------------------------------
            def _align_z(arr: np.ndarray, target_z: int) -> np.ndarray:
                dz = int(arr.shape[0]) - int(target_z)
                if dz == 0:
                    return arr
                if dz > 0:
                    # 层数更多 → 中心裁剪，优先保留中间层（通常病灶靠中心）
                    start = dz // 2
                    return arr[start : start + target_z]
                # 层数更少 → 两端 0-pad
                pad_before = (-dz) // 2
                pad_after = (-dz) - pad_before
                pad_width = [(pad_before, pad_after)] + [(0, 0)] * (arr.ndim - 1)
                return np.pad(arr, pad_width, mode="constant", constant_values=0)

            min_z = min(t1_vol.shape[0], t2_vol.shape[0],
                        t1m_vol.shape[0], t2m_vol.shape[0])
            t1_vol  = _align_z(t1_vol,  min_z)
            t2_vol  = _align_z(t2_vol,  min_z)
            t1m_vol = _align_z(t1m_vol, min_z)
            t2m_vol = _align_z(t2m_vol, min_z)

            # combine masks for the purpose of slice selection (we want to keep
            # slices that are relevant for ANY of the two modalities).
            sel_mask = np.asarray((t1m_vol > 0.5) | (t2m_vol > 0.5), dtype=np.float32)
            strat = _slice_strategy(sel_mask, context_window, far_keep_p, far_keep_N, rng)
            info: Dict[str, object] = {
                "t1_vol": t1_vol,
                "t2_vol": t2_vol,
                "t1m_vol": t1m_vol,
                "t2m_vol": t2m_vol,
            }
            info.update(strat)
            entries.append((pdir, info))
        if skipped:
            print(f"[WARN] skipped {len(skipped)} patients:")
            for s in skipped[:10]:
                print(f"       - {s}")
            if len(skipped) > 10:
                print(f"       ... and {len(skipped) - 10} more")
        return entries

    print("Loading patients under", raw_train_root, "for train/val ...")
    train_entries = _load_patients(train_pats)
    val_entries = _load_patients(val_pats)

    # ---------------------- test root (all patients, no resplit) ----------
    all_test_patients = _enumerate_patients(raw_test_root)
    test_entries_all = _load_patients(all_test_patients)

    # ---------------------- test slice policy (less aggressive) -----------
    # For the held-out test set we want a faithful evaluation: keep *every*
    # positive/context slice and keep more far negatives than train (e.g. 30%)
    def _relax_test(entry: Tuple[Path, Dict[str, object]]) -> Tuple[Path, Dict[str, object]]:
        pdir, info = entry
        D = int(len(info["positive"]))  # type: ignore[arg-type]
        pos = np.asarray(info["positive"])  # type: ignore[index]
        ctx = np.asarray(info["context"])  # type: ignore[index]
        far_idx = [s for s in range(D) if not bool(pos[s]) and not bool(ctx[s])]
        keep = np.zeros(D, dtype=bool)
        keep[pos] = True
        keep[ctx] = True
        # keep one every 2 -> 50% for a realistic held-out evaluation
        for i in range(0, len(far_idx), 2):
            keep[far_idx[i]] = True
        info["keep"] = keep
        return pdir, info

    test_entries = [_relax_test(e) for e in test_entries_all]

    # ---------------------- save ------------------------------------------
    print("Saving train split ...")
    train_pos, train_ctx, train_far, train_slices = _save_split(
        train_out, "train", train_entries, mask_combine, trainsize
    )
    print("Saving val split ...")
    val_pos, val_ctx, val_far, val_slices = _save_split(
        train_out, "val", val_entries, mask_combine, trainsize
    )
    print("Saving test split ...")
    test_pos, test_ctx, test_far, test_slices = _save_split(
        test_out, "test", test_entries, mask_combine, trainsize
    )

    def _ppr(patient_entries: List[Tuple[Path, Dict[str, object]]]) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        for _pdir, info in patient_entries:
            if "per_patient_record" in info:
                out.append(info["per_patient_record"])  # type: ignore[index]
        return out

    train_report = {
        "split": "train",
        "patients": {"total": len(train_pats), "loaded": len(train_entries)},
        "samples": {
            "total": len(train_slices),
            "positive": train_pos,
            "context_negative": train_ctx,
            "far_negative": train_far,
        },
        "slice_records": train_slices,
        "per_patient": _ppr(train_entries),
    }
    val_report = {
        "split": "val",
        "patients": {"total": len(val_pats), "loaded": len(val_entries)},
        "samples": {
            "total": len(val_slices),
            "positive": val_pos,
            "context_negative": val_ctx,
            "far_negative": val_far,
        },
        "slice_records": val_slices,
        "per_patient": _ppr(val_entries),
    }
    test_report = {
        "split": "test",
        "patients": {"total": len(all_test_patients), "loaded": len(test_entries)},
        "samples": {
            "total": len(test_slices),
            "positive": test_pos,
            "context_negative": test_ctx,
            "far_negative": test_far,
        },
        "slice_records": test_slices,
        "per_patient": _ppr(test_entries),
    }

    (train_out / "preprocess_report.json").write_text(
        json.dumps({"train": train_report, "val": val_report}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (test_out / "preprocess_report.json").write_text(
        json.dumps({"test": test_report}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    def _pct(n: int, total: int) -> str:
        if total <= 0:
            return "0.0%"
        return f"{100.0 * n / total:.2f}%"

    print()
    print("================ Preprocessing summary ================")
    print(f"train samples : total={len(train_slices):>6d}  pos={train_pos:>5d}({_pct(train_pos,len(train_slices))})  "
          f"ctx={train_ctx:>5d}({_pct(train_ctx,len(train_slices))})  far={train_far:>5d}({_pct(train_far,len(train_slices))})")
    print(f"val   samples : total={len(val_slices):>6d}  pos={val_pos:>5d}({_pct(val_pos,len(val_slices))})  "
          f"ctx={val_ctx:>5d}({_pct(val_ctx,len(val_slices))})  far={val_far:>5d}({_pct(val_far,len(val_slices))})")
    print(f"test  samples : total={len(test_slices):>6d}  pos={test_pos:>5d}({_pct(test_pos,len(test_slices))})  "
          f"ctx={test_ctx:>5d}({_pct(test_ctx,len(test_slices))})  far={test_far:>5d}({_pct(test_far,len(test_slices))})")
    print("reports:")
    print(f"  - {train_out / 'preprocess_report.json'}")
    print(f"  - {test_out / 'preprocess_report.json'}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stratified 3D->2.5D preprocessing for the paired MRI dataset.",
    )
    p.add_argument("--raw-train-root", type=Path, default=Path("/root/CFA-NPC-main-main/dataset/TrainDataset"))
    p.add_argument("--raw-test-root", type=Path, default=Path("/root/CFA-NPC-main-main/dataset/TestDataset"))
    p.add_argument("--train-out", type=Path, default=Path("/root/CFA-NPC-main-main/TrainDataset_strat"))
    p.add_argument("--test-out", type=Path, default=Path("/root/CFA-NPC-main-main/TestDataset_strat"))
    p.add_argument("--trainsize", type=int, default=256)
    p.add_argument("--mask-combine", type=str, default="or", choices=["or", "and", "t1", "t2"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.2)

    p.add_argument(
        "--context-window",
        type=int,
        default=3,
        help="Keep all empty slices within ±N of any positive slice (context negatives).",
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--far-keep-p",
        type=float,
        default=None,
        help="For slices outside context window: keep them with probability p (0.1-0.2 recommended).",
    )
    group.add_argument(
        "--far-keep-N",
        type=int,
        default=3,
        help="For slices outside context window: keep 1 every N slices (default: every 3rd).",
    )

    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    if (0.8 + 0.2) != (args.train_ratio + args.val_ratio):
        print(f"[WARN] train_ratio + val_ratio = {args.train_ratio + args.val_ratio:.3f}. "
              "For this script train+val must cover the training root fully.")
    build_and_save(
        raw_train_root=args.raw_train_root,
        raw_test_root=args.raw_test_root,
        train_out=args.train_out,
        test_out=args.test_out,
        trainsize=int(args.trainsize),
        mask_combine=args.mask_combine,
        context_window=int(args.context_window),
        far_keep_p=args.far_keep_p,
        far_keep_N=args.far_keep_N,
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )
