#!/usr/bin/env python3
"""
Per-patient 3D MRI preprocessing pipeline (Steps 1, 2, 4 of the spec).

==========================================================================
 WHY A NEW SCRIPT (instead of preprocess_dataset_stratified.py)?
==========================================================================
The old script dumps EVERY patient's 2.5D slices into one flat
``images_t1/`` / ``images_t2/`` / ``masks/`` directory.  Even though a
``slice_records`` JSON records each sample's ``patient_id``, the 2.5D
neighbour-stacking at training time can still borrow slices from the
*next* patient at folder-order boundaries (patient A's last slice next
to patient B's first slice) -> anatomically meaningless context.

The user explicitly requires: "每个患者的数据要单独存放，不要用旧方法把所有
患者的切片存放在一起".  This script honours that by writing one folder
per patient, each containing the full preprocessed 3-D volume.  The
2.5D stack (Step 3, context=2) is then built ONLINE from that single
patient's volume, so cross-patient context leakage is *structurally
impossible*.

==========================================================================
 WHAT IT DOES (per patient)
==========================================================================
  Step 1  Resample T1 / T2 / masks to isotropic 1.0 x 1.0 x 1.0 mm.
          SimpleITK ResampleImageFilter is used (spacing + direction
          aware).  Images -> BSpline (smooth), masks -> NearestNeighbour
          (keeps labels binary {0,1}).
  Step 2  Intensity normalization on the resampled image volume:
          non-background-voxel (value > 0) Z-score, then clip to [-3, 3].
          Applied per modality (T1, T2) on the full 3-D volume so the
          statistics match what the network sees at train time.
          T1/T2 are aligned along Z (centre-crop / zero-pad to min depth).
          Masks are combined with OR (T1 | T2) for slice selection and
          as the training target.
  Step 4  Per-slice sampling policy on the combined mask:
            * positive slice (any foreground)           -> keep, weight 1.0
            * edge slice   (empty, within +/-2 of pos)  -> keep, weight 0.5
            * far background (empty, beyond +/-2)       -> keep 15%, weight 0.1
          For the test split we keep all positive + all edge + 50% far
          background so lesion-level evaluation is faithful.

  Storage (per patient):
    <out>/<split>/<patient_id>/
        t1.npy       float16 (D, H, W)   resampled + z-scored + clipped
        t2.npy       float16 (D, H, W)
        mask.npy     uint8  (D, H, W)    combined T1|T2, resampled (nearest)
        slices.json               list of {slice_idx, kind, weight}
    <out>/manifest.json           global split assignment + per-patient stats

  Steps 3 (2.5D 5-channel stack, context=2), 5 (ROI patch crop 256x256,
  70% ROI / 30% random) and 6 (geometric + intensity augmentation) are
  applied ONLINE by ``per_patient_dataset.PerPatientDataset`` during
  training.  float16 storage halves disk usage (z-scored values in
  [-3,3] are well within float16 precision); images are cast back to
  float32 at load time.

Reuses:
  * ``preprocess_dataset_stratified._pair_files`` / ``_find_patient_pairs``
    for robust T1/T2 image+mask pairing across the dataset.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk

# Reuse the robust file-pairing helpers from the existing stratified script
# (handles the "_4" image suffix vs "_5" mask suffix mismatch in this dataset).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess_dataset_stratified import (  # noqa: E402
    _find_patient_pairs,
    _enumerate_patients,
    patient_level_split,
)


# ---------------------------------------------------------------------------
# Step 1: spacing-aware isotropic resampling (SimpleITK, direction-aware)
# ---------------------------------------------------------------------------
def _resample_iso(itk_img: sitk.Image,
                  target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                  is_mask: bool = False) -> sitk.Image:
    """Resample a SimpleITK image to ``target_spacing`` (mm), preserving
    origin / direction.  Images use BSpline, masks use NearestNeighbour
    so labels stay strictly binary.

    Why BSpline (order 3) and not linear for images?
      BSpline gives a C2-continuous, less-aliased result for the smooth
      anatomical intensities of MR, which matters after the 6-8 mm -> 1 mm
      Z up-sampling (a 6x-8x interpolation) where linear would introduce
      visible staircase artefacts on small lymph-node borders.
    Why nearest for masks?
      Any higher-order interpolation turns the {0,1} label into fractional
      values, fabricating false-positive pixels or eroding tiny lesions
      after thresholding.  Nearest is the segmentation gold standard.
    """
    orig_spacing = itk_img.GetSpacing()
    orig_size = itk_img.GetSize()  # (sx, sy, sz) order per SimpleITK
    # new_size[i] = round(orig_size[i] * orig_spacing[i] / target_spacing[i])
    new_size = [int(round(osz * ospc / tspc))
                for osz, ospc, tspc in zip(orig_size, orig_spacing, target_spacing)]
    # Guard against zero-size axes after rounding.
    new_size = [max(1, s) for s in new_size]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(itk_img.GetOrigin())
    resampler.SetOutputDirection(itk_img.GetDirection())
    resampler.SetOutputPixelType(sitk.sitkFloat32 if not is_mask else sitk.sitkUInt8)
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkBSpline)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(itk_img)


def _load_and_resample(path: Path, is_mask: bool,
                       target_spacing: Tuple[float, float, float]) -> np.ndarray:
    itk = sitk.ReadImage(str(path))
    if is_mask:
        # Binarise first (some labels may have >1 values), then resample.
        itk = sitk.Cast(itk > 0, sitk.sitkUInt8)
        itk = _resample_iso(itk, target_spacing, is_mask=True)
        arr = sitk.GetArrayFromImage(itk)  # (D, H, W)
        return (np.asarray(arr) > 0).astype(np.uint8)
    else:
        itk = sitk.Cast(itk, sitk.sitkFloat32)
        itk = _resample_iso(itk, target_spacing, is_mask=False)
        return np.asarray(sitk.GetArrayFromImage(itk), dtype=np.float32)


# ---------------------------------------------------------------------------
# Step 2: intensity normalization (non-background Z-score + clip [-3, 3])
# ---------------------------------------------------------------------------
def _normalize_volume(vol: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score on non-background voxels (value > 0) then clip to [-3, 3].

    Background (air/empty) voxels are 0 in these MR volumes; including them
    in mean/std would bias the statistics toward 0 and under-state the
    real tissue contrast.  We compute mean/std on the foreground only,
    then normalise the WHOLE volume (background becomes a small negative
    value, which is fine) and clip.
    """
    v = vol.astype(np.float32, copy=False)
    fg = v > 0.0
    if fg.any():
        mean = float(v[fg].mean())
        std = float(v[fg].std())
    else:
        mean, std = 0.0, 1.0
    if std < eps:
        std = eps
    out = (v - mean) / std
    out = np.clip(out, -3.0, 3.0)
    return out.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Shape alignment between T1 and T2 along ALL axes (D, H, W)
# ---------------------------------------------------------------------------
def _align_axis(arr: np.ndarray, axis: int, target: int) -> np.ndarray:
    """Align ``arr`` so that ``arr.shape[axis] == target``.

    Larger -> centre-crop (keeps the middle, where lesions usually are).
    Smaller -> zero-pad both ends.
    """
    cur = int(arr.shape[axis])
    if cur == target:
        return arr
    if cur > target:
        start = (cur - target) // 2
        sl = [slice(None)] * arr.ndim
        sl[axis] = slice(start, start + target)
        return arr[tuple(sl)]
    # smaller -> zero-pad both ends
    pad_before = (target - cur) // 2
    pad_after = (target - cur) - pad_before
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (pad_before, pad_after)
    return np.pad(arr, pad_width, mode="constant", constant_values=0)


def _align_to_shape(arr: np.ndarray, shape: Tuple[int, int, int]) -> np.ndarray:
    """Centre-crop / zero-pad ``arr`` to exactly ``shape`` along (D, H, W)."""
    out = arr
    for ax, tgt in enumerate(shape):
        out = _align_axis(out, ax, int(tgt))
    return out


# ---------------------------------------------------------------------------
# Step 4: per-slice sampling policy with explicit weights
# ---------------------------------------------------------------------------
def _slice_policy(mask_vol: np.ndarray,
                  edge_window: int,
                  far_keep_p: float,
                  rng: random.Random) -> List[Dict[str, object]]:
    """Return list of {slice_idx, kind, weight} for the slices to KEEP.

    kind \in {'positive', 'edge', 'background'}
      positive  : mask has any foreground -> weight 1.0  (keep all)
      edge      : empty, within +/-edge_window of any positive -> weight 0.5
      background: empty, beyond +/-edge_window -> keep ``far_keep_p`` fraction, weight 0.1
    """
    D = int(mask_vol.shape[0])
    positive = np.zeros(D, dtype=bool)
    for s in range(D):
        if np.any(mask_vol[s] > 0):
            positive[s] = True
    pos_idx = np.where(positive)[0]

    edge = np.zeros(D, dtype=bool)
    if pos_idx.size > 0:
        lo = max(0, int(pos_idx.min()) - edge_window)
        hi = min(D - 1, int(pos_idx.max()) + edge_window)
        edge[lo:hi + 1] = True
    edge = edge & (~positive)

    far = (~positive) & (~edge)
    far_idx = np.where(far)[0].tolist()

    records: List[Dict[str, object]] = []
    for s in range(D):
        if positive[s]:
            records.append({"slice_idx": int(s), "kind": "positive", "weight": 1.0})
        elif edge[s]:
            records.append({"slice_idx": int(s), "kind": "edge", "weight": 0.5})
    # far background: keep fraction
    kept_far = [i for i in far_idx if rng.random() < far_keep_p]
    for s in kept_far:
        records.append({"slice_idx": int(s), "kind": "background", "weight": 0.1})

    records.sort(key=lambda r: r["slice_idx"])
    return records


# ---------------------------------------------------------------------------
# Per-patient processing
# ---------------------------------------------------------------------------
def _process_patient(patient_dir: Path,
                     target_spacing: Tuple[float, float, float],
                     edge_window: int,
                     far_keep_p: float,
                     rng: random.Random) -> Optional[Dict[str, object]]:
    """Return a dict of {pid, t1, t2, mask, slices} or None on failure."""
    pair = _find_patient_pairs(patient_dir)
    if pair is None:
        return None
    t1i, t1m, t2i, t2m = pair
    try:
        t1 = _load_and_resample(t1i, is_mask=False, target_spacing=target_spacing)
        t2 = _load_and_resample(t2i, is_mask=False, target_spacing=target_spacing)
        m1 = _load_and_resample(t1m, is_mask=True,  target_spacing=target_spacing)
        m2 = _load_and_resample(t2m, is_mask=True,  target_spacing=target_spacing)
    except Exception as e:
        print(f"[WARN] {patient_dir.name}: load/resample failed: {e}")
        return None

    # Align ALL axes across modalities. T1/T2 are separate acquisitions and
    # may differ in Z-depth (e.g. 16 vs 13 slices) AND in-plane FOV (e.g.
    # 230mm vs 260mm) after isotropic resampling. We crop/pad each volume
    # to the per-axis min so the T1|T2 mask combine is broadcast-safe and
    # no lesion is fabricated/lost by interpolation.
    min_d = min(t1.shape[0], t2.shape[0], m1.shape[0], m2.shape[0])
    min_h = min(t1.shape[1], t2.shape[1], m1.shape[1], m2.shape[1])
    min_w = min(t1.shape[2], t2.shape[2], m1.shape[2], m2.shape[2])
    if min(min_d, min_h, min_w) == 0:
        print(f"[WARN] {patient_dir.name}: zero-size axis after resample "
              f"(t1={t1.shape} t2={t2.shape})")
        return None
    target_shape = (min_d, min_h, min_w)
    t1 = _align_to_shape(t1, target_shape); t2 = _align_to_shape(t2, target_shape)
    m1 = _align_to_shape(m1, target_shape); m2 = _align_to_shape(m2, target_shape)

    # Step 2: intensity normalization (per modality, full volume).
    t1 = _normalize_volume(t1)
    t2 = _normalize_volume(t2)

    # Combined mask (OR of T1/T2 node labels).
    mask = ((m1 > 0) | (m2 > 0)).astype(np.uint8)

    # Step 4: slice sampling policy with weights.
    slices = _slice_policy(mask, edge_window, far_keep_p, rng)

    return {
        "pid": patient_dir.name,
        "t1": t1,
        "t2": t2,
        "mask": mask,
        "slices": slices,
    }


def _save_patient(out_split_dir: Path, info: Dict[str, object]) -> Dict[str, object]:
    pid = str(info["pid"])
    pdir = out_split_dir / pid
    pdir.mkdir(parents=True, exist_ok=True)
    # float16 for images (z-scored values in [-3,3] fit float16 precision);
    # uint8 for the binary mask. Cast back to float32 at load time.
    t1 = np.asarray(info["t1"], dtype=np.float16)
    t2 = np.asarray(info["t2"], dtype=np.float16)
    mk = np.asarray(info["mask"], dtype=np.uint8)
    np.save(pdir / "t1.npy", t1)
    np.save(pdir / "t2.npy", t2)
    np.save(pdir / "mask.npy", mk)
    slices = info["slices"]
    (pdir / "slices.json").write_text(
        json.dumps(slices, ensure_ascii=False), encoding="utf-8"
    )

    n_pos = sum(1 for r in slices if r["kind"] == "positive")
    n_edge = sum(1 for r in slices if r["kind"] == "edge")
    n_bg = sum(1 for r in slices if r["kind"] == "background")
    return {
        "patient_id": pid,
        "depth": int(mk.shape[0]),
        "shape": [int(mk.shape[0]), int(mk.shape[1]), int(mk.shape[2])],
        "kept_slices": len(slices),
        "positive": n_pos,
        "edge": n_edge,
        "background": n_bg,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def build(raw_train_root: Path, raw_test_root: Path, out_root: Path,
          target_spacing: Tuple[float, float, float],
          edge_window: int, far_keep_p_train: float, far_keep_p_test: float,
          train_ratio: float, val_ratio: float, seed: int) -> None:
    rng = random.Random(seed)
    np.random.seed(seed)
    out_root.mkdir(parents=True, exist_ok=True)

    # ---- enumerate + patient-level split of the training root ----
    all_train = _enumerate_patients(raw_train_root)
    train_pats, val_pats = patient_level_split(all_train, train_ratio, val_ratio, seed)
    all_test = _enumerate_patients(raw_test_root)
    print(f"[info] train patients={len(train_pats)}  val patients={len(val_pats)}  "
          f"test patients={len(all_test)}")

    manifest: Dict[str, object] = {
        "target_spacing": list(target_spacing),
        "edge_window": edge_window,
        "far_keep_p_train": far_keep_p_train,
        "far_keep_p_test": far_keep_p_test,
        "seed": seed,
        "splits": {"train": [], "val": [], "test": []},
        "patients": {},
    }

    def _run(patients: List[Path], split: str, far_keep_p: float) -> None:
        out_split_dir = out_root / split
        out_split_dir.mkdir(parents=True, exist_ok=True)
        n_ok = 0
        for i, pdir in enumerate(patients):
            info = _process_patient(pdir, target_spacing, edge_window, far_keep_p, rng)
            if info is None:
                continue
            stats = _save_patient(out_split_dir, info)
            manifest["splits"][split].append(pdir.name)  # type: ignore[index]
            manifest["patients"][pdir.name] = {**stats, "split": split}  # type: ignore[index]
            n_ok += 1
            if (i + 1) % 25 == 0 or (i + 1) == len(patients):
                print(f"  [{split}] {i+1}/{len(patients)} processed "
                      f"(kept {n_ok}, pid={pdir.name})")
        print(f"[{split}] done: {n_ok}/{len(patients)} patients saved under {out_split_dir}")

    print("[step] processing train split ...")
    _run(train_pats, "train", far_keep_p_train)
    print("[step] processing val split ...")
    _run(val_pats, "val", far_keep_p_train)  # val uses same policy as train
    print("[step] processing test split ...")
    _run(all_test, "test", far_keep_p_test)

    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ---- summary ----
    def _agg(split: str) -> Tuple[int, int, int, int]:
        pats = [manifest["patients"][p] for p in manifest["splits"][split]]  # type: ignore[index]
        if not pats:
            return 0, 0, 0, 0
        s = sum(p["kept_slices"] for p in pats)
        pos = sum(p["positive"] for p in pats)
        edge = sum(p["edge"] for p in pats)
        bg = sum(p["background"] for p in pats)
        return s, pos, edge, bg

    print("\n================ Per-patient preprocessing summary ================")
    for sp in ("train", "val", "test"):
        s, pos, edge, bg = _agg(sp)
        n = len(manifest["splits"][sp])  # type: ignore[index]
        print(f"  {sp:5s}: patients={n:<4d} kept_slices={s:<5d}  "
              f"pos={pos} edge={edge} bg={bg}")
    print(f"  manifest -> {out_root / 'manifest.json'}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-patient 3D MRI preprocessing (Steps 1,2,4 of the spec)."
    )
    p.add_argument("--raw-train-root", type=Path,
                   default=Path("/root/CFA-NPC-main-main/dataset/TrainDataset"))
    p.add_argument("--raw-test-root", type=Path,
                   default=Path("/root/CFA-NPC-main-main/dataset/TestDataset"))
    p.add_argument("--out-root", type=Path,
                   default=Path("/root/CFA-NPC-main-main/dataset_per_patient"))
    p.add_argument("--target-spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                   help="Isotropic target spacing (sx sy sz) in mm.")
    p.add_argument("--edge-window", type=int, default=2,
                   help="+/-N slices around positives counted as 'edge' (weight 0.5).")
    p.add_argument("--far-keep-p-train", type=float, default=0.15,
                   help="Fraction of far-background slices kept for train/val (weight 0.1).")
    p.add_argument("--far-keep-p-test", type=float, default=0.5,
                   help="Fraction of far-background slices kept for test (faithful eval).")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    try:
        build(
            raw_train_root=args.raw_train_root,
            raw_test_root=args.raw_test_root,
            out_root=args.out_root,
            target_spacing=tuple(float(x) for x in args.target_spacing),
            edge_window=int(args.edge_window),
            far_keep_p_train=float(args.far_keep_p_train),
            far_keep_p_test=float(args.far_keep_p_test),
            train_ratio=float(args.train_ratio),
            val_ratio=float(args.val_ratio),
            seed=int(args.seed),
        )
    except Exception:
        traceback.print_exc()
        sys.exit(1)
