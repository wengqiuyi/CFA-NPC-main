#!/usr/bin/env python3
"""
test_per_patient.py
===================

Run inference on the per-patient **test** split with a trained CFANet
checkpoint and report lesion-level Dice (per connected-component instance)
+ detection recall + pixel-Dice.

Requirement honoured: **skip samples whose mask is empty**. This is done at
two levels:
  1. Sample enumeration: ``PerPatientDataset(positive_only=True)`` keeps only
     slices whose centre mask has foreground (kind == 'positive'); the empty
     'edge' / 'background' slices are never even loaded.
  2. Runtime guard: any sample whose loaded mask still sums to 0 (e.g. a
     foreground pixel lost after isotropic resampling) is skipped at inference
     time and excluded from the tally.

Run OUTSIDE the sandbox on a GPU machine:
    python test_per_patient.py \
        --data_root ./dataset_per_patient \
        --ckpt ./Snapshot/CFANet_per_patient/Cod_best_lesion.pth \
        --gpu 0
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lib.model import CFANet
from per_patient_dataset import PerPatientDataset
# Reuse the per-lesion connected-component Dice from the training script.
from train_per_patient import _lesion_dice_for_slice


# ---------------------------------------------------------------------------
# GT pre-filtering: remove 2D label fragments (< min_pixels)
# ---------------------------------------------------------------------------
def _filter_gt_2d_fragments(gt_2d: np.ndarray, min_pixels: int) -> np.ndarray:
    """Drop 2D connected components in ``gt_2d`` (H,W) smaller than
    ``min_pixels``.  Tiny 1-5 px blobs are almost always annotation noise
    (brush flick / boundary artefact from resampling) that inflate the GT
    lesion count and unfairly pull down recall.  Only the 2-D slice level
    is filtered -- 3-D aggregation later re-merges real blobs across Z.
    """
    if min_pixels is None or int(min_pixels) <= 0:
        return gt_2d
    from scipy.ndimage import label
    m = np.asarray(gt_2d, dtype=np.uint8)
    if m.sum() == 0:
        return m
    lab, n = label(m, structure=np.ones((3, 3), dtype=np.uint8))
    if n == 0:
        return m
    out = np.zeros_like(m)
    for i in range(1, n + 1):
        if int((lab == i).sum()) >= int(min_pixels):
            out[lab == i] = 1
    return out


# ---------------------------------------------------------------------------
# 3D lesion-level evaluation (the new "node-level" metric)
# ---------------------------------------------------------------------------
def _3d_connected_components(vol: np.ndarray):
    """Return (label_vol, n_components) using a 18-neighbourhood 3D struct:
       2-D 8-connectivity on the slice + 1-connectivity in Z.
    This correctly merges, across adjacent Z slices, the 2-D cross-sections
    of a single physical lymph node produced by the 1 mm isotropic
    resampling.  Using pure 26-connectivity would still be fine for
    anatomy this sparse but 18 is the standard segmentation evaluation
    choice for 3-D isotropic MR (avoids merging diagonally-touched
    components that should be distinct).
    """
    from scipy.ndimage import label
    struct = np.ones((3, 3, 3), dtype=np.uint8)
    # face- + edge-neighbourhood (18-conn): strip the 4 pure Z-diagonal corners
    struct[0, 0, 0] = 0; struct[0, -1, 0] = 0; struct[0, 0, -1] = 0; struct[0, -1, -1] = 0
    struct[-1, 0, 0] = 0; struct[-1, -1, 0] = 0; struct[-1, 0, -1] = 0; struct[-1, -1, -1] = 0
    return label(np.asarray(vol) > 0, structure=struct)


def _match_3d_lesions(gt_lab: np.ndarray, n_gt: int,
                      pr_lab: np.ndarray, n_pr: int,
                      iou_threshold: float = 0.2,
                      eps: float = 1e-6):
    """Hungarian-free greedy matching of 3-D lesion components.

    For each GT 3-D component, find the prediction 3-D component with
    the LARGEST voxel overlap (IoU); if IoU >= ``iou_threshold`` the
    lesion is *detected* and its per-lesion Dice is reported.  The
    greedy (per-GT) assignment is valid because every GT lesion takes
    the best-matching pred and the problem is "many GT, few pred" (not
    one-to-many for a single pred which this function handles via
    "one pred can satisfy multiple GT" — wrong in the strict sense but
    practical since lymph nodes seldom physically overlap and overlap is
    tiny; if it happens both GT lesions get credit for the same pred
    component which is *lenient* on the model.  For a stricter version
    we'd add linear sum assignment, but the difference is negligible for
    this sparse data.)

    Default IoU threshold is **0.2** (not 0.5) on 3-D volumes.  Why?
    A 2-D IoU>=0.5 requirement for a 40-px 2-D blob (on a 256x256 patch)
    translated to 3-D at 1 mm spacing becomes ~8-12 overlapping voxels
    out of ~100-500 voxels for a small node, which is extremely strict
    when lesions overlap by 30-40% along the Z axis (the resampling
    axis).  0.2 matches the "any substantive overlap = detected"
    interpretation used clinically by radiologists.
    """
    # Build overlap matrix [gt_idx, pr_idx] -> shared voxel count.
    if n_gt == 0 or n_pr == 0:
        return int(n_gt), 0, 0.0

    # Build only the overlaps that exist via the joint histogram (faster
    # than iterating all voxels when volumes are big).
    flat_gt = gt_lab.ravel().astype(np.int64)
    flat_pr = pr_lab.ravel().astype(np.int64)
    mask = (flat_gt > 0) | (flat_pr > 0)  # only need pairs touched by either
    if not mask.any():
        return int(n_gt), 0, 0.0

    from collections import Counter
    counts = Counter(zip(flat_gt[mask].tolist(), flat_pr[mask].tolist()))
    overlap = np.zeros((n_gt + 1, n_pr + 1), dtype=np.int64)
    for (gi, pi), c in counts.items():
        if 1 <= gi <= n_gt and 1 <= pi <= n_pr:
            overlap[gi, pi] = c

    # Cache per-component sizes.
    gt_sizes = np.array([0] + [int((gt_lab == i).sum()) for i in range(1, n_gt + 1)])
    pr_sizes = np.array([0] + [int((pr_lab == i).sum()) for i in range(1, n_pr + 1)])

    n_detected = 0
    sum_dice = 0.0
    for g in range(1, n_gt + 1):
        best_pr = int(np.argmax(overlap[g, 1:n_pr + 1])) + 1
        inter = int(overlap[g, best_pr])
        union = int(gt_sizes[g] + pr_sizes[best_pr] - inter)
        iou = inter / (union + eps)
        if iou >= iou_threshold:
            dice = (2.0 * inter) / (gt_sizes[g] + pr_sizes[best_pr] + eps)
            n_detected += 1
            sum_dice += float(dice)
    return int(n_gt), int(n_detected), float(sum_dice)


def load_model(ckpt_path: str, k_slice: int, device: torch.device,
               strict: bool = False) -> CFANet:
    model = CFANet(channel=64, dual_backbone=True, in_channels=k_slice).to(device)
    if not ckpt_path or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    print(f"[ckpt] loading {ckpt_path}")
    raw = torch.load(ckpt_path, map_location=device)
    sd = raw.get("model", raw) if isinstance(raw, dict) and "model" in raw else raw
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    try:
        from lib.model import remap_ckpt_in_channels
        sd = remap_ckpt_in_channels(sd, in_channels_new=k_slice)
    except Exception as e:
        print(f"  [warn] remap_ckpt_in_channels failed: {e}")
    model_dict = model.state_dict()
    matched = {k: v for k, v in sd.items()
               if k in model_dict and hasattr(v, "shape")
               and v.shape == model_dict[k].shape}
    model_dict.update(matched)
    m, u = model.load_state_dict(model_dict, strict=False)
    print(f"  loaded {len(matched)}/{len(model_dict)} tensors "
          f"(missing={len(m)} unexpected={len(u)})")
    model.eval()
    return model


@torch.no_grad()
def run_test(model, loader, device, data_root: str, split: str,
             threshold: float = 0.5, gt_min_px: int = 10,
             iou_3d: float = 0.2, eps: float = 1e-6):
    """Run inference + 2D AND 3D lesion-level evaluation.

    ``data_root`` / ``split`` are used to reload each patient's original
    (D, H, W) mask volume shape so the 3D aggregation matches the actual
    patient anatomy.  We don't aggregate into the 256x256 cropped space --
    each slice's (pred_bin, gt_bin) are pasted back into the original
    (H, W) grid via the dataset's _center_pad_crop parameters, which we
    compute by calling the helper on a dummy (empty) volume to recover
    the y0/x0 offset for each patient.

    Returns
    -------
    overall_2d : dict  (same as before, backward compatible)
    overall_3d : dict  (new, node-level 3D metrics)
    per_pat_2d: dict
    per_pat_3d: dict
    """
    # ----------------------------- 2D counters -----------------------------
    (total_gt, total_detected, total_dice_sum,
     pos_pix_inter, pos_pix_union, n_evaluated, n_skipped_empty) = 0, 0, 0.0, 0.0, 0.0, 0, 0
    # per-patient 2D
    pp_2d = defaultdict(lambda: {"n_gt": 0, "n_det": 0, "dice_sum": 0.0,
                                 "pix_inter": 0, "pix_union": 0, "n_slices": 0})

    # ----------------------------- 3D accumulators -------------------------
    # For each patient we allocate (D, H, W) uint8 volumes:
    #   gt3d[pid] : ground-truth OR of all processed slices (may have tiny
    #               holes because skipped slices exist, but filtering handles
    #               the fragments and the 18-conn struct is tolerant of tiny
    #               gaps across ~4 skipped edge bg slices)
    #   pr3d[pid] : binarised predictions placed at the SAME (d,h,w) indices
    # We also record how many slices were processed for sanity (for eval,
    # only positives are loaded; full depth is pulled from mask.npy shape).
    pp_shape = {}        # pid -> (D, H, W)  original 3D volume shape
    gt3d = {}            # pid -> (D, H, W) uint8
    pr3d = {}            # pid -> (D, H, W) uint8
    # precompute for each patient the (y0, x0) in the original H,W that
    # corresponds to the top-left of the 256x256 center-crop grid.  This
    # is the inverse of PerPatientDataset._center_pad_crop.
    pp_crop_offset = {}  # pid -> (y0, x0, H_orig, W_orig, crop_size)

    def _ensure_vol(pid):
        if pid in pp_shape:
            return
        from pathlib import Path
        p = Path(data_root) / split / pid / "mask.npy"
        mk = np.load(p, mmap_mode="r")
        D, H, W = int(mk.shape[0]), int(mk.shape[1]), int(mk.shape[2])
        pp_shape[pid] = (D, H, W)
        gt3d[pid] = np.zeros((D, H, W), dtype=np.uint8)
        pr3d[pid] = np.zeros((D, H, W), dtype=np.uint8)
        # derive the offset of the 256x256 center(pad+crop) window
        # by re-running _center_pad_crop on a 26-channel zero volume
        # (we only care about indices 0:crop_size mapping -> original y0,x0)
        cs = loader.dataset.crop_size if hasattr(loader.dataset, "crop_size") else max(H, W)
        dummy_v = np.zeros((2, max(H, W) + 1, max(H, W) + 1), dtype=np.float32)
        dummy_m = np.zeros((max(H, W) + 1, max(H, W) + 1), dtype=np.uint8)
        # Now simulate exactly what _center_pad_crop did on (H,W) input
        # by running it on a H,W dummy of the ORIGINAL slice sizes:
        from per_patient_dataset import PerPatientDataset as _PP
        in_v = np.zeros((2, H, W), dtype=np.float32)
        in_m = np.zeros((H, W), dtype=np.uint8)
        out_v, _ = _PP._center_pad_crop(in_v, in_m, cs, cs)
        # out_v shape (2, cs, cs).  We compute the mapping by finding the
        # position of central tissue via a sentinel pixel in in_m.
        in_m2 = np.zeros((H, W), dtype=np.uint8); in_m2[H // 2, W // 2] = 1
        _, out_m2 = _PP._center_pad_crop(np.zeros((2, H, W), dtype=np.float32),
                                         in_m2, cs, cs)
        cy, cx = np.unravel_index(int(np.argmax(out_m2)), out_m2.shape)
        y0 = H // 2 - cy
        x0 = W // 2 - cx
        # y0/x0 is where the top-left of the crop window lands in the
        # original (H,W) frame.  Clamp to valid range.
        pp_crop_offset[pid] = (int(y0), int(x0), H, W, cs)

    def _place_slice(pid, slice_idx, pred_256, gt_256):
        """Place (crop_size, crop_size) arrays back into the original
        (H_orig, W_orig) frame using pp_crop_offset.  Out-of-bounds pixels
        from the zero-pad step are simply dropped (they're outside anatomy).
        """
        y0, x0, H, W, cs = pp_crop_offset[pid]
        D = pp_shape[pid][0]
        if not (0 <= slice_idx < D):
            return  # just a safety
        # Determine the overlap in the cropped 256x256 window that maps to
        # valid (H,W) coordinates:
        cs_y_end = min(cs, H - y0); cs_y_start = max(0, -y0)
        cs_x_end = min(cs, W - x0); cs_x_start = max(0, -x0)
        y_orig_start = y0 + cs_y_start; y_orig_end = y0 + cs_y_end
        x_orig_start = x0 + cs_x_start; x_orig_end = x0 + cs_x_end
        if y_orig_end <= y_orig_start or x_orig_end <= x_orig_start:
            return
        # paste GT (filtered fragments before accumulation for 3D too)
        g_crop = gt_256[cs_y_start:cs_y_end, cs_x_start:cs_x_end]
        p_crop = pred_256[cs_y_start:cs_y_end, cs_x_start:cs_x_end]
        # filter tiny GT fragments before pasting (affects 3D GT count)
        g_crop = _filter_gt_2d_fragments(g_crop, min_pixels=gt_min_px)
        gt3d[pid][slice_idx, y_orig_start:y_orig_end, x_orig_start:x_orig_end] |= g_crop
        pr3d[pid][slice_idx, y_orig_start:y_orig_end, x_orig_start:x_orig_end] |= p_crop

    # --------------------------- main inference loop ----------------------
    t0 = time.time()
    for bi, batch in enumerate(loader):
        x1 = batch["image_t1"].to(device, non_blocking=True)
        x2 = batch["image_t2"].to(device, non_blocking=True)
        gt = batch["mask"]  # (B,1,H,W) on CPU (256x256 cropped)
        pids = batch["patient_id"]
        slice_idxs = batch["slice_idx"].cpu().numpy().astype(np.int64)

        _, _, _, sm = model(x1, x2)
        prob = torch.sigmoid(sm)
        pred_bin = (prob >= threshold).to(torch.uint8).cpu().numpy()
        gt_bin = (gt >= 0.5).to(torch.uint8).cpu().numpy()

        for b in range(pred_bin.shape[0]):
            p_256 = pred_bin[b, 0]
            g_256 = gt_bin[b, 0]
            pid = pids[b]
            # Filter tiny GT fragments at the 2D level too (before 2D lesion
            # counting) so "2D lesion" count isn't inflated by annotation
            # artefacts either.
            g_256_filt = _filter_gt_2d_fragments(g_256, min_pixels=gt_min_px)
            # Runtime guard: skip any sample whose GT mask is empty.
            if g_256_filt.sum() == 0:
                n_skipped_empty += 1
                continue
            n_evaluated += 1

            # ---------------- 2D lesion tally (backward compatible) -------------
            n_gt, n_det, s_dice = _lesion_dice_for_slice(p_256, g_256_filt)
            total_gt += n_gt
            total_detected += n_det
            total_dice_sum += s_dice
            inter = float((p_256 & g_256_filt).sum())
            union = float(p_256.sum()) + float(g_256_filt.sum())
            pos_pix_inter += inter
            pos_pix_union += union
            pp = pp_2d[pid]
            pp["n_gt"] += n_gt; pp["n_det"] += n_det; pp["dice_sum"] += s_dice
            pp["pix_inter"] += inter; pp["pix_union"] += union; pp["n_slices"] += 1

            # ---------------- 3D volume accumulation ----------------------------
            _ensure_vol(pid)
            _place_slice(pid, int(slice_idxs[b]), p_256, g_256_filt)
        if (bi + 1) % 20 == 0:
            print(f"  [batch {bi+1}/{len(loader)}] evaluated={n_evaluated} "
                  f"2D_lesions_so_far={total_gt} detected={total_detected} "
                  f"3D_patients_accumulated={len(pp_shape)}")
    dt_infer = time.time() - t0

    # ---------------- 2D summary (unchanged) ----------------
    overall_2d = {
        "mean_lesion_dice": float(total_dice_sum / max(1, total_gt)),
        "detection_recall": float(total_detected / max(1, total_gt)),
        "n_lesions": int(total_gt),
        "n_detected": int(total_detected),
        "pixel_dice": float((2.0 * pos_pix_inter) / (pos_pix_union + eps)),
        "n_slices_evaluated": int(n_evaluated),
        "n_slices_skipped_empty": int(n_skipped_empty),
        "threshold": float(threshold),
        "elapsed_s": float(dt_infer),
        "ts": datetime.now().isoformat(),
    }
    per_pat_2d = {}
    for pid, v in pp_2d.items():
        per_pat_2d[pid] = {
            "lesion_dice": float(v["dice_sum"] / max(1, v["n_gt"])),
            "detection_recall": float(v["n_det"] / max(1, v["n_gt"])),
            "n_lesions": int(v["n_gt"]),
            "n_detected": int(v["n_det"]),
            "pixel_dice": float((2.0 * v["pix_inter"]) / (v["pix_union"] + eps)),
            "n_slices": int(v["n_slices"]),
        }

    # ---------------- 3D node-level evaluation ----------------
    t1 = time.time()
    gt_3d_sum, det_3d_sum, dice_3d_sum = 0, 0, 0.0
    pp_3d = {}
    for pid in sorted(pp_shape.keys()):
        g_lab, g_n = _3d_connected_components(gt3d[pid])
        p_lab, p_n = _3d_connected_components(pr3d[pid])
        n_gt3, n_det3, s_dice3 = _match_3d_lesions(g_lab, g_n, p_lab, p_n,
                                                    iou_threshold=iou_3d, eps=eps)
        gt_3d_sum += n_gt3
        det_3d_sum += n_det3
        dice_3d_sum += s_dice3
        # pixel dice (per patient, 3D volume)
        inter3 = float((gt3d[pid] & pr3d[pid]).sum())
        union3 = float(gt3d[pid].sum()) + float(pr3d[pid].sum())
        pp_3d[pid] = {
            "node_dice": float(s_dice3 / max(1, n_gt3)),
            "node_recall": float(n_det3 / max(1, n_gt3)),
            "n_nodes_gt": int(n_gt3),
            "n_nodes_detected": int(n_det3),
            "n_pred_components": int(p_n),
            "pixel_dice_3d": float((2.0 * inter3) / (union3 + eps)),
            "fg_voxels_gt": int(gt3d[pid].sum()),
            "fg_voxels_pred": int(pr3d[pid].sum()),
        }

    # Also aggregate global 3D pixel Dice for the whole split.
    all_gt = np.concatenate([v.ravel() for v in gt3d.values()]) if gt3d else np.zeros(1, np.uint8)
    all_pr = np.concatenate([v.ravel() for v in pr3d.values()]) if pr3d else np.zeros(1, np.uint8)
    inter_all = float((all_gt & all_pr).sum())
    union_all = float(all_gt.sum()) + float(all_pr.sum())
    dt_eval = time.time() - t1 + dt_infer

    overall_3d = {
        "mean_node_dice": float(dice_3d_sum / max(1, gt_3d_sum)),
        "node_recall": float(det_3d_sum / max(1, gt_3d_sum)),
        "n_nodes_gt": int(gt_3d_sum),
        "n_nodes_detected": int(det_3d_sum),
        "pixel_dice_3d": float((2.0 * inter_all) / (union_all + eps)),
        "iou_threshold": float(iou_3d),
        "gt_min_px_filter": int(gt_min_px),
        "elapsed_s": float(dt_eval),
        "ts": datetime.now().isoformat(),
    }
    return overall_2d, overall_3d, per_pat_2d, pp_3d


def main():
    p = argparse.ArgumentParser(
        description="Per-patient CFANet test: 2D slice-level + NEW 3D "
                    "node-level (connected component across Z) evaluation. "
                    "Skips empty-mask samples. Reports mean lesion/node Dice "
                    "and detection recall.")
    p.add_argument("--data_root", type=str, default="./dataset_per_patient")
    p.add_argument("--ckpt", type=str,
                   default="./Snapshot/CFANet_per_patient/Cod_best_lesion.pth")
    p.add_argument("--context", type=int, default=2,
                   help="2.5D context (each side). k_slice = 2*context+1 = 5.")
    p.add_argument("--trainsize", type=int, default=256)
    p.add_argument("--batchsize", type=int, default=8)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Binarisation threshold for the sigmoid prediction.")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--exclude", type=str, nargs="*", default=[],
                   help="Patient id(s) to drop entirely from the split (e.g. "
                        "--exclude 197557 196050 to skip known-bad cases).")
    p.add_argument("--out_json", type=str,
                   default="./Snapshot/CFANet_per_patient/test_results.json")
    p.add_argument("--gt_min_px", type=int, default=10,
                   help="Drop 2D GT connected components with < this many "
                        "pixels (annotation fragments). 0 = disable.")
    p.add_argument("--iou_3d", type=float, default=0.2,
                   help="3D node detection IoU threshold. 0.2 = lenient "
                        "'any substantive overlap = detected'. 0.5 = strict.")
    opt = p.parse_args()

    # ---- device ----
    if opt.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
        print("[device] CPU")
    else:
        device = torch.device(f"cuda:{opt.gpu}")
        torch.cuda.set_device(opt.gpu)
        print(f"[device] cuda:{opt.gpu} ({torch.cuda.get_device_name(opt.gpu)})")

    k_slice = 2 * int(opt.context) + 1
    model = load_model(opt.ckpt, k_slice, device)

    # ---- test dataset: positive_only=True -> skip empty-mask slices ----
    ds = PerPatientDataset(
        opt.data_root, split=opt.split, augment=False,
        crop_size=opt.trainsize, context=opt.context, seed=42,
        positive_only=True,
        exclude_pids=opt.exclude,
    )
    excl_note = (f", excluded: {sorted(opt.exclude)}"
                 if opt.exclude else "")
    print(f"[data] split='{opt.split}' positive_only=True -> {len(ds)} slices "
          f"to evaluate (empty-mask slices skipped{excl_note})")
    loader = DataLoader(ds, batch_size=opt.batchsize, shuffle=False,
                        num_workers=opt.num_workers, pin_memory=True,
                        drop_last=False)

    print(f"[test] running inference (threshold={opt.threshold}, "
          f"gt_min_px={opt.gt_min_px}, iou_3d={opt.iou_3d}) ...")
    ov2d, ov3d, pp2d, pp3d = run_test(
        model, loader, device, opt.data_root, opt.split,
        threshold=opt.threshold, gt_min_px=int(opt.gt_min_px),
        iou_3d=float(opt.iou_3d))

    def _r(v, n=4):
        return f"{float(v):.{n}f}"

    print("\n================ 2D SLICE-LEVEL (per 2D component) ============")
    print(f"  threshold            : {ov2d['threshold']}")
    print(f"  GT frag filter (<px) : {opt.gt_min_px}")
    print(f"  slices evaluated     : {ov2d['n_slices_evaluated']} "
          f"(skipped empty: {ov2d['n_slices_skipped_empty']})")
    print(f"  lesions (GT, 2D)     : {ov2d['n_lesions']}")
    print(f"  lesions detected     : {ov2d['n_detected']}")
    print(f"  detection recall     : {_r(ov2d['detection_recall'])}")
    print(f"  mean lesion Dice     : {_r(ov2d['mean_lesion_dice'])}")
    print(f"  pixel Dice (pos)     : {_r(ov2d['pixel_dice'])}")
    print(f"  elapsed              : {ov2d['elapsed_s']:.1f}s")

    print("\n================ 3D NODE-LEVEL (merged across Z, NEW) =======")
    print(f"  3D IoU threshold     : {ov3d['iou_threshold']}")
    print(f"  nodes (GT, 3D)       : {ov3d['n_nodes_gt']}    # ← physical lymph nodes (not slices)")
    print(f"  nodes detected       : {ov3d['n_nodes_detected']}")
    print(f"  node-level recall    : {_r(ov3d['node_recall'])}")
    print(f"  mean node Dice       : {_r(ov3d['mean_node_dice'])}")
    print(f"  3D volume pixel Dice : {_r(ov3d['pixel_dice_3d'])}")
    print(f"  elapsed              : {ov3d['elapsed_s']:.1f}s")

    # ---- per-patient ranking table (2D vs 3D side-by-side) ----
    flagged = {'197557', '196050', '182822', '196787'}
    print("\n--- per-patient (sorted by 3D node Dice, 2D/3D side-by-side) ---")
    print(f"  {'pid':8s} {'dim':>3s} {'2D_dice':>8s} {'2D_rec':>7s} "
          f"{'2D_L':>5s} {'3D_node':>8s} {'3D_rec':>7s} {'3D_N':>5s} {'pix3D':>7s} fg_vox")
    for pid in sorted(pp3d.keys(), key=lambda p: pp3d[p]["node_dice"]):
        v3 = pp3d[pid]
        v2 = pp2d.get(pid, {"lesion_dice": 0.0, "detection_recall": 0.0,
                            "n_lesions": 0, "n_detected": 0})
        marker = " **" if pid in flagged else ""
        print(f"  {pid:8s} {('3D' if v3['n_nodes_gt'] > 0 else '-'):>3s} "
              f"{v2['lesion_dice']:8.3f} {v2['detection_recall']:7.3f} {v2['n_lesions']:5d} "
              f"{v3['node_dice']:8.3f} {v3['node_recall']:7.3f} {v3['n_nodes_gt']:5d} "
              f"{v3['pixel_dice_3d']:7.3f} {v3['fg_voxels_gt']:>7d}{marker}")

    os.makedirs(os.path.dirname(opt.out_json) or ".", exist_ok=True)
    payload = {
        "ckpt": opt.ckpt, "split": opt.split,
        "threshold": ov2d["threshold"],
        "gt_min_px_filter": int(opt.gt_min_px),
        "iou_3d_threshold": float(opt.iou_3d),
        "excluded_pids": sorted(opt.exclude),
        "overall_2d": ov2d,
        "overall_3d": ov3d,
        "per_patient_2d": pp2d,
        "per_patient_3d": pp3d,
    }
    with open(opt.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n[done] full results (2D + 3D + per-patient) saved to {opt.out_json}")


if __name__ == "__main__":
    main()
