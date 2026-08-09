#!/usr/bin/env python3
"""
train_per_patient.py
====================

Fresh training entry point for the per-patient preprocessed dataset that
implements Step 7 of the spec:

  * Loss      : Dice (0.5) + Focal (0.3) + Boundary (0.2)   [combined_loss]
  * Optimizer : AdamW + CosineAnnealingLR over the full run
  * Evaluation: Dice per LESION (connected-component instance level),
                not per-pixel.  Reports mean lesion Dice + detection recall.

Consumes the per-patient layout produced by
``scripts/preprocess_per_patient.py`` and the online 2.5D + ROI-crop +
augmentation provided by ``per_patient_dataset.PerPatientDataset``.

Run OUTSIDE the sandbox on a GPU machine:
    python train_per_patient.py \
        --data_root ./dataset_per_patient \
        --epoch 100 --batchsize 8 --lr 1e-4 --trainsize 256 --gpu 0
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from lib.model import CFANet
# Reuse the spec-matched loss functions + checkpoint helpers from train.py.
from train import (
    combined_loss,
    unwrap_state_dict,
    looks_like_single_res2net_backbone,
    map_single_backbone_to_dual_backbone,
)
from per_patient_dataset import PerPatientDataset


# ---------------------------------------------------------------------------
# Lesion-level evaluation (Step 7: "Dice per lesion")
# ---------------------------------------------------------------------------
def _lesion_dice_for_slice(pred_bin: np.ndarray, gt_bin: np.ndarray,
                           iou_thresh: float = 0.5,
                           eps: float = 1e-6):
    """Per-lesion Dice for a single 2-D slice via connected components.

    A "lesion" = one connected component (8-connectivity) in the GT mask.
    For each GT lesion we find the prediction component with the largest
    pixel overlap; if that overlap >= ``iou_thresh * gt_lesion_pixels``
    the lesion is considered *detected* and its Dice is
    ``2*overlap / (gt_pixels + pred_component_pixels)``.  Missed lesions
    contribute Dice = 0.

    Returns
    -------
    (n_gt, n_detected, sum_dice)
        n_gt         : number of GT lesions in this slice (0 if empty)
        n_detected   : how many of them were detected
        sum_dice     : sum of per-lesion Dice (divide by n_gt for the mean)
    """
    from scipy.ndimage import label

    if gt_bin.sum() == 0:
        return 0, 0, 0.0

    gt_lab, n_gt = label(gt_bin > 0, structure=np.ones((3, 3), dtype=np.uint8))
    if n_gt == 0:
        return 0, 0, 0.0
    pr_lab, n_pr = label(pred_bin > 0, structure=np.ones((3, 3), dtype=np.uint8))

    # Build overlap matrix [gt_idx, pr_idx] -> shared pixel count.
    if n_pr == 0:
        # every GT lesion missed
        return int(n_gt), 0, 0.0

    overlap = np.zeros((n_gt + 1, n_pr + 1), dtype=np.int64)
    # np.add.at accumulates counts for each (gt_id, pr_id) pair.
    np.add.at(overlap, (gt_lab.ravel(), pr_lab.ravel()), 1)
    overlap = overlap[1:, 1:]  # drop background row/col

    n_detected = 0
    sum_dice = 0.0
    for g in range(n_gt):
        gt_pixels = int((gt_lab == (g + 1)).sum())
        best_pr = int(np.argmax(overlap[g]))
        inter = int(overlap[g, best_pr])
        if inter >= iou_thresh * gt_pixels:
            pr_pixels = int((pr_lab == (best_pr + 1)).sum())
            dice = (2.0 * inter) / (gt_pixels + pr_pixels + eps)
            n_detected += 1
            sum_dice += float(dice)
        # else: missed -> dice contribution 0
    return int(n_gt), int(n_detected), float(sum_dice)


def eval_lesion_dice(val_loader, model, threshold=0.5, device="cuda"):
    """Run lesion-level Dice over a validation loader.

    Returns dict with mean_lesion_dice, detection_recall, n_lesions,
    plus a per-slice pixel-Dice on positive slices for backward comparability.
    """
    model.eval()
    total_gt = 0
    total_detected = 0
    total_dice_sum = 0.0
    pos_slice_dice_sum = 0.0
    pos_slice_n = 0
    with torch.no_grad():
        for batch in val_loader:
            x1 = batch["image_t1"].to(device, non_blocking=True)
            x2 = batch["image_t2"].to(device, non_blocking=True)
            gt = batch["mask"].to(device, non_blocking=True)  # (B,1,H,W)

            _, _, _, sm = model(x1, x2)
            prob = torch.sigmoid(sm)
            pred_bin = (prob >= threshold).to(torch.uint8).cpu().numpy()
            gt_bin = (gt >= 0.5).to(torch.uint8).cpu().numpy()

            for b in range(pred_bin.shape[0]):
                p = pred_bin[b, 0]
                g = gt_bin[b, 0]
                n_gt, n_det, s_dice = _lesion_dice_for_slice(p, g)
                total_gt += n_gt
                total_detected += n_det
                total_dice_sum += s_dice
                # pixel-Dice on positive slices (for logging continuity)
                if g.sum() > 0:
                    inter = float((p & g).sum())
                    pdice = (2.0 * inter) / (float(p.sum()) + float(g.sum()) + 1e-6)
                    pos_slice_dice_sum += pdice
                    pos_slice_n += 1
    model.train()
    mean_lesion = total_dice_sum / max(1, total_gt)
    recall = total_detected / max(1, total_gt)
    pos_pix = pos_slice_dice_sum / max(1, pos_slice_n)
    return {
        "mean_lesion_dice": float(mean_lesion),
        "detection_recall": float(recall),
        "n_lesions": int(total_gt),
        "n_detected": int(total_detected),
        "pos_pixel_dice": float(pos_pix),
        "n_pos_slices": int(pos_slice_n),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_one_epoch(train_loader, model, optimizer, scheduler, loss_func,
                    epoch, opt, device, deep_sup_w, grad_clip, log_every=10):
    model.train()
    size_rates = [float(r) for r in opt.size_rates.split(",")] if opt.size_rates else [1.0]
    total_step = len(train_loader)
    running = 0.0
    running_n = 0
    for step, batch in enumerate(train_loader):
        images_t1 = batch["image_t1"].to(device, non_blocking=True)
        images_t2 = batch["image_t2"].to(device, non_blocking=True)
        gts = batch["mask"].to(device, non_blocking=True)

        t1_0, t2_0, gts_0 = images_t1, images_t2, gts
        for rate in size_rates:
            optimizer.zero_grad()
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if abs(rate - 1.0) > 1e-6:
                images_t1 = F.interpolate(t1_0, size=(trainsize, trainsize),
                                          mode="bilinear", align_corners=False)
                images_t2 = F.interpolate(t2_0, size=(trainsize, trainsize),
                                          mode="bilinear", align_corners=False)
                gts = F.interpolate(gts_0, size=(trainsize, trainsize), mode="nearest")
            else:
                images_t1, images_t2, gts = t1_0, t2_0, gts_0

            sal_out1, sal_out2, sal_out3, mask = model(images_t1, images_t2)
            loss_sal1 = loss_func(sal_out1, gts)
            loss_sal2 = loss_func(sal_out2, gts)
            loss_sal3 = loss_func(sal_out3, gts)
            loss_mask = loss_func(mask, gts)
            w1, w2, w3, wm = deep_sup_w
            loss_total = w1 * loss_sal1 + w2 * loss_sal2 + w3 * loss_sal3 + wm * loss_mask

            if not torch.isfinite(loss_total):
                print(f"[{datetime.now()}] [WARN] non-finite loss at step {step}, skipping")
                if step % log_every == 0 or step == total_step:
                    print(f"[{datetime.now()}] Epoch {epoch:03d}/{opt.epoch} "
                          f"Step {step:04d}/{total_step} loss_total=NaN skip")
                continue

            loss_total.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            running += float(loss_total.detach())
            running_n += 1

        if step % log_every == 0 or step == total_step:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"[{datetime.now()}] Epoch {epoch:03d}/{opt.epoch} "
                  f"Step {step:04d}/{total_step} "
                  f"loss_sal1={float(loss_sal1):.4f} loss_sal2={float(loss_sal2):.4f} "
                  f"loss_sal3={float(loss_sal3):.4f} loss_mask={float(loss_mask):.4f} "
                  f"loss_total={float(loss_total):.4f} lr={lr_now:.2e}")
    scheduler.step()
    return running / max(1, running_n)


def build_loaders(opt):
    train_ds = PerPatientDataset(
        opt.data_root, split="train", augment=True,
        crop_size=opt.trainsize, context=opt.context, seed=opt.seed,
        roi_center_ratio=opt.roi_center_ratio,
    )
    val_ds = PerPatientDataset(
        opt.data_root, split="val", augment=False,
        crop_size=opt.trainsize, context=opt.context, seed=opt.seed,
    )
    test_ds = PerPatientDataset(
        opt.data_root, split="test", augment=False,
        crop_size=opt.trainsize, context=opt.context, seed=opt.seed,
    )

    sampler = None
    shuffle = True
    if opt.pos_sample_weight > 1.0:
        # Step 4 weights drive the sampler: positive slices (w=1.0),
        # edge (0.5), background (0.1).  Multiplying by pos_sample_weight
        # boosts positive sampling further on top of the per-slice weights.
        w = np.array(train_ds.sample_weights, dtype=np.float64)
        w = w * np.where([s[2] > 0.5 for s in train_ds.samples],
                         opt.pos_sample_weight, 1.0)
        # The per-sample weights already encode 1.0/0.5/0.1; the extra
        # pos_sample_weight multiplier reweights positives up further.
        # Normalize to a probability distribution for clean sampling.
        w = w / w.sum()
        sampler = WeightedRandomSampler(
            torch.tensor(w, dtype=torch.double),
            num_samples=len(w), replacement=True)
        shuffle = False

    train_loader = DataLoader(train_ds, batch_size=opt.batchsize, shuffle=shuffle,
                              sampler=sampler, num_workers=opt.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=opt.batchsize, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=opt.batchsize, shuffle=False,
                             num_workers=opt.num_workers, pin_memory=True, drop_last=False)
    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds


def main():
    p = argparse.ArgumentParser(description="Per-patient CFANet training (Step 7 spec).")
    p.add_argument("--data_root", type=str, default="./dataset_per_patient")
    p.add_argument("--epoch", type=int, default=100)
    p.add_argument("--start_epoch", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batchsize", type=int, default=8)
    p.add_argument("--trainsize", type=int, default=256)
    p.add_argument("--context", type=int, default=2,
                   help="2.5D context (each side). k_slice = 2*context+1 = 5.")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--roi_center_ratio", type=float, default=0.7,
                   help="Step 5: probability of ROI-centred crop (70%).")
    p.add_argument("--pos_sample_weight", type=float, default=3.0,
                   help="Extra positive oversampling multiplier on top of Step 4 weights.")
    p.add_argument("--size_rates", type=str, default="1",
                   help="Multi-scale rates. Keep '1' for small targets.")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--deep_sup_w", type=str, default="0.5,0.75,0.75,1.0")
    p.add_argument("--boundary_k", type=int, default=5)
    p.add_argument("--fl_alpha", type=float, default=0.25)
    p.add_argument("--fl_gamma", type=float, default=2.0)
    p.add_argument("--val_eval_interval", type=int, default=5)
    p.add_argument("--save_epoch", type=int, default=10)
    p.add_argument("--save_model", type=str, default="./Snapshot/CFANet_per_patient")
    p.add_argument("--pretrain_ckpt", type=str, default="")
    p.add_argument("--load_backbone_only", action="store_true")
    p.add_argument("--strict_load", action="store_true")
    p.add_argument("--backbone_lr_mult", type=float, default=0.1)
    p.add_argument("--freeze_backbone_epochs", type=int, default=0)
    p.add_argument("--log_every", type=int, default=20)
    opt = p.parse_args()

    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    # ---- device ----
    if opt.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
        print("[device] CUDA not available or --cpu set -> running on CPU")
    else:
        device = torch.device(f"cuda:{opt.gpu}")
        torch.cuda.set_device(opt.gpu)
        print(f"[device] using cuda:{opt.gpu} ({torch.cuda.get_device_name(opt.gpu)})")

    os.makedirs(opt.save_model, exist_ok=True)

    # ---- model (k_slice = 2*context+1 = 5 for context=2) ----
    k_slice = 2 * int(opt.context) + 1
    model = CFANet(channel=64, dual_backbone=True, in_channels=k_slice).to(device)
    total = sum(p.nelement() for p in model.parameters())
    print(f"[model] CFANet dual_backbone in_channels={k_slice} params={total/1e6:.2f}M")

    # ---- optional pretrained weights ----
    if opt.pretrain_ckpt and os.path.isfile(opt.pretrain_ckpt):
        print(f"[pretrain] loading {opt.pretrain_ckpt}")
        ckpt = unwrap_state_dict(torch.load(opt.pretrain_ckpt, map_location=device))
        try:
            from lib.model import remap_ckpt_in_channels
            ckpt = remap_ckpt_in_channels(ckpt, in_channels_new=k_slice)
        except Exception as e:
            print(f"  [warn] remap_ckpt_in_channels failed: {e}")
        model_dict = model.state_dict()
        if looks_like_single_res2net_backbone(ckpt):
            mapped = map_single_backbone_to_dual_backbone(ckpt, model_dict)
            model_dict.update(mapped)
            m, u = model.load_state_dict(model_dict, strict=False)
            print(f"  single-backbone Res2Net -> dual, copied {len(mapped)} tensors, "
                  f"missing={len(m)} unexpected={len(u)}")
        elif opt.load_backbone_only:
            bb = {k: v for k, v in ckpt.items()
                  if k in model_dict and hasattr(v, "shape")
                  and v.shape == model_dict[k].shape
                  and (k.startswith("backbone_t1.") or k.startswith("backbone_t2."))}
            model_dict.update(bb)
            m, u = model.load_state_dict(model_dict, strict=False)
            print(f"  loaded {len(bb)} backbone tensors, missing={len(m)} unexpected={len(u)}")
        elif opt.strict_load:
            model.load_state_dict(ckpt, strict=True)
            print("  strict load OK")
        else:
            matched = {k: v for k, v in ckpt.items()
                       if k in model_dict and hasattr(v, "shape")
                       and v.shape == model_dict[k].shape}
            model_dict.update(matched)
            m, u = model.load_state_dict(model_dict, strict=False)
            print(f"  loaded {len(matched)}/{len(model_dict)} tensors, "
                  f"missing={len(m)} unexpected={len(u)}")
    elif opt.pretrain_ckpt:
        print(f"[warn] --pretrain_ckpt={opt.pretrain_ckpt} not found, training from scratch")
    else:
        print("[pretrain] none -> ImageNet Res2Net-50 backbone (auto-downloaded)")

    # ---- freeze backbone for first N epochs (optional) ----
    if opt.freeze_backbone_epochs and opt.freeze_backbone_epochs > 0:
        for n, par in model.named_parameters():
            if n.startswith("backbone_t1.") or n.startswith("backbone_t2."):
                par.requires_grad = False

    # ---- optimizer: AdamW with backbone LR mult ----
    bb_mult = float(opt.backbone_lr_mult)
    if abs(bb_mult - 1.0) < 1e-9:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        bb, other = [], []
        for n, par in model.named_parameters():
            if not par.requires_grad:
                continue
            (bb if n.startswith(("backbone_t1.", "backbone_t2.")) else other).append(par)
        optimizer = torch.optim.AdamW(
            [{"params": bb, "lr": opt.lr * bb_mult, "_bb_factor": bb_mult},
             {"params": other, "lr": opt.lr, "_bb_factor": 1.0}],
            weight_decay=opt.weight_decay)

    # ---- Step 7: CosineAnnealingLR over the full run ----
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(opt.epoch), eta_min=float(opt.min_lr))

    # ---- loss (Dice 0.5 + Focal 0.3 + Boundary 0.2) ----
    # combined_loss already implements 0.5*Dice + 0.3*Focal + 0.2*Boundary.
    # Override focal alpha/gamma and boundary k via a thin wrapper.
    import train as _train_mod
    _fl_alpha = float(opt.fl_alpha)
    _fl_gamma = float(opt.fl_gamma)
    _bk = int(opt.boundary_k)

    def loss_func(pred, mask):
        dice = _train_mod.dice_loss(pred, mask)
        focal = _train_mod.binary_focal_loss(pred, mask, alpha=_fl_alpha, gamma=_fl_gamma)
        bound = _train_mod.boundary_loss(pred, mask, k=_bk)
        return 0.5 * dice + 0.3 * focal + 0.2 * bound

    deep_sup_w = tuple(float(x) for x in opt.deep_sup_w.split(","))
    assert len(deep_sup_w) == 4

    # ---- data ----
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = build_loaders(opt)
    print(f"[data] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} slices")
    print(f"[data] train batches/epoch={len(train_loader)}")

    best_lesion_dice = -1.0
    best_epoch = -1
    log_path = os.path.join(opt.save_model, "train_log.jsonl")

    for epoch in range(int(opt.start_epoch), int(opt.epoch) + 1):
        # unfreeze backbone after warmup
        if opt.freeze_backbone_epochs and epoch == int(opt.freeze_backbone_epochs) + 1:
            for n, par in model.named_parameters():
                if n.startswith("backbone_t1.") or n.startswith("backbone_t2."):
                    par.requires_grad = True
            # re-add newly-trainable params to optimizer (rebuild param groups)
            print(f"[epoch {epoch}] unfreezing backbone")

        t0 = time.time()
        avg_loss = train_one_epoch(train_loader, model, optimizer, scheduler,
                                   loss_func, epoch, opt, device,
                                   deep_sup_w, opt.grad_clip, opt.log_every)
        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[epoch {epoch:03d}/{opt.epoch}] avg_loss={avg_loss:.4f} "
              f"lr={lr_now:.2e} time={dt:.1f}s")

        log_entry = {
            "epoch": epoch, "avg_loss": avg_loss, "lr": lr_now,
            "time_s": dt, "ts": datetime.now().isoformat(),
        }

        if opt.val_eval_interval and opt.val_eval_interval > 0 and \
                (epoch % opt.val_eval_interval == 0 or epoch == opt.epoch):
            metrics = eval_lesion_dice(val_loader, model,
                                       threshold=0.5, device=device)
            print(f"[epoch {epoch:03d}] VAL lesion_dice={metrics['mean_lesion_dice']:.4f} "
                  f"recall={metrics['detection_recall']:.4f} "
                  f"(n_lesions={metrics['n_lesions']} detected={metrics['n_detected']}) "
                  f"pos_pixel_dice={metrics['pos_pixel_dice']:.4f} "
                  f"(n_pos_slices={metrics['n_pos_slices']})")
            log_entry["val"] = metrics
            if metrics["n_lesions"] > 0 and metrics["mean_lesion_dice"] > best_lesion_dice:
                best_lesion_dice = metrics["mean_lesion_dice"]
                best_epoch = epoch
                torch.save(model.state_dict(),
                           os.path.join(opt.save_model, "Cod_best_lesion.pth"))
                print(f"[epoch {epoch:03d}] new best lesion_dice={best_lesion_dice:.4f} "
                      f"-> saved Cod_best_lesion.pth")

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        if opt.save_epoch and epoch % opt.save_epoch == 0:
            torch.save(model.state_dict(),
                       os.path.join(opt.save_model, f"CODNet_{epoch}.pth"))

    # ---- final test eval ----
    print("\n[final] evaluating on TEST split ...")
    test_metrics = eval_lesion_dice(test_loader, model, threshold=0.5, device=device)
    print(f"[test] lesion_dice={test_metrics['mean_lesion_dice']:.4f} "
          f"recall={test_metrics['detection_recall']:.4f} "
          f"(n_lesions={test_metrics['n_lesions']} detected={test_metrics['n_detected']}) "
          f"pos_pixel_dice={test_metrics['pos_pixel_dice']:.4f}")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"final_test": test_metrics,
                            "best_val_lesion_dice": best_lesion_dice,
                            "best_epoch": best_epoch}, ensure_ascii=False) + "\n")
    print(f"[done] best val lesion_dice={best_lesion_dice:.4f} @ epoch {best_epoch}")


if __name__ == "__main__":
    main()
