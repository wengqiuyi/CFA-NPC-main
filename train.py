import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import argparse

from lib.model import CFANet
from utils.trainer import adjust_lr, clip_gradient
# from utils.dataloader import get_loader,test_dataset   # deprecated: now use data.dataset
from datetime import datetime

import logging
best_val_fg_dice = -1.0
best_val_fg_epoch = 0


#经典的医学图像分割损失函数，让模型更关注边界区域和小目标，而不是被大面积背景主导。
def structure_loss(pred, mask):
    """
    Original structure loss (BCE + weighted IoU).
    Kept for backward compatibility.
    """
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()


# ===================================================================== #
#   Small-Target Oriented Loss  (polyp / COD / small-object segmentation)
# ===================================================================== #
#   Loss = alpha * FocalTversky   (region-level, handles tiny objects
#                                 and severe class imbalance)
#       + beta  * BoundaryAwareBCE (focuses on hard, blurred edges)
#       + gamma * Dice             (smooth region overlap)
# ===================================================================== #
#标准的软 Dice 损失，与前面那段 structure_loss是互补关系
def dice_loss(pred, mask, eps=1e-6):
    """
    Soft Dice loss — robust to class imbalance, ideal for small targets.
    """
    p = torch.sigmoid(pred)#激活函数
    inter = (p * mask).sum(dim=(2, 3))
    union = (p + mask).sum(dim=(2, 3))
    dice  = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()

# Focal Tversky Loss（焦点 Tversky 损失）​ ，门针对极度类别不平衡 + 小目标难分割的任务设计。
def focal_tversky_loss(pred, mask, alpha=0.7, beta=0.3, gamma=0.75, eps=1e-6):
    """
    Focal Tversky loss — adds an emphasis factor to Tversky index so
    that small / hard-to-segment regions are weighted more heavily.
    Recommended for highly imbalanced small-object tasks.

        TI  = TP / (TP + alpha*FP + beta*FN)
        FL  = (1 - TI) ** gamma

    alpha  = FP weight (decrease to penalize false positives)
    beta   = FN weight (increase to penalize missed small targets)
    gamma  = focusing exponent (>=1 sharpens focus on hard pixels)
    """
    p = torch.sigmoid(pred)
    tp = (p * mask).sum(dim=(2, 3))
    fp = (p * (1 - mask)).sum(dim=(2, 3))
    fn = ((1 - p) * mask).sum(dim=(2, 3))
    ti = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return ((1 - ti) ** gamma).mean()
#边界感知 BCE（Boundary-Aware Binary Cross Entropy），用形态学操作圈出 GT mask 的边界带，然后给边界像素更高的 BCE 权重，迫使模型把边缘学得更锐利。
def boundary_aware_bce(pred, mask, k=15):
    """
    BCE re-weighted by the boundary-band of the GT mask.
    Pixels within ``k`` pixels of a GT boundary are weighted higher
    to sharpen the edges of small objects.
    """
    # Distance-from-boundary: morphological erosion proxy via avg-pool
    kernel = 2 * k + 1
    mask_dil  = (F.avg_pool2d(mask, kernel_size=kernel, stride=1,
                              padding=k) > 0.5).float()
    mask_erod = (F.avg_pool2d(mask, kernel_size=kernel, stride=1,
                              padding=k) >= 1.0).float()
    boundary  = (mask_dil - mask_erod).clamp(min=0)              # 0/1 band

    # Weight:  3.0 on boundary pixels, 1.0 elsewhere
    weight    = 1.0 + 2.0 * boundary
    bce       = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    return (weight * bce).mean()

#复合损失函数设计，把上面三个损失函数揉在了一起
def small_target_loss(pred, mask,
                      w_ft=2.0, w_bce=0.15, w_dice=0.5,
                      ft_alpha=0.3, ft_beta=0.7, ft_gamma=1.0,
                      boundary_k=5):
    """
    Combined loss for small-object binary segmentation (legacy).

        L = w_ft   * FocalTversky
          + w_bce  * BoundaryAwareBCE
          + w_dice * Dice

    Args:
        pred, mask : logits / binary mask  (B, 1, H, W)
        w_ft       : weight of Focal-Tversky (region-level, small-target).
                     Default 2.0 — dominant term for extremely small targets.
        w_bce      : weight of boundary-aware BCE.  Default 0.15
                     so the many negative slices do not overpower the lesion signal.
        w_dice     : weight of Dice (overlap, imbalance-robust).
        boundary_k : half-width of the boundary band.  5 keeps the band tight
                     on tiny lesions in 256x256 inputs.
    """
    return (w_ft   * focal_tversky_loss(pred, mask, ft_alpha, ft_beta, ft_gamma)
          + w_bce  * boundary_aware_bce (pred, mask, boundary_k)
          + w_dice * dice_loss           (pred, mask))


# ===================================================================== #
#   用户显式推荐的组合损失：Dice (0.5) + Focal (0.3) + Boundary (0.2)       #
# ===================================================================== #
#   用户原话："小目标在交叉熵损失中贡献极小，必须使用组合损失。                #
#   ——  Dice 处理类别不平衡 /  Focal 压制易分背景 /  Boundary 抓边缘      #
# ===================================================================== #
def binary_focal_loss(pred, mask, alpha=0.25, gamma=2.0, eps=1e-6):
    """
    Binary Focal Loss (BCE-based, logits input).

        pt = p        when y==1
           = 1 - p    when y==0
        FL = - w_alpha * (1 - pt)^gamma * log(pt)

    - alpha=0.25: 给正像素更高基础权重（头颈部淋巴结正像素仅占 <0.1%）。
    - gamma=2.0 : 易分样本 pt≈1 时 (1-pt)^gamma ≈ 0，大面积背景像素的损失被压掉，
                  只剩下困难像素（靠近病灶边缘）驱动梯度。
    - 为什么小目标友好：
        普通 BCE 里 99.9% 的梯度来自背景，小目标的梯度完全被淹没；
        Focal 把背景损失压到可忽略的量级，小目标像素的梯度占据主导。
    """
    p = torch.sigmoid(pred).clamp(eps, 1 - eps)
    bce = -(mask * torch.log(p) + (1 - mask) * torch.log(1 - p))
    pt = p * mask + (1 - p) * (1 - mask)                 # 预测对的置信度 ∈[eps,1-eps]
    alpha_weight = alpha * mask + (1 - alpha) * (1 - mask)
    focal = alpha_weight * ((1 - pt) ** gamma) * bce
    return focal.mean()


def boundary_loss(pred, mask, k=5, eps=1e-6):
    """
    Boundary Loss —— boundary-band overlap (pred soft boundary ↔ GT hard boundary).

    流程（全部可微分，纯 torch）:
        (1) GT 的 k 像素边界带：hard dilate(>0.5) − hard erode(≥1.0)
        (2) Prediction 的 soft 边界带：对 sigmoid(p) 直接做 avg_pool soft-dilate / soft-erode
            → b_pred = soft_dil(p) − soft_ero(p)   （连续可微，梯度能完整回传到 logits）
        (3) 对每个样本算 b_pred / b_gt 的 Dice，只有存在 GT 边界带的样本才计入。

    为什么小目标必须有这一项：
        15×15 淋巴结 1 像素轮廓偏差 → 全局 Dice 只掉 ~3%，但边界带 Dice 掉 >20%；
        独立监督边界带才能把小目标的边缘压锐利。
    为什么对 pred 用 soft 版本:
        hard > 0.5 threshold 不可微分，pred 边界带用 soft-dilate/soft-erode 的差，
        保证 loss → logits 梯度链完整（如果用 hard，boundary 项几乎没有梯度，学不动）。
    """
    kernel = 2 * k + 1
    pad = k

    def _hard_band(x_hard):
        dil = (F.avg_pool2d(x_hard, kernel_size=kernel, stride=1, padding=pad) > 0.5).float()
        ero = (F.avg_pool2d(x_hard, kernel_size=kernel, stride=1, padding=pad) >= 1.0).float()
        return (dil - ero).clamp(min=0)

    def _soft_band(p_soft):
        # soft dilate: avg_pool + 只要邻域内存在高 p 就升高
        dil = F.avg_pool2d(p_soft, kernel_size=kernel, stride=1, padding=pad)
        # soft erode: 把 p 反过来（1-p）做 dilate，再反回来
        ero = 1.0 - F.avg_pool2d(1.0 - p_soft, kernel_size=kernel, stride=1, padding=pad)
        return (dil - ero).clamp(min=0, max=1)

    b_gt = _hard_band(mask)
    p_soft = torch.sigmoid(pred).clamp(eps, 1 - eps)
    b_pred = _soft_band(p_soft)

    has_gt = b_gt.sum(dim=(2, 3)) > 0                        # (B,1) bool
    safe = has_gt.float()
    inter = (b_pred * b_gt).sum(dim=(2, 3))
    union = (b_pred + b_gt).sum(dim=(2, 3))
    bdice_per = (2.0 * inter + eps) / (union + eps)
    if safe.sum() == 0:
        return (bdice_per * 0.0).mean()                      # 可微分 0
    return 1.0 - (bdice_per * safe).sum() / safe.sum()


def combined_loss(pred, target):
    """
    用户指定的小目标专用组合损失：Dice (0.5) + Focal (0.3) + Boundary (0.2)。

        L = 0.5 * Dice
          + 0.3 * Focal
          + 0.2 * Boundary

    - Dice    ：全局区域优化 + 天然抗类别不平衡。
    - Focal   ：99% 背景像素的 BCE 损失被 (1-pt)^gamma 压到接近 0，
                只剩下小目标 / 困难边缘像素主导梯度。
    - Boundary：独立监督 k=5 的边界带 Dice，逼 15×15 小目标边缘锐利。
    """
    dice  = dice_loss(pred, target)
    focal = binary_focal_loss(pred, target)
    bound = boundary_loss(pred, target)
    return 0.5 * dice + 0.3 * focal + 0.2 * bound

#非常实用的 checkpoint 解包工具函数，用于解决不同框架/不同训练脚本保存 checkpoint 的格式不统一
def unwrap_state_dict(ckpt):
    """Unwrap common checkpoint wrappers and return a plain state_dict."""
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        return ckpt['state_dict']
    if isinstance(ckpt, dict) and 'model' in ckpt:
        return ckpt['model']
    return ckpt


def looks_like_single_res2net_backbone(state_dict):
    """
    Official Res2Net backbone weights use keys like:
        conv1.0.weight, bn1.weight, layer1.0.conv1.weight
    rather than backbone_t1./backbone_t2. prefixes.
    """
    if not isinstance(state_dict, dict) or not state_dict:
        return False
    sample_keys = list(state_dict.keys())[:20]
    return (not any(k.startswith('backbone_t1.') or k.startswith('backbone_t2.')
                    for k in sample_keys)
            and any(k.startswith('conv1.') or k.startswith('bn1.') or k.startswith('layer1.')
                    for k in sample_keys))

#判断一个state_dict是不是官方 Res2Net 单骨干权重
def map_single_backbone_to_dual_backbone(state_dict, model_dict):
    """
    Copy one official Res2Net backbone state_dict into both encoder branches:
        conv1.0.weight -> backbone_t1.conv1.0.weight / backbone_t2.conv1.0.weight
    """
    mapped = {}
    for prefix in ('backbone_t1.', 'backbone_t2.'):
        for k, v in state_dict.items():
            mk = prefix + k
            if mk in model_dict and hasattr(v, 'shape') and v.shape == model_dict[mk].shape:
                mapped[mk] = v
    return mapped


# --------------------------------------------------------------------------- #
#   DataLoader wrapper
# --------------------------------------------------------------------------- #
def get_loader(root, split, batchsize, trainsize, num_workers=4,
               augment=True, shuffle=None,
               crop_size=0, mask_combine='or',
               resplit=True, seed=42,
               train_ratio=0.8, val_ratio=0.2, test_ratio=0.0,
               data_format='npy',
               pos_sample_weight=1.0,
               k_slice=3,
               use_roi_crop=False):
    """
    Build a torch.utils.data.DataLoader.

    Supported formats
    -----------------
    npy   : preprocessed slices saved under TrainDataset/TestDataset
    nifti : online 3-D NIfTI -> 2-D slicing pipeline from data/dataset.py
    """
    if shuffle is None:
        shuffle = (split == 'train')

    if data_format == 'npy':
        from preprocessed_dataset import PreprocessedDataset
        ds = PreprocessedDataset(root, split=split, augment=augment, seed=seed,
                                 crop_size=crop_size, k_slice=k_slice,
                                 use_roi_crop=use_roi_crop)
        sampler = None
        if split == 'train' and pos_sample_weight > 1.0:
            # 策略 B (Oversampling tiny targets):
            # build_sample_weights 默认 mode='area_inverse'，
            # 给 <400px（~20x20）的极小淋巴结额外 ×3 权重 + 面积反平
            # 方根加权，避免大病灶垄断梯度、小病灶永远学不到。
            # 如果想退回到旧的二元加权，请把 mode 改成 'binary'。
            weights = ds.build_sample_weights(
                pos_sample_weight,
                mode='area_inverse',
                tiny_threshold=400,
                tiny_weight_boost=3.0,
            )
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            shuffle = False
        return DataLoader(ds, batch_size=batchsize, shuffle=shuffle,
                          sampler=sampler, num_workers=num_workers, pin_memory=True,
                          drop_last=(split == 'train'))

    if data_format != 'nifti':
        raise ValueError(f'Unknown --data_format: {data_format}')

    from data.dataset import MedicalSliceDataset
    # resplit only makes sense on the training root; the test root is
    # the held-out set so we want every patient yielded.
    use_resplit = resplit and (split != 'test')
    ds = MedicalSliceDataset(root=root, split=split, trainsize=trainsize,
                             augment=augment,
                             crop_size=crop_size,
                             mask_combine=mask_combine,
                             resplit=use_resplit, seed=seed,
                             train_ratio=train_ratio,
                             val_ratio=val_ratio,
                             test_ratio=test_ratio)
    sampler = None
    #正样本过采样，解决类别不平衡问题
    if split == 'train' and pos_sample_weight > 1.0 and hasattr(ds, 'samples'):
        weights = [float(pos_sample_weight) if bool(s[5]) else 1.0 for s in ds.samples]
        sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double),
                                        num_samples=len(weights), replacement=True)
        shuffle = False
    return DataLoader(ds, batch_size=batchsize, shuffle=shuffle,
                      sampler=sampler, num_workers=num_workers,
                      pin_memory=True, drop_last=(split == 'train'))


def train(train_loader, model, optimizer, epoch, opt, loss_func, total_step,
          deep_sup_w=(1.0, 1.0, 1.0, 1.0), grad_clip=1.0, log_every=10):
    """
    Training iteration

    Stability tweaks (added to fix high per-batch loss variance):
    - Multi-scale is OFF by default (use --size_rates 1 1 1 to keep it on).
      Small targets (0.4% positive pixels) suffer at scale 1.25 because
      the lesion shrinks to a handful of pixels.
    - Grad-norm clipping (default 1.0) prevents the Adam optimizer from
      being kicked by an occasional pathological batch.
    - Deep-supervision weights (1.0, 1.0, 1.0, 1.0) so the three
      intermediate heads all contribute (was implicit before but the
      same mask was used as ground truth for all 4 outputs).
    """
    model.train()

    # 多尺度控制，小目标（0.4% 正像素）在 scale=1.25时会被缩到只剩几个像素，直接消失。多尺度对大目标有用，对小目标反而有害
    size_rates = [float(r) for r in opt.size_rates.split(',')] if opt.size_rates else [1.0]
    epoch_lrs = [pg['lr'] for pg in optimizer.param_groups]
    device = next(model.parameters()).device

    for step, batch in enumerate(train_loader):

        # ---- unpack batch (paired T1+T2 only) ----
        images_t1 = batch['image_t1'].to(device, non_blocking=True)
        images_t2 = batch['image_t2'].to(device, non_blocking=True)
        gts       = batch['mask'].to(device, non_blocking=True)

        images_t1_0 = images_t1
        images_t2_0 = images_t2
        gts_0 = gts

        for rate in size_rates:

            optimizer.zero_grad()

            #  尺寸对齐（32 的倍数）
            trainsize = int(round(opt.trainsize*rate/32)*32)
            if rate != 1:
                images_t1 = F.interpolate(images_t1_0, size=(trainsize, trainsize),
                                          mode='bilinear', align_corners=False)
                images_t2 = F.interpolate(images_t2_0, size=(trainsize, trainsize),
                                          mode='bilinear', align_corners=False)
                gts       = F.interpolate(gts_0, size=(trainsize, trainsize),
                                          mode='nearest')
            else:
                images_t1 = images_t1_0
                images_t2 = images_t2_0
                gts = gts_0

            # 前向传播
            sal_out1, sal_out2, sal_out3, mask = model(images_t1, images_t2)

            #  深监督损失， 四个输出共用同一个 GT
            loss_sal1  = loss_func(sal_out1, gts)
            loss_sal2  = loss_func(sal_out2, gts)
            loss_sal3  = loss_func(sal_out3, gts)
            loss_mask  = loss_func(mask,    gts)

            w1, w2, w3, wm = deep_sup_w
            loss_total = w1*loss_sal1 + w2*loss_sal2 + w3*loss_sal3 + wm*loss_mask

            # NaN/Inf 守卫。这是防止训练崩溃的最后一道防线。没有它，一个 NaN batch 就能毁掉整个 epoch。
            if not torch.isfinite(loss_total):
                print('[{}] => [WARN] non-finite loss at step {}, skipping update '
                      '(sal1={:.3f} sal2={:.3f} sal3={:.3f} mask={:.3f})'.format(
                          datetime.now(), step, float(loss_sal1), float(loss_sal2),
                          float(loss_sal3), float(loss_mask)))
                optimizer.zero_grad()
                # don't apply gradients; just continue
                if step % log_every == 0 or step == total_step:
                    print('[{}] => [Epoch Num: {:03d}/{:03d}] => [Global Step: {:04d}/{:04d}] => [Loss_sal1: {:.4f} Loss_sal2: {:.4f} Loss_sal3: {:.4f} Loss_mask: {:.4f} Loss_total: {:.4f}]'.
                          format(datetime.now(), epoch, opt.epoch, step, total_step,
                                 loss_sal1.data, loss_sal2.data, loss_sal3.data,
                                 loss_mask.data, loss_total.data))
                continue

            loss_total.backward()

            # ---- gradient clipping (stability) ----
            if grad_clip and grad_clip > 0:
                #梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            if opt.warmup_steps > 0:
                warm = min(1.0, (step + 1) / max(1, opt.warmup_steps))
                for i, pg in enumerate(optimizer.param_groups):
                    pg['lr'] = epoch_lrs[i] * warm

            optimizer.step()

        if step % log_every == 0 or step == total_step:
            print('[{}] => [Epoch Num: {:03d}/{:03d}] => [Global Step: {:04d}/{:04d}] => [Loss_sal1: {:.4f} Loss_sal2: {:.4f} Loss_sal3: {:.4f} Loss_mask: {:.4f} Loss_total: {:.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, step, total_step, loss_sal1.data, loss_sal2.data, loss_sal3.data, loss_mask.data, loss_total.data))

            logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss_sal1: {:.4f} Loss_sal2: {:.4f} Loss_sal3: {:.4f} Loss_mask: {:.4f} Loss_total: {:.4f}'.
                    format( epoch, opt.epoch, step, total_step, loss_sal1.data, loss_sal2.data, loss_sal3.data, loss_mask.data, loss_total.data))

    #模型保存逻辑
    if (epoch) % opt.save_epoch == 0:
        torch.save(model.state_dict(), os.path.join(opt.save_model, 'CODNet_%d.pth' % (epoch)))
        
#针对小目标设计的Dice 评估函数，把"有病灶的切片"和"全部切片"分开统计
def eval_val_foreground_dice(val_loader, model, threshold=0.5, eps=1e-7):
    model.eval()
    device = next(model.parameters()).device
    fg = []
    all_d = []
    with torch.no_grad():
        for batch in val_loader:
            x1 = batch['image_t1'].to(device, non_blocking=True)
            x2 = batch['image_t2'].to(device, non_blocking=True)
            gt = batch['mask'].to(device, non_blocking=True)

            _, _, _, sm = model(x1, x2)
            probs = torch.sigmoid(sm)
            pred = (probs >= threshold).to(gt.dtype)
            gt_b = (gt >= 0.5).to(gt.dtype)

            pred_f = pred.flatten(1)
            gt_f = gt_b.flatten(1)
            tp = (pred_f * gt_f).sum(dim=1)
            fp = (pred_f * (1 - gt_f)).sum(dim=1)
            fn = ((1 - pred_f) * gt_f).sum(dim=1)
            dice = (2 * tp) / (2 * tp + fp + fn + eps)

            all_d.extend(dice.detach().cpu().tolist())
            has_fg = (gt_f.sum(dim=1) > 0)
            if has_fg.any():
                fg.extend(dice[has_fg].detach().cpu().tolist())

    model.train()
    fg_mean = float(sum(fg) / max(1, len(fg)))#	只含淋巴结的切片的平均 Dice
    all_mean = float(sum(all_d) / max(1, len(all_d)))
    return fg_mean, all_mean, int(len(fg)), int(len(all_d))
                
       
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch',       type=int,   default=200,  help='epoch number, default=30')
    parser.add_argument('--lr',          type=float, default=1e-4, help='init learning rate, try `lr=1e-4`')
    parser.add_argument('--batchsize',   type=int,   default=10,   help='training batch size (Note: ~500MB per img in GPU)')
    parser.add_argument('--trainsize',   type=int,   default=352,  help='the size of training image, try small resolutions for speed (like 256)')
    parser.add_argument('--clip',        type=float, default=0.5,  help='gradient clipping margin')
    parser.add_argument('--decay_rate',  type=float, default=0.1,  help='decay rate of learning rate per decay step')
    parser.add_argument('--decay_epoch', type=int,   default=30,   help='every N epochs decay lr')
    parser.add_argument('--gpu',         type=int,   default=0,    help='choose which gpu you use')
    parser.add_argument('--save_epoch',  type=int,   default=5,    help='every N epochs save your trained snapshot')
    parser.add_argument('--save_model',  type=str,   default='./Snapshot/CFANet/')
    parser.add_argument('--val_eval_interval', type=int, default=5,
                        help='Evaluate on val every N epochs. 0 disables.')
    parser.add_argument('--val_threshold', type=float, default=0.5,
                        help='Binarization threshold for val evaluation.')
    
    
    parser.add_argument('--data_format',  type=str, default='npy',
                        choices=['npy', 'nifti'],
                        help='Training data source. "npy" reads preprocessed '
                             'TrainDataset/TestDataset slices, "nifti" uses the '
                             'online MedicalSliceDataset pipeline.')
    parser.add_argument('--train_root',   type=str, default='./TrainDataset',
                        help='Path to the training data root. For --data_format=npy, '
                             'this should contain ./train and ./val subfolders.')
    parser.add_argument('--val_root',     type=str, default='./TrainDataset',
                        help='Path to the validation data root. For --data_format=npy, '
                             'reuse ./TrainDataset so split="val" loads val/.')
    parser.add_argument('--test_root',    type=str, default='./TestDataset',
                        help='Path to the test data root. For --data_format=npy, '
                             'this should contain ./test subfolder.')

    # ---- DataLoader / augmentation options for the slice-level pipeline ----
    # NOTE: every slice of every split is yielded (no empty-mask filtering).
    parser.add_argument('--crop_size',    type=int,   default=0,
                        help='Random-crop size at native resolution BEFORE '
                             'resize to --trainsize.  0 disables.  Try '
                             '384 for small-target MRI segmentation.')
    parser.add_argument('--mask_combine', type=str,   default='or',
                        choices=['or', 'and', 't1', 't2'],
                        help="How to combine T1 and T2 masks for the same slice. "
                             "'or'=union (default), 'and'=intersection, "
                             "'t1'/'t2'=use that mask alone.")
    parser.add_argument('--resplit',      action='store_true', default=True,
                        help='Re-split paired (T1,T2) patients deterministically '
                             '(default 80/20).  Automatically disabled for the '
                             'test root.')
    parser.add_argument('--no_resplit',   dest='resplit', action='store_false',
                        help='Use the original manifest split instead of re-splitting.')
    parser.add_argument('--resplit_seed', type=int, default=42,
                        help='Random seed for the deterministic re-split.')
    parser.add_argument('--train_ratio',  type=float, default=0.8)
    parser.add_argument('--val_ratio',    type=float, default=0.2)
    parser.add_argument('--test_ratio',   type=float, default=0.0,
                        help='Fraction of traindata patients used for in-domain test. '
                             'The held-out test set comes from --test_root.')

    # ---- Loss / stability knobs (used to fix per-batch loss variance) ----
    parser.add_argument('--k_slice',       type=int,   default=3,
                        help='Number of adjacent 2-D slices stacked together '
                             'as pseudo-3D input.  MUST be an odd positive '
                             'integer:  3 = original 2.5D (centre ± 1 slice); '
                             '9 / 15 = wider z-window to model inter-slice '
                             'continuity.  The first Res2Net convolution is '
                             'linearly interpolated from the 3-channel ImageNet '
                             'checkpoint so k_slice > 3 still starts from a '
                             'meaningful visual prior.  '
                             'Input tensors shape per modality: (N, k_slice, H, W).')
    parser.add_argument('--loss_fn',       type=str,   default='combined',
                        choices=['combined', 'small_target_legacy'],
                        help='损失函数选择：\n'
                             '  - combined (默认，用户指定) : 0.5*Dice + 0.3*Focal + 0.2*Boundary\n'
                             '  - small_target_legacy       : 2.0*FocalTversky + 0.15*BoundaryBCE + 0.5*Dice')
    parser.add_argument('--fl_alpha',      type=float, default=0.25,
                        help='Focal Loss alpha 正/负像素基础权重。淋巴结 <0.1% 像素时 0.25 是合理起点。')
    parser.add_argument('--fl_gamma',      type=float, default=2.0,
                        help='Focal Loss focusing exponent. 越大越只关注困难像素。')
    parser.add_argument('--boundary_k',    type=int,   default=5,
                        help='Boundary Loss (combined) 的边界带宽 half-width。'
                             'k=5 → band width=11 px。越大越宽的边缘监督。')
    parser.add_argument('--w_ft',        type=float, default=2.0,
                        help='[legacy only] Weight of Focal-Tversky (small-target friendly).')
    parser.add_argument('--w_bce',       type=float, default=0.15,
                        help='[legacy only] Weight of boundary-aware BCE. Use smaller values when empty slices dominate.')
    parser.add_argument('--w_dice',      type=float, default=0.5,
                        help='[legacy only] Weight of Dice.')
    parser.add_argument('--ft_alpha',    type=float, default=0.3,
                        help='[legacy only] Focal-Tversky FP weight. Smaller values are more tolerant to false positives.')
    parser.add_argument('--ft_beta',     type=float, default=0.7,
                        help='[legacy only] Focal-Tversky FN weight. Larger values reduce missed tiny lesions.')
    parser.add_argument('--ft_gamma',    type=float, default=1.0,
                        help='[legacy only] Focusing exponent for Focal-Tversky. 1.0 is a good starting point for tiny targets.')
    parser.add_argument('--size_rates',  type=str,   default='1',
                        help='Comma-separated multi-scale rates (e.g. "0.75,1,1.25"). '
                             'Default single-scale "1" to keep loss stable on tiny targets.')
    parser.add_argument('--grad_clip',   type=float, default=1.0,
                        help='Max gradient norm.  0 disables clipping.')
    parser.add_argument('--warmup_steps', type=int,  default=200,
                        help='Linear LR warmup over the first N global steps.  0 disables.')
    parser.add_argument('--deep_sup_w',  type=str,   default='0.5,0.75,0.75,1.0',
                        help='Deep-supervision weights for (sal1, sal2, sal3, mask). '
                             'Comma-separated, e.g. "0.5,0.5,0.5,1.0".')
    parser.add_argument('--pos_sample_weight', type=float, default=3.0,
                        help='Oversampling weight for positive slices in the training loader. '
                             '1.0 disables oversampling.')
    parser.add_argument('--log_every',   type=int,   default=10,
                        help='Print the per-step loss every N global steps. '
                             'Use 1 for maximum verbosity (e.g. when debugging crashes).')

    # ---- Pretrained-weight options ----
    parser.add_argument('--pretrain_ckpt', type=str, default='',
                        help='Path to a pretrained CFANet .pth (transfer learning). '
                             'Leave empty to train from scratch (only ImageNet Res2Net weights will be used).')
    parser.add_argument('--load_backbone_only', action='store_true',
                        help='When --pretrain_ckpt is given, only load the backbone weights '
                             'and skip mismatching decoder / EFC layers.')
    parser.add_argument('--strict_load', action='store_true',
                        help='Require a strict state_dict match (default: tolerant).')
    parser.add_argument('--backbone_lr_mult', type=float, default=0.1,
                        help='Backbone LR multiplier. 0.1 is a common default when using ImageNet pretrained backbones.')
    parser.add_argument('--freeze_backbone_epochs', type=int, default=0,
                        help='Freeze backbone parameters for the first N epochs, then unfreeze.')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU training even if CUDA is available (sandbox / debug use).')
    # ---- Compatibility aliases (silently remapped below so the CLI is friendly)
    parser.add_argument('--root',  type=str, default=None,
                        help='[Alias] When set, overrides --train_root/--val_root to this '
                             'directory AND sets --test_root to "<root>_strat" style naming. '
                             'Convenience for paired stratify-split datasets.')
    parser.add_argument('--snapshot', type=str, default=None,
                        help='[Alias] Same as --save_model: directory to save CODNet_*.pth and Cod_best_fg.pth.')
    parser.add_argument('--nEpochs', type=int, default=None, dest='nEpochs_alias',
                        metavar='N',
                        help='[Alias] Same as --epoch: total epochs to train.  Takes precedence over --epoch when set.')
    parser.add_argument('--start_epoch', type=int, default=1,
                        help='Start epoch number used for LR scheduling / resume printouts. '
                             'Default 1 (from scratch).  Typical warm-start usage: 101 when continuing from epoch-100 ckpt.')
    parser.add_argument('--min_lr', type=float, default=None,
                        help='If set, enable a cosine annealing schedule over each --decay_epoch window '
                             '(wrapped onto adjust_lr so baseline StepLR decay still applies). '
                             'Leave None for the original plain StepLR schedule.')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='AdamW weight decay.  Default 0.0 keeps the original pure Adam behaviour. '
                             'Typical good default for medical imaging is 1e-4.')
    parser.add_argument('--use_roi_crop', action='store_true',
                        help='Use ROI-aware random-crop centred on foreground when available '
                             '(requires PreprocessedDataset ROI support, which is the default for npy format).')

    # ---- User-specified combined-loss weights (override combined_loss defaults)
    parser.add_argument('--dice_weight',  type=float, default=None,
                        help='Override soft-Dice weight in combined loss. Default 0.5 (per user request).')
    parser.add_argument('--focal_weight', type=float, default=None,
                        help='Override Binary Focal weight in combined loss. Default 0.3 (per user request).')
    parser.add_argument('--boundary_weight', type=float, default=None,
                        help='Override Boundary (band-Dice) weight in combined loss. Default 0.2 (per user request).')


    opt = parser.parse_args()

    # =====================================================================
    #   Alias post-processing (fix legacy CLI flags users already typed)
    # =====================================================================
    if opt.root is not None:
        # Common pattern: user points --root at TrainDataset_strat so the
        # paired test set lives in the same-named TestDataset_strat sibling.
        opt.train_root = opt.root
        opt.val_root   = opt.root
        if opt.test_root == './TestDataset' or opt.test_root == '':
            import re
            base = re.sub(r'TrainDataset', 'TestDataset', opt.root, count=1, flags=re.IGNORECASE)
            if os.path.isdir(base):
                opt.test_root = base
    if opt.snapshot is not None:
        opt.save_model = opt.snapshot
    if opt.nEpochs_alias is not None:
        opt.epoch = opt.nEpochs_alias

    # Sanity: start_epoch / epoch
    if opt.start_epoch < 1:
        opt.start_epoch = 1
    if opt.epoch < opt.start_epoch:
        opt.epoch = opt.start_epoch

    # Combined-loss weight overrides → build a closure with explicit scalars
    _dw = 0.5 if opt.dice_weight     is None else float(opt.dice_weight)
    _fw = 0.3 if opt.focal_weight    is None else float(opt.focal_weight)
    _bw = 0.2 if opt.boundary_weight is None else float(opt.boundary_weight)
    opt._combined_weights = (_dw, _fw, _bw)

    # Sandbox-friendly device selection: fail softly if CUDA isn't available,
    # fall back to CPU (with a clear print) unless --cpu is set.
    if opt.cpu or not torch.cuda.is_available():
        device = torch.device('cpu')
        opt.gpu_used = None
        print('[device] CUDA not available or --cpu set → running on CPU')
    else:
        device = torch.device(f'cuda:{opt.gpu}')
        opt.gpu_used = opt.gpu
        torch.cuda.set_device(opt.gpu)
        print(f'[device] using cuda:{opt.gpu}')

    save_path = opt.save_model
    os.makedirs(save_path, exist_ok=True)


    logging.basicConfig(filename=opt.save_model+'/log.log',format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level = logging.INFO,filemode='a',datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("COD-Train")
    logging.info("Config")
    logging.info('epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};save_path:{};decay_epoch:{}'.format(opt.epoch,opt.lr,opt.batchsize,opt.trainsize,opt.clip,opt.decay_rate,opt.save_model,opt.decay_epoch))



    # TIPS: you also can use deeper network for better performance like channel=64
    # Paired T1/T2 dataset → must use dual_backbone=True so the two
    # encoders are separate and FFTCMA / GlobalFusion can actually fuse
    # the two modalities (otherwise both inputs are the same image).
    #
    # --k_slice controls pseudo-3D input width per modality:
    #   3  = legacy 2.5D (centre ± 1 slice)
    #   9  = centre ± 4 slices — smooth slice continuity in receptive field
    #   15 = centre ± 7 slices — full 3D context over the typical neck volume
    # The encoder's first conv weight is linearly interpolated from the 3ch
    # ImageNet init when k_slice > 3, so training starts from a meaningful
    # visual representation (not random).
    model = CFANet(channel=64, dual_backbone=True, in_channels=opt.k_slice).to(device)
    #print('-' * 30, model, '-' * 30)

    # =================================================================
    #   Load pretrained weights (Res2Net backbone is already loaded by
    #   Res2Net_model(50) from the URL in res2net_v1b_base.py when the
    #   weight file is not present locally).
    #
    #   Recommended pretrained sources:
    #     (a) ImageNet-pretrained Res2Net-50 (auto-downloaded by
    #         res2net_v1b_base.py if ./lib/res2net50_v1b_26w_4s-3cf99910.pth
    #         is missing). Backbone-only, no decoder prior.
    #     (b) CFANet original polyp weights from the paper
    #         Google Drive: 1pgvgYebjVVm-QZN-VbGdtYmAyccQmKxZ
    #         -> save as ./checkpoint/CFANet.pth
    #         then pass --pretrain_ckpt ./checkpoint/CFANet.pth
    #     (c) Your previous training snapshot
    #         -> pass --pretrain_ckpt ./Snapshot/CFANet/CODNet_30.pth
    #     (d) ResNet-50 weights (resnet_50.pth / resnet_50_23dataset.pth)
    #         -> partial transfer: conv1+bn1 can be loaded directly.
    #            Use --load_backbone_only --allow_resnet_to_res2net
    #            (the inner Bottle2neck 3x3 hierarchical convs differ
    #             from ResNet Bottleneck, so only the 1x1 reductions and
    #             the input conv1 can be transferred reliably).
    # =================================================================
    if opt.pretrain_ckpt and os.path.isfile(opt.pretrain_ckpt):
        print('Loading pretrained weights from {}'.format(opt.pretrain_ckpt))
        ckpt = torch.load(opt.pretrain_ckpt, map_location=device)
        ckpt = unwrap_state_dict(ckpt)

        # --k_slice transfer: if this checkpoint was trained with a different
        # k_slice (legacy default = 3), remap the backbone input convs so the
        # whole decoder / fuser still transfer 1:1.  Only conv1.0.weight
        # changes shape; all later layers are identical regardless of k_slice.
        try:
            from lib.model import remap_ckpt_in_channels
            ckpt = remap_ckpt_in_channels(ckpt, in_channels_new=opt.k_slice)
        except Exception as _e:
            print(f'  [warn] remap_ckpt_in_channels failed: {_e}. Using raw ckpt.')

        model_dict = model.state_dict()

        if looks_like_single_res2net_backbone(ckpt):
            backbone_keys = map_single_backbone_to_dual_backbone(ckpt, model_dict)
            model_dict.update(backbone_keys)
            missing, unexpected = model.load_state_dict(model_dict, strict=False)
            print('  detected official single-backbone Res2Net weights')
            print('  copied {} tensors into backbone_t1/backbone_t2, missing={}, unexpected={}'.format(
                len(backbone_keys), len(missing), len(unexpected)))
        elif opt.load_backbone_only:
            # Load only backbone tensors from a dual-backbone / full-model checkpoint.
            backbone_keys = {k: v for k, v in ckpt.items()
                             if k in model_dict and hasattr(v, 'shape')
                             and v.shape == model_dict[k].shape
                             and (k.startswith('backbone_t1.') or k.startswith('backbone_t2.'))}
            model_dict.update(backbone_keys)
            missing, unexpected = model.load_state_dict(model_dict, strict=False)
            print('  loaded {} backbone tensors from checkpoint, missing={}, unexpected={}'.format(
                len(backbone_keys), len(missing), len(unexpected)))
        else:
            # tolerant load: fill matching keys, ignore the rest
            if opt.strict_load:
                model.load_state_dict(ckpt, strict=True)
                print('  strict load OK')
            else:
                matched = {k: v for k, v in ckpt.items()
                           if k in model_dict and hasattr(v, 'shape') and v.shape == model_dict[k].shape}
                model_dict.update(matched)
                missing, unexpected = model.load_state_dict(model_dict, strict=False)
                print('  loaded {}/{} tensors, missing={}, unexpected={}'.format(
                    len(matched), len(model_dict), len(missing), len(unexpected)))
    else:
        if opt.pretrain_ckpt:
            print('[Warn] --pretrain_ckpt={} not found, training from scratch'.format(opt.pretrain_ckpt))
        else:
            print('No pretrained CFANet weights provided. Backbone uses ImageNet Res2Net-50 (auto-downloaded).')

    total = sum([param.nelement() for param in model.parameters()])
    print('Number of parameter:%.2fM' % (total/1e6))



    if opt.freeze_backbone_epochs and opt.freeze_backbone_epochs > 0:
        for n, p in model.named_parameters():
            if n.startswith('backbone_t1.') or n.startswith('backbone_t2.'):
                p.requires_grad = False

    bb_mult = float(opt.backbone_lr_mult)
    if bb_mult <= 0:
        raise ValueError('--backbone_lr_mult must be > 0')
    if abs(bb_mult - 1.0) < 1e-9:
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            opt.lr, weight_decay=float(opt.weight_decay),
        )
    else:
        backbone_params = []
        other_params = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if n.startswith('backbone_t1.') or n.startswith('backbone_t2.'):
                backbone_params.append(p)
            else:
                other_params.append(p)
        optimizer = torch.optim.AdamW(
            [
                {'params': backbone_params, 'lr': opt.lr * bb_mult,
                 '_bb_factor': float(bb_mult)},
                {'params': other_params,    'lr': opt.lr,
                 '_bb_factor': 1.0},
            ],
            weight_decay=float(opt.weight_decay),
        )

    # "训练前准备区"代码，负责把损失函数、深监督权重、三个数据加载器全部装配好
    if opt.loss_fn == 'combined':
        _dw, _fw, _bw = opt._combined_weights
        # Close over the (possibly CLI-overridden) Dice/Focal/Boundary weights
        # so the user's explicit 0.5 / 0.3 / 0.2 request is honoured.
        def make_loss(p, m, _dw=_dw, _fw=_fw, _bw=_bw):
            dice  = dice_loss(p, m)
            focal = binary_focal_loss(p, m, alpha=opt.fl_alpha, gamma=opt.fl_gamma)
            bound = boundary_loss(p, m, k=opt.boundary_k)
            return _dw * dice + _fw * focal + _bw * bound
    else:  # small_target_legacy
        def make_loss(p, m):
            return small_target_loss(
                p, m,
                w_ft=opt.w_ft, w_bce=opt.w_bce, w_dice=opt.w_dice,
                ft_alpha=opt.ft_alpha, ft_beta=opt.ft_beta,
                ft_gamma=opt.ft_gamma, boundary_k=opt.boundary_k,
            )
    loss_func = make_loss
    deep_sup_w = tuple(float(x) for x in opt.deep_sup_w.split(','))
    assert len(deep_sup_w) == 4, f'--deep_sup_w must have 4 values, got {opt.deep_sup_w}'

    # ------------------ DataLoaders ------------------ #
    # For preprocessed .npy slices we can safely use a few workers.
    # For online NIfTI loading, 0 workers remains the safest default.
    loader_workers = 2 if opt.data_format == 'npy' else 0
    train_loader = get_loader(opt.train_root, split='train',
                              batchsize=opt.batchsize, trainsize=opt.trainsize,
                              num_workers=loader_workers, augment=True,
                              crop_size=opt.crop_size,
                              mask_combine=opt.mask_combine,
                              resplit=opt.resplit, seed=opt.resplit_seed,
                              train_ratio=opt.train_ratio,
                              val_ratio=opt.val_ratio,
                              test_ratio=opt.test_ratio,
                              data_format=opt.data_format,
                              pos_sample_weight=opt.pos_sample_weight,
                              k_slice=opt.k_slice,
                              use_roi_crop=opt.use_roi_crop)
    val_loader   = get_loader(opt.val_root, split='val',
                              batchsize=opt.batchsize, trainsize=opt.trainsize,
                              num_workers=loader_workers, augment=False,
                              mask_combine=opt.mask_combine,
                              resplit=opt.resplit, seed=opt.resplit_seed,
                              train_ratio=opt.train_ratio,
                              val_ratio=opt.val_ratio,
                              test_ratio=opt.test_ratio,
                              data_format=opt.data_format,
                              pos_sample_weight=1.0,
                              k_slice=opt.k_slice,
                              use_roi_crop=False)
    test_loader  = get_loader(opt.test_root, split='test',
                              batchsize=opt.batchsize, trainsize=opt.trainsize,
                              num_workers=loader_workers, augment=False,
                              mask_combine=opt.mask_combine,
                              resplit=opt.resplit, seed=opt.resplit_seed,
                              train_ratio=opt.train_ratio,
                              val_ratio=opt.val_ratio,
                              test_ratio=opt.test_ratio,
                              data_format=opt.data_format,
                              pos_sample_weight=1.0,
                              k_slice=opt.k_slice,
                              use_roi_crop=False)

    total_step = len(train_loader)

    print('-' * 30, "\n[Training Dataset INFO]\nformat: {}\ntrain_root: {}\nval_root: {}\n"
                    "test_root: {}\nLearning Rate: {}\nBatch Size: {}\nTraining Save: {}\n"
                    "total_num: {}\n".format(opt.data_format, opt.train_root, opt.val_root,
                                             opt.test_root, opt.lr, opt.batchsize,
                                             opt.save_model, total_step), '-' * 30)

    for epoch_iter in range(int(opt.start_epoch), int(opt.epoch) + 1):
        if opt.freeze_backbone_epochs and epoch_iter == int(opt.freeze_backbone_epochs) + 1:
            for n, p in model.named_parameters():
                if n.startswith('backbone_t1.') or n.startswith('backbone_t2.'):
                    p.requires_grad = True

        adjust_lr(optimizer, epoch_iter, opt.decay_rate, opt.decay_epoch,
                  min_lr=opt.min_lr)

        train(train_loader, model, optimizer, epoch_iter, opt, loss_func, total_step,
              deep_sup_w=deep_sup_w, grad_clip=opt.grad_clip)
        if opt.val_eval_interval and opt.val_eval_interval > 0:
            if (epoch_iter % opt.val_eval_interval == 0) or (epoch_iter == opt.epoch):
                fg_mean, all_mean, fg_n, all_n = eval_val_foreground_dice(
                    val_loader, model, threshold=opt.val_threshold
                )
                print('Epoch: {} ValDice(fg): {:.4f} (n={}) ValDice(all): {:.4f} (n={})  bestValFgDice: {:.4f} bestEpoch: {}'.format(
                    epoch_iter, fg_mean, fg_n, all_mean, all_n, best_val_fg_dice, best_val_fg_epoch
                ))
                logging.info('#VAL#:Epoch {:03d}/{:03d}, fg_dice: {:.4f} (n={}), all_dice: {:.4f} (n={}), best_fg_dice: {:.4f} best_epoch: {}'.format(
                    epoch_iter, opt.epoch, fg_mean, fg_n, all_mean, all_n, best_val_fg_dice, best_val_fg_epoch
                ))
                if fg_n > 0 and fg_mean > best_val_fg_dice:
                    best_val_fg_dice = fg_mean
                    best_val_fg_epoch = epoch_iter
                    torch.save(model.state_dict(), os.path.join(opt.save_model, 'Cod_best_fg.pth'))
                    print('best val foreground dice epoch:{}'.format(epoch_iter))
                    logging.info('#VAL#:best foreground dice epoch: {}'.format(epoch_iter))
        #test(test_loader,   model, epoch_iter, opt.save_model)
        
        
        
