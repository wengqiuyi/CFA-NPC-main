import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import argparse

from lib.model import CFANet
from utils.trainer import adjust_lr, clip_gradient, eval_mae
# from utils.dataloader import get_loader,test_dataset   # deprecated: now use data.dataset
from datetime import datetime

import numpy as np
import logging

best_mae   = 1
best_epoch = 0
best_val_fg_dice = -1.0
best_val_fg_epoch = 0


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

def dice_loss(pred, mask, eps=1e-6):
    """
    Soft Dice loss — robust to class imbalance, ideal for small targets.
    """
    p = torch.sigmoid(pred)
    inter = (p * mask).sum(dim=(2, 3))
    union = (p + mask).sum(dim=(2, 3))
    dice  = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()


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


def small_target_loss(pred, mask,
                      w_ft=2.0, w_bce=0.15, w_dice=0.5,
                      ft_alpha=0.3, ft_beta=0.7, ft_gamma=1.0,
                      boundary_k=5):
    """
    Combined loss for small-object binary segmentation.

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
               pos_sample_weight=1.0):
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
        ds = PreprocessedDataset(root, split=split)
        sampler = None
        if split == 'train' and pos_sample_weight > 1.0:
            weights = ds.build_sample_weights(pos_sample_weight)
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

    # multi-scale controlled by CLI; default is single-scale for stability
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

            # ---- rescale ----
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

            # ---- forward ----
            sal_out1, sal_out2, sal_out3, mask = model(images_t1, images_t2)

            # small_target_loss is parameterised via CLI; pass through
            loss_sal1  = loss_func(sal_out1, gts)
            loss_sal2  = loss_func(sal_out2, gts)
            loss_sal3  = loss_func(sal_out3, gts)
            loss_mask  = loss_func(mask,    gts)

            w1, w2, w3, wm = deep_sup_w
            loss_total = w1*loss_sal1 + w2*loss_sal2 + w3*loss_sal3 + wm*loss_mask

            # ---- NaN / inf guard: skip the optimizer step if loss blew up ----
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # ---- warmup (linear over the first N global steps) ----
            # IMPORTANT: multiply the *original* learning rate by warm
            # each step.  Do NOT compound a `base_lr` across steps —
            # the previous version used `base_lr = pg['lr'] / warm`
            # and then `pg['lr'] = base_lr * warm`, which made the LR
            # *quadratically* ramp up to 200x the configured value
            # over 200 warmup steps.  That blew up to 1e-2 and
            # produced the NaN that ended epoch 6.
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


    if (epoch) % opt.save_epoch == 0:
        torch.save(model.state_dict(), os.path.join(opt.save_model, 'CODNet_%d.pth' % (epoch)))
        
        
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

            s1, s2, s3, sm = model(x1, x2)
            probs = (torch.sigmoid(s1) + torch.sigmoid(s2) +
                     torch.sigmoid(s3) + torch.sigmoid(sm)) / 4.0
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
    fg_mean = float(sum(fg) / max(1, len(fg)))
    all_mean = float(sum(all_d) / max(1, len(all_d)))
    return fg_mean, all_mean, int(len(fg)), int(len(all_d))


def test(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()

    with torch.no_grad():
        mae_sum = 0.0
        n = 0

        if hasattr(test_loader, 'size') and hasattr(test_loader, 'load_data'):
            for _ in range(test_loader.size):
                image, gt, _name = test_loader.load_data()
                gt = np.asarray(gt, np.float32)
                gt /= (gt.max() + 1e-8)

                image = image.cuda()
                _, _, _, res = model(image)
                res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                mae_sum += float(np.mean(np.abs(res - gt)))
                n += 1
        else:
            device = next(model.parameters()).device
            for batch in test_loader:
                x1 = batch['image_t1'].to(device, non_blocking=True)
                x2 = batch['image_t2'].to(device, non_blocking=True)
                gt = batch['mask'].to(device, non_blocking=True)

                s1, s2, s3, sm = model(x1, x2)
                probs = (torch.sigmoid(s1) + torch.sigmoid(s2) +
                         torch.sigmoid(s3) + torch.sigmoid(sm)) / 4.0

                if probs.shape[-2:] != gt.shape[-2:]:
                    probs = F.interpolate(probs, size=gt.shape[-2:], mode='bilinear', align_corners=False)

                mae_sum += float(torch.abs(probs - gt).mean().item()) * probs.shape[0]
                n += int(probs.shape[0])

        mae = mae_sum / max(1, n)

        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(save_path, 'Cod_best.pth'))
                print('best epoch:{}'.format(epoch))
                
       
        

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
    parser.add_argument('--w_ft',        type=float, default=2.0,
                        help='Weight of Focal-Tversky (small-target friendly).')
    parser.add_argument('--w_bce',       type=float, default=0.15,
                        help='Weight of boundary-aware BCE. Use smaller values when empty slices dominate.')
    parser.add_argument('--w_dice',      type=float, default=0.5,
                        help='Weight of Dice.')
    parser.add_argument('--ft_alpha',    type=float, default=0.3,
                        help='Focal-Tversky FP weight. Smaller values are more tolerant to false positives.')
    parser.add_argument('--ft_beta',     type=float, default=0.7,
                        help='Focal-Tversky FN weight. Larger values reduce missed tiny lesions.')
    parser.add_argument('--ft_gamma',    type=float, default=1.0,
                        help='Focusing exponent for Focal-Tversky. 1.0 is a good starting point for tiny targets.')
    parser.add_argument('--boundary_k',  type=int,   default=5,
                        help='Half-width of the boundary band for boundary-aware BCE. '
                             '5 keeps the boundary focus tighter for tiny targets.')
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


    opt = parser.parse_args()

    torch.cuda.set_device(opt.gpu)

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
    model = CFANet(channel=64, dual_backbone=True).cuda()
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
        ckpt = torch.load(opt.pretrain_ckpt, map_location='cuda:{}'.format(opt.gpu))
        ckpt = unwrap_state_dict(ckpt)

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



    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    # ------------------ Configurable loss (CLI-controlled weights) ---------- #
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
                              pos_sample_weight=opt.pos_sample_weight)
    val_loader   = get_loader(opt.val_root, split='val',
                              batchsize=opt.batchsize, trainsize=opt.trainsize,
                              num_workers=loader_workers, augment=False,
                              mask_combine=opt.mask_combine,
                              resplit=opt.resplit, seed=opt.resplit_seed,
                              train_ratio=opt.train_ratio,
                              val_ratio=opt.val_ratio,
                              test_ratio=opt.test_ratio,
                              data_format=opt.data_format,
                              pos_sample_weight=1.0)
    test_loader  = get_loader(opt.test_root, split='test',
                              batchsize=opt.batchsize, trainsize=opt.trainsize,
                              num_workers=loader_workers, augment=False,
                              mask_combine=opt.mask_combine,
                              resplit=opt.resplit, seed=opt.resplit_seed,
                              train_ratio=opt.train_ratio,
                              val_ratio=opt.val_ratio,
                              test_ratio=opt.test_ratio,
                              data_format=opt.data_format,
                              pos_sample_weight=1.0)

    total_step = len(train_loader)

    print('-' * 30, "\n[Training Dataset INFO]\nformat: {}\ntrain_root: {}\nval_root: {}\n"
                    "test_root: {}\nLearning Rate: {}\nBatch Size: {}\nTraining Save: {}\n"
                    "total_num: {}\n".format(opt.data_format, opt.train_root, opt.val_root,
                                             opt.test_root, opt.lr, opt.batchsize,
                                             opt.save_model, total_step), '-' * 30)

    for epoch_iter in range(1, opt.epoch + 1):

        adjust_lr(optimizer, epoch_iter, opt.decay_rate, opt.decay_epoch)

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
        
        
        
