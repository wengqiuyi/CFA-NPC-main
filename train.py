import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse

from lib.model import CFANet
from utils.trainer import adjust_lr, clip_gradient, eval_mae
# from utils.dataloader import get_loader,test_dataset   # deprecated: now use data.dataset
from datetime import datetime

import numpy as np
import logging

best_mae   = 1
best_epoch = 0


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
                      w_ft=1.0, w_bce=0.5, w_dice=0.5,
                      ft_alpha=0.7, ft_beta=0.3, ft_gamma=0.75,
                      boundary_k=15):
    """
    Combined loss for small-object binary segmentation.

        L = w_ft   * FocalTversky
          + w_bce  * BoundaryAwareBCE
          + w_dice * Dice

    Args:
        pred, mask : logits / binary mask  (B, 1, H, W)
        w_ft       : weight of Focal-Tversky (region-level, small-target)
        w_bce      : weight of boundary-aware BCE (edge focus)
        w_dice     : weight of Dice (overlap, imbalance-robust)
    """
    return (w_ft   * focal_tversky_loss(pred, mask, ft_alpha, ft_beta, ft_gamma)
          + w_bce  * boundary_aware_bce (pred, mask, boundary_k)
          + w_dice * dice_loss           (pred, mask))


# --------------------------------------------------------------------------- #
#   DataLoader wrapper
# --------------------------------------------------------------------------- #
def get_loader(root, split, batchsize, trainsize, num_workers=4,
               augment=True, shuffle=None,
               crop_size=0, mask_combine='or',
               resplit=True, seed=42,
               train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Build a torch.utils.data.DataLoader for the NIfTI medical-slice dataset.

    Every sample is a *paired* (image_t1, image_t2, mask) — the two
    modalities of the same patient's same slice.  The split is
    rebuilt deterministically from the paired (patient, idx_in_patient)
    groups, since the original manifest split is made independently
    per modality and would leave val/test with virtually no T1/T2
    pairs.
    """
    from data.dataset import MedicalSliceDataset
    if shuffle is None:
        shuffle = (split == 'train')
    ds = MedicalSliceDataset(root=root, split=split, trainsize=trainsize,
                             augment=augment,
                             crop_size=crop_size,
                             mask_combine=mask_combine,
                             resplit=resplit, seed=seed,
                             train_ratio=train_ratio,
                             val_ratio=val_ratio,
                             test_ratio=test_ratio)
    return DataLoader(ds, batch_size=batchsize, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, drop_last=(split == 'train'))


def train(train_loader, model, optimizer, epoch, opt, loss_func, total_step):
    """
    Training iteration
    :param train_loader:
    :param model:
    :param optimizer:
    :param epoch:
    :param opt:
    :param loss_func:
    :param total_step:
    :return:
    """
    model.train()


    size_rates = [0.75, 1, 1.25]


    for step, batch in enumerate(train_loader):

        # ---- unpack batch (paired T1+T2 only) ----
        images_t1 = batch['image_t1'].cuda()
        images_t2 = batch['image_t2'].cuda()
        gts       = batch['mask'].cuda()

        for rate in size_rates:

            optimizer.zero_grad()

            # ---- rescale ----
            trainsize = int(round(opt.trainsize*rate/32)*32)
            if rate != 1:
                images_t1 = F.interpolate(images_t1, size=(trainsize, trainsize),
                                          mode='bilinear', align_corners=True)
                images_t2 = F.interpolate(images_t2, size=(trainsize, trainsize),
                                          mode='bilinear', align_corners=True)
                gts       = F.interpolate(gts, size=(trainsize, trainsize),
                                          mode='bilinear', align_corners=True)

            # ---- forward ----
            sal_out1, sal_out2, sal_out3, mask = model(images_t1, images_t2)
            loss_sal1  = small_target_loss(sal_out1, gts)
            loss_sal2  = small_target_loss(sal_out2, gts)
            loss_sal3  = small_target_loss(sal_out3, gts)
            loss_mask  = small_target_loss(mask,    gts)

            loss_total = loss_sal1 + loss_sal2 + loss_sal3 + loss_mask

            loss_total.backward()
            optimizer.step()

        if step % 10 == 0 or step == total_step:
            print('[{}] => [Epoch Num: {:03d}/{:03d}] => [Global Step: {:04d}/{:04d}] => [Loss_sal1: {:.4f} Loss_sal2: {:.4f} Loss_sal3: {:.4f} Loss_mask: {:.4f} Loss_total: {:.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, step, total_step, loss_sal1.data, loss_sal2.data, loss_sal3.data, loss_mask.data, loss_total.data))

            logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss_sal1: {:.4f} Loss_sal2: {:.4f} Loss_sal3: {:.4f} Loss_mask: {:.4f} Loss_total: {:.4f}'.
                    format( epoch, opt.epoch, step, total_step, loss_sal1.data, loss_sal2.data, loss_sal3.data, loss_mask.data, loss_total.data))


    if (epoch) % opt.save_epoch == 0:
        torch.save(model.state_dict(), save_path + 'CODNet_%d.pth' % (epoch))
        
        
def test(test_loader,model,epoch,save_path):
    
    global best_mae,best_epoch
    model.eval()
    
    with torch.no_grad():
        mae_sum=0
        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            
            gt /= (gt.max() + 1e-8)
            
            image = image.cuda()

            _,_,_,res = model(image)
            res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum +=np.sum(np.abs(res-gt))*1.0/(gt.shape[0]*gt.shape[1])
            
        mae = mae_sum / test_loader.size
      
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch,mae,best_mae,best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae   = mae
                best_epoch = epoch
                
                torch.save(model.state_dict(), save_path+'/Cod_best.pth')
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
    
    
    parser.add_argument('--train_root',   type=str, default='./TrainDataset')
    parser.add_argument('--val_root',     type=str, default='./TrainDataset')
    parser.add_argument('--test_root',    type=str, default='./TrainDataset')

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
                        help='Re-split paired (T1,T2) groups deterministically '
                             '(80/10/10).  The original manifest split is per-modality '
                             'and would leave val/test with almost no T1/T2 pairs.')
    parser.add_argument('--no_resplit',   dest='resplit', action='store_false',
                        help='Use the original manifest split instead of re-splitting.')
    parser.add_argument('--resplit_seed', type=int, default=42,
                        help='Random seed for the deterministic re-split.')
    parser.add_argument('--train_ratio',  type=float, default=0.8)
    parser.add_argument('--val_ratio',    type=float, default=0.1)
    parser.add_argument('--test_ratio',   type=float, default=0.1)

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

        # If checkpoint is a dict with 'state_dict' or 'model' wrapper, unwrap it
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        elif isinstance(ckpt, dict) and 'model' in ckpt:
            ckpt = ckpt['model']

        model_dict = model.state_dict()

        if opt.load_backbone_only:
            # Only load keys whose name exists in the model AND has matching shape.
            # This works for: (a) Res2Net official weights, (b) previous CFANet ckpt
            backbone_keys = {k: v for k, v in ckpt.items()
                             if k in model_dict and v.shape == model_dict[k].shape
                             and (k.startswith('backbone_t1.') or k.startswith('backbone_t2.'))}
            model_dict.update(backbone_keys)
            missing, unexpected = model.load_state_dict(model_dict, strict=False)
            print('  loaded {} backbone tensors'.format(len(backbone_keys)))
        else:
            # tolerant load: fill matching keys, ignore the rest
            if opt.strict_load:
                model.load_state_dict(ckpt, strict=True)
                print('  strict load OK')
            else:
                matched = {k: v for k, v in ckpt.items()
                           if k in model_dict and v.shape == model_dict[k].shape}
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
    LogitsBCE = torch.nn.BCEWithLogitsLoss()

    # ------------------ DataLoaders (NIfTI medical slices) ------------------ #
    train_loader = get_loader(opt.train_root, split='train',
                              batchsize=opt.batchsize, trainsize=opt.trainsize,
                              num_workers=4, augment=True,
                              crop_size=opt.crop_size,
                              mask_combine=opt.mask_combine,
                              resplit=opt.resplit, seed=opt.resplit_seed,
                              train_ratio=opt.train_ratio,
                              val_ratio=opt.val_ratio,
                              test_ratio=opt.test_ratio)
    val_loader   = get_loader(opt.val_root, split='val',
                              batchsize=opt.batchsize, trainsize=opt.trainsize,
                              num_workers=2, augment=False,
                              mask_combine=opt.mask_combine,
                              resplit=opt.resplit, seed=opt.resplit_seed,
                              train_ratio=opt.train_ratio,
                              val_ratio=opt.val_ratio,
                              test_ratio=opt.test_ratio)
    test_loader  = get_loader(opt.test_root, split='test',
                              batchsize=opt.batchsize, trainsize=opt.trainsize,
                              num_workers=2, augment=False,
                              mask_combine=opt.mask_combine,
                              resplit=opt.resplit, seed=opt.resplit_seed,
                              train_ratio=opt.train_ratio,
                              val_ratio=opt.val_ratio,
                              test_ratio=opt.test_ratio)

    total_step = len(train_loader)

    print('-' * 30, "\n[Training Dataset INFO]\nroot: {}\nLearning Rate: {}\nBatch Size: {}\n"
                    "Training Save: {}\ntotal_num: {}\n".format(opt.train_root, opt.lr,
                                                              opt.batchsize, opt.save_model, total_step), '-' * 30)

    for epoch_iter in range(1, opt.epoch):
        
        adjust_lr(optimizer, epoch_iter, opt.decay_rate, opt.decay_epoch)
        
        train(train_loader, model, optimizer, epoch_iter,opt, LogitsBCE, total_step)
        #test(test_loader,   model, epoch_iter, opt.save_model)
        
        
        
