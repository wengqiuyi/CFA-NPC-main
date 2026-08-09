import torch
from torch.autograd import Variable
from datetime import datetime
import os
#from apex import amp
import torch.nn.functional as F


def eval_mae(y_pred, y):
    """
    evaluate MAE (for test or validation phase)
    :param y_pred:
    :param y:
    :return: Mean Absolute Error
    """
    return torch.abs(y_pred - y).mean()




def numpy2tensor(numpy):
    """
    convert numpy_array in cpu to tensor in gpu
    :param numpy:
    :return: torch.from_numpy(numpy).cuda()
    """
    return torch.from_numpy(numpy).cuda()


def clip_gradient(optimizer, grad_clip):
    """
    recalibrate the misdirection in the training
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, epoch, decay_rate=0.1, decay_epoch=30, min_lr=None):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        if 'initial_lr' not in param_group:
            param_group['initial_lr'] = param_group['lr']
        new_lr = param_group['initial_lr'] * decay
        # Optional: inside each decay_epoch window, add a cosine-annealing
        # shape down to min_lr (if provided).  This matches the common
        # "StepLR floor + warm cosine" recipe used by modern segmentation
        # papers, and is a no-op when min_lr is None.
        if min_lr is not None and float(min_lr) >= 0:
            local_idx = (epoch - 1) % int(decay_epoch)
            import math
            frac = 0.5 * (1.0 + math.cos(math.pi * local_idx / max(1, int(decay_epoch))))
            # frac ∈ [1, 0] across the window, so lr decreases smoothly.
            hi = new_lr
            lo = float(min_lr) * (param_group.get('_bb_factor', 1.0))
            new_lr = lo + (hi - lo) * frac
        param_group['lr'] = new_lr
