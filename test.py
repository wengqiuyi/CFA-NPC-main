import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse

from lib.model import CFANet
from utils.dataloader import get_loader,test_dataset
from utils.eva_funcs import eval_Smeasure,eval_mae,numpy2tensor


import scipy.io as scio 
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='./checkpoint/CFANet.pth')

for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
    
    print('-----------strating -------------')
    
    data_path = '/test/Polpy/Dataset/TestDataset/{}/'.format(_data_name)
    save_path = './Snapshot/seg_maps/{}/'.format(_data_name)
    
    
    
    opt   = parser.parse_args()
    model = CFANet(channel=64, dual_backbone=True).cuda()
    
    
    state = torch.load(opt.pth_path)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    elif isinstance(state, dict) and 'model' in state:
        state = state['model']
    msg = model.load_state_dict(state, strict=False)
    print('loaded {}  missing={}  unexpected={}'.format(opt.pth_path, len(msg.missing_keys), len(msg.unexpected_keys)))
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    
    image_root = '{}/images/'.format(data_path)
    gt_root    = '{}/masks/'.format(data_path)
    

    
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        
        print(['--------------processing-------------', i])
        
        image, gt, name = test_loader.load_data()
        
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        _,_,_,res = model(image)

        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        cv2.imwrite(save_path+name, (res * 255).astype(np.uint8))
