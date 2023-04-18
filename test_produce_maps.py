import torch
import torch.nn.functional as F
import sys
import torch.nn as nn
import numpy as np
import os, argparse
import cv2
from Code.lib.model import HiDANet
from Code.utils.data import test_dataset
import time

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--gpu_id',   type=str, default='2', help='select gpu id')
parser.add_argument('--test_path',type=str, default='/xxxxxxx/TestingSet/',help='test dataset path')
opt = parser.parse_args()

dataset_path = opt.test_path

#set device for test
#if opt.gpu_id=='0':
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#print('USE GPU 0')
 

#load the model
model = HiDANet(32)
model.cuda()

model.load_state_dict(torch.load('./HiDANet_epoch_best.pth'), strict = False)
model.eval()

#test
test_datasets = ['NJU2K','NLPR', 'DES', 'SSD','SIP', 'STERE'] 

#test_datasets = ['STERE'] 


for dataset in test_datasets:
    save_path = './test_maps/HiDANet/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    image_root  = dataset_path + dataset + '/RGB/'
    gt_root     = dataset_path + dataset + '/GT/'
    depth_root  = dataset_path + dataset + '/depth/'
    test_loader = test_dataset(image_root, gt_root,depth_root, opt.testsize)
    times = []
    for i in range(test_loader.size):
        image, gt,depth, name, image_for_post, bin = test_loader.load_data()
        
        gt      = np.asarray(gt, np.float32)
        gt     /= (gt.max() + 1e-8)
        image   = image.cuda()
        depth   = depth.cuda()
        bin = bin.cuda()
        s = time.time()
        pre_res = model(image,depth, bin)
        end = time.time()
        times.append(end-s)
        res     = pre_res[2]     
        res     = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res     = res.sigmoid().data.cpu().numpy().squeeze()
        res     = (res - res.min()) / (res.max() - res.min() + 1e-8)
        
        print('save img to: ',save_path+name)
        cv2.imwrite(save_path+name,res*255)
    print('Test Done!')
    time_sum = 0
    for i in times:
        time_sum += i
    print("FPS: %f" % (1.0 / (time_sum / len(times))))

