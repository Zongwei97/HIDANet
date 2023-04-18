import os
import torch
import torch.nn.functional as F
import sys
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from Code.lib.model import HiDANet
from Code.utils.data import get_loader,test_dataset
from Code.utils.utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from Code.utils.options import opt



#set the device for training
#if opt.gpu_id=='2':
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#print('USE GPU 2')

  
cudnn.benchmark = True

#build the model
model = HiDANet(32)
if(opt.load is not None):
    model.load_state_dict(torch.load(opt.load))
    print('load model from ',opt.load)
    
model.cuda()
params    = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)


#set the path
train_image_root = opt.rgb_label_root
train_gt_root    = opt.gt_label_root
train_depth_root = opt.depth_label_root

val_image_root   = opt.val_rgb_root
val_gt_root      = opt.val_gt_root
val_depth_root   = opt.val_depth_root
save_path        = opt.save_path


if not os.path.exists(save_path):
    os.makedirs(save_path)

#load data
print('load data...')
train_loader = get_loader(train_image_root, train_gt_root,train_depth_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
test_loader  = test_dataset(val_image_root, val_gt_root,val_depth_root, opt.trainsize)

val_image_root   = opt.val_rgb_root.replace('SIP', 'STERE')
val_gt_root	 = opt.val_gt_root.replace('SIP', 'STERE')
val_depth_root   = opt.val_depth_root.replace('SIP', 'STERE')

test_loader1  = test_dataset(val_image_root, val_gt_root,val_depth_root, opt.trainsize)

total_step   = len(train_loader)


logging.basicConfig(filename=save_path+'log.log',format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level = logging.INFO,filemode='a',datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("BBSNet_unif-Train")
logging.info("Config")
logging.info('epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(opt.epoch,opt.lr,opt.batchsize,opt.trainsize,opt.clip,opt.decay_rate,opt.load,save_path,opt.decay_epoch))

#set loss function
CE   = torch.nn.BCEWithLogitsLoss()

step = 0
writer     = SummaryWriter(save_path+'summary')
best_mae   = 1
best_epoch = 0

print(len(train_loader))


def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()


def iou_loss(pred, mask):
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return wiou.mean()

def train(train_loader, model, optimizer, epoch,save_path):
    global step
    model.train()
    loss_all=0
    epoch_step=0
    try:
        for i, (images, gts, depths, bin) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            
            images   = images.cuda()
            gts      = gts.cuda()
            depths   = depths.cuda()
            bin = bin.cuda()

            ##
            pre_res  = model(images,depths, bin)
            
            loss1    = structure_loss(pre_res[0], gts) 
            loss2    = structure_loss(pre_res[1], gts)
            loss3    = structure_loss(pre_res[2], gts) 
            loss1u = iou_loss(pre_res[0], gts)
            loss2u = iou_loss(pre_res[1], gts)
            loss3u = iou_loss(pre_res[2], gts)

            loss3r    = structure_loss(pre_res[3], gts) 
            loss4r    = structure_loss(pre_res[4], gts)
            loss5r    = structure_loss(pre_res[5], gts) 
            loss6r    = structure_loss(pre_res[6], gts) 
            loss3ru = iou_loss(pre_res[3], gts)
            loss4ru = iou_loss(pre_res[4], gts)
            loss5ru = iou_loss(pre_res[5], gts)
            loss6ru = iou_loss(pre_res[6], gts)

            loss3d    = structure_loss(pre_res[7], gts) 
            loss4d    = structure_loss(pre_res[8], gts)
            loss5d    = structure_loss(pre_res[9], gts) 
            loss6d    = structure_loss(pre_res[10], gts) 
            loss3du = iou_loss(pre_res[7], gts)
            loss4du = iou_loss(pre_res[8], gts)
            loss5du = iou_loss(pre_res[9], gts)
            loss6du = iou_loss(pre_res[10], gts)

            loss3m    = structure_loss(pre_res[11], gts) 
            loss4m    = structure_loss(pre_res[12], gts)
            loss5m    = structure_loss(pre_res[13], gts) 
            loss6m    = structure_loss(pre_res[14], gts) 
            loss3mu = iou_loss(pre_res[11], gts)
            loss4mu = iou_loss(pre_res[12], gts)
            loss5mu = iou_loss(pre_res[13], gts)
            loss6mu = iou_loss(pre_res[14], gts)

            
            
            loss_seg = loss1 + loss2 + loss3 + loss1u + loss2u + loss3u \
                   + 0.8 * (loss3r + loss3ru + loss3d + loss3du + loss3m + loss3mu) \
                   + 0.6 * (loss4r + loss4ru + loss4d + loss4du + loss4m + loss4mu) \
                   + 0.4 * (loss5r + loss5ru + loss5d + loss5du + loss5m + loss5mu) \
                   + 0.2 * (loss6r + loss6ru + loss6d + loss6du + loss6m + loss6mu)

            loss = loss_seg 
            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step+=1
            epoch_step+=1
            loss_all+=loss.data
            if i % 50 == 0 or i == total_step or i==1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f} Loss2: {:0.4f} Loss3: {:0.4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step, loss1.data, loss2.data,  loss3.data))
                logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f} Loss2: {:0.4f} Loss3: {:0.4f}'.
                    format( epoch, opt.epoch, i, total_step, loss1.data, loss2.data, loss3.data))
                
        loss_all/=epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format( epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        
        if (epoch) % 5 == 0:
            torch.save(model.state_dict(), save_path+'HyperNet_epoch_{}.pth'.format(epoch))
            
    except KeyboardInterrupt: 
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path+'HyperNet_epoch_{}.pth'.format(epoch+1))
        print('save checkpoints successfully!')
        raise
        
        
        
#test function
def val(test_loader, test_loader1, model,epoch,save_path):
    global best_mae,best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum=0
        for i in range(test_loader.size):
            image, gt,depth,  name,img_for_post, bin = test_loader.load_data()
            gt      = np.asarray(gt, np.float32)
            gt     /= (gt.max() + 1e-8)
            image   = image.cuda()
            depth   = depth.cuda()
            bin = bin.cuda()
            pre_res = model(image,depth, bin)
            res     = pre_res[2]
            res     = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res     = res.sigmoid().data.cpu().numpy().squeeze()
            res     = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res-gt))*1.0/(gt.shape[0]*gt.shape[1])
            

        #mae_sum1=0
        #for i in range(test_loader1.size):
        #    image, gt,depth,  name,img_for_post, bin = test_loader1.load_data()
        #    gt      = np.asarray(gt, np.float32)
        #    gt     /= (gt.max() + 1e-8)
        #    image   = image.cuda()
        #    depth   = depth.cuda()
        #    bin = bin.cuda()
        #    pre_res = model(image,depth, bin)
        #    res     = pre_res[2]
        #    res     = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        #    res     = res.sigmoid().data.cpu().numpy().squeeze()
        #    res     = (res - res.min()) / (res.max() - res.min() + 1e-8)
        #    mae_sum1 += np.sum(np.abs(res-gt))*1.0/(gt.shape[0]*gt.shape[1])

        mae = mae_sum/test_loader.size #+ mae_sum1/test_loader1.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch,mae,best_mae,best_epoch))
        if epoch==1:
            best_mae = mae
        else:
            if mae<best_mae:
                best_mae   = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path+'HiDANet_epoch_best.pth')
                print('best epoch:{}'.format(epoch))
                
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch,mae,best_epoch,best_mae))
 
if __name__ == '__main__':
    print("Start train...")
    
    for epoch in range(1, opt.epoch):
        
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        # train
        train(train_loader, model, optimizer, epoch,save_path)
        
        #test
        val(test_loader, test_loader1, model,epoch,save_path)
