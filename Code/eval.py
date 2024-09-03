import os
import argparse
import torch
import h5py
import math
import nibabel as nib
import numpy as np
from medpy import metric
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from Network.VNet import Vnet
from tensorboardX import SummaryWriter

parser=argparse.ArgumentParser()
parser.add_argument('--data_path',type=str,default="../Data/123dataset/")
parser.add_argument("--model",type=str,default="mt")
parser.add_argument("--gpu",type=str,default='0')
parser.add_argument('--label_num',type=int,default=12)
parser.add_argument('--time',type=str,default='1720271142.0315888')
parser.add_argument('--model_choose',type=str,default='teacher')
args=parser.parse_args()

num_classes=2
label_num=args.label_num
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
snapshot_path="../Model/"+args.model+'/'+args.time+'/'
with open(args.data_path+'train.list','r') as f:
    image_list=f.readlines()
image_list=[args.data_path+item.replace('\n','')+'/mri_norm2.h5' for item in image_list]

def cal_metric_curracy(pred,gt):
    dice=metric.binary.dc(pred,gt)
    jaccard=metric.binary.jc(pred,gt)
    hausdorff=metric.binary.hd95(pred,gt)
    averdis=metric.binary.asd(pred,gt)
    return dice,jaccard,hausdorff,averdis

def eval_single(net,image,stride,patch_size,num_classes):
    w,h,d=image.shape
    add_pad=False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    # print(add_pad)
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww,hh,dd = image.shape
    # print(stride)
    sx = math.ceil((ww - patch_size[0]) / stride[0]) + 1
    sy = math.ceil((hh - patch_size[1]) / stride[1]) + 1
    sz = math.ceil((dd - patch_size[2]) / stride[2]) + 1 #滑动窗口的次数
    # print(sx,sy,sz)
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)#对每个像素点计数,便于计算平均预测概率
    for x in range(0, sx):
        xs = min(stride[0] * x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride[1] * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride[2] * z, dd-patch_size[2])
                #print(xs,ys,zs)
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                # print("test_patch {}".format(test_patch.shape))
                ytemp = net(test_patch)
                y = F.softmax(ytemp, dim=1)
                # print("yyyy:{}".format(y.shape))
                # print(y)
                # if 0 in y[0,0,:,:,:]>y[0,1,:,:,:]:
                #     print("right!")
                # else:
                #     print("error!")
                # return
                y = y.cpu().data.numpy()
                y = y[0,:,:,:,:]
                # print("y:{}".format(y))
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    # print("score map")
    # print(score_map.shape)
    score_map = score_map / np.expand_dims(cnt, axis=0)
    # print(score_map)
    pred_map = np.argmax(score_map, axis=0)
    # if 1 in pred_map.flatten():
    #     print("yes")
    # else:
    #     print("no")
    if add_pad:
        pred_map = pred_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        score_map = score_map[:, wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
    return pred_map,score_map
def eval_all(net,image_list,num_classes,patch_size=(112,112,80),stride=(18,18,4),eval_save_path=None,save_result=True,preprocess_fn=None):
    total_metric=0.0
    for image_path in tqdm(image_list):
        id=image_path.split('/')[-2]
        h5f=h5py.File(image_path,'r')
        image=h5f['image'][:]
        label=h5f['label'][:]
        if preprocess_fn is not None:
            image=preprocess_fn(image)
        # print(image.shape)
        # print(stride)
        # print(patch_size)
        # print(num_classes)
        pred,score=eval_single(net,image,stride,patch_size,num_classes=num_classes)
        # if 1 in pred.flatten():
        #     print("yes")
        # else:
        #     print("no")
        if np.sum(pred)==0:
            single_metric=(0,0,0,0)
        else:
            single_metric=cal_metric_curracy(pred,label)
        # print("single metric is")
        # print("{:.6f} {:.6f} {:.6f} {:.6f}".format(single_metric[0],single_metric[1],single_metric[2],single_metric[3]))
        total_metric+=np.asarray(single_metric)
        if save_result:
            nib.save(nib.Nifti1Image(pred.astype(np.float32),np.eye(4)),eval_save_path+str(id)+'_pred.nii.gz')
            nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), eval_save_path + str(id) + '_img.nii.gz')
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), eval_save_path + str(id) + '_gt.nii.gz')
    avg_metric=total_metric/len(image_list)
    return avg_metric
if __name__=='__main__':
    log_path=snapshot_path+'eval_'+args.model_choose+'_log_'+str(label_num)+'/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    writer=SummaryWriter(log_path)
    for iter_num in range(15000,30001,500):
        eval_save_path = snapshot_path +args.model_choose+ '_eval_save_label_num'+str(label_num)+'_'+str(iter_num)+'/'
        if not os.path.exists(eval_save_path):
            os.mkdir(eval_save_path)
        train_save_path=os.path.join(snapshot_path,args.model_choose+'_iter_' + str(iter_num) + '_with_label_num=' + str(label_num) + '.pth')
        if not os.path.exists(train_save_path):
            continue
        print('init weight from {}'.format(train_save_path))
        net=Vnet(n_channels=1,n_classes=num_classes,normalization='batchnorm',has_dropout=False)
        net=nn.DataParallel(net)
        net.to(device)
        net.load_state_dict(torch.load(train_save_path),strict=False)
        net.eval()
        avg_metric=eval_all(net=net,image_list=image_list,num_classes=num_classes,eval_save_path=eval_save_path,save_result=False)
        print("aver metric is {}".format(avg_metric))
        writer.add_scalar('dice',avg_metric[0],iter_num)
        writer.add_scalar('jaccard',avg_metric[1],iter_num)
        writer.add_scalar('haus',avg_metric[2],iter_num)
        writer.add_scalar('averdis',avg_metric[3],iter_num)
    writer.close()