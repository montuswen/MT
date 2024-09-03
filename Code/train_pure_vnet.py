import os
import argparse
import logging
import time
import random
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from Network.VNet import Vnet
from Loss.loss import get_dice,get_dice_once
from Dataloaders.Dataenhancement import Getdata,CenterCrop,RandomCrop,RandomRotFlip,ToTensor

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default="../Data/12dataset/")
parser.add_argument('--model',type=str,default="pure_vnet")
parser.add_argument('--seed', type=int, default=1337)
parser.add_argument('--max_iterations', type=int, default=8000)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--label_num',type=int,default=12)
parser.add_argument('--base_lr', type=float, default=0.01)
parser.add_argument('--patch_size', type=float, default=(112, 112, 80))
parser.add_argument('--gpu',type=str,default='0')
parser.add_argument('--deterministic',type=bool,default=1,help='use deterministic training or not')
parser.add_argument('--normalization',type=str,default='batchnorm')
args = parser.parse_args()

num_classes=2
data_path=args.data_path
max_iterations=args.max_iterations
batch_size=args.batch_size*len(args.gpu.split(','))
label_num=args.label_num
base_lr=args.base_lr
patch_size=args.patch_size
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
snapshot_path="../model/"+args.model+'/'
if not os.path.exists(snapshot_path):
    os.makedirs(snapshot_path)

if args.deterministic:
    cudnn.benchmark=False
    cudnn.deterministic=True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def worker_init_fn(worker_id):
    random.seed(args.seed+worker_id)

if __name__=='__main__':
    net=Vnet(n_channels=1,n_classes=num_classes,normalization=args.normalization,has_dropout=True).cuda()
    db_train=Getdata(base_dir=data_path,split='train',transform=transforms.Compose([RandomRotFlip(),RandomCrop(patch_size),ToTensor()]))
    db_test=Getdata(base_dir=data_path,split='test',transform=transforms.Compose([CenterCrop(patch_size),ToTensor()]))
    train_loader=DataLoader(db_train,batch_size=batch_size,shuffle=True,num_workers=4,pin_memory=True,worker_init_fn=worker_init_fn)
    test_loader=DataLoader(db_test,batch_size=1,num_workers=4,pin_memory=True)
    optimizer=optim.SGD(net.parameters(),lr=base_lr,momentum=0.9,weight_decay=0.0001)

    log_path=snapshot_path+'log_'+str(label_num)+'/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    writer=SummaryWriter(log_path)
    iter_num=0
    lr_=base_lr
    net.train()
    while True:
        for i_batch,sampled_batch in enumerate(train_loader):
            time1 = time.time()
            iter_num+=1
            volume_batch,label_batch=sampled_batch['image'].cuda(),sampled_batch['label'].cuda()
            output=net(volume_batch)

            loss_seg=F.cross_entropy(output,label_batch)
            output_soft=F.softmax(output,dim=1)
            loss_seg_dice=get_dice(output_soft[:,1,:,:,:],label_batch==1)
            loss=0.5*loss_seg+(1-0.5)*(1-loss_seg_dice)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('lr',lr_,iter_num)
            writer.add_scalar('loss_seg',loss_seg, iter_num)
            writer.add_scalar('loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('loss', loss, iter_num)
            logging.info('iteration {}: loss {:.6f}'.format(iter_num,loss.item()))
            print('iteration {}: loss {:.6f}'.format(iter_num,loss.item()))

            if iter_num % 50 ==0:
                image=volume_batch[0,0:1,:,:,20:61:10].permute(3,0,1,2).repeat(1,3,1,1)
                grid_image=make_grid(image,5,normalize=True)
                writer.add_image('train/Image',grid_image,iter_num)

                output_soft=F.softmax(output,1)
                image=output_soft[0,1:2,:,:,20:61:10].permute(3,0,1,2).repeat(1,3,1,1)
                grid_image=make_grid(image,5,normalize=False)
                writer.add_image('train/Predict_label',grid_image,iter_num)

                image=label_batch[0,:,:,20:61:10].unsqueeze(0).permute(3,0,1,2).repeat(1,3,1,1)
                grid_image=make_grid(image,5,normalize=False)
                writer.add_image('train/Groundtruth_label',grid_image,iter_num)
            if iter_num % 2500 ==0:
                lr_ =base_lr*0.1**(iter_num//2500)
                for param_group in optimizer.param_groups:
                    param_group['lr']=lr_
            if iter_num % 500 ==0:
                train_save_path=os.path.join(snapshot_path,'iter_'+str(iter_num)+'_with_label_num='+str(label_num)+'.pth')
                torch.save(net.state_dict(),train_save_path)
                logging.info('save model to {}'.format(train_save_path))
                print('save model to {}'.format(train_save_path))
            print("this iteration costs {}".format(time.time()-time1))
            if iter_num > max_iterations:
                break
        if iter_num >max_iterations:
            break
    writer.close()