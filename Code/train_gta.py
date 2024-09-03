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
import Loss.loss as Loss
import Utils.ramps as ramps
from Dataloaders.Dataenhancement import Getdata,CenterCrop,RandomCrop,RandomRotFlip,ToTensor,TwoStreamBatchSampler

parser=argparse.ArgumentParser()
parser.add_argument('--data_path',type=str,default="../Data/123dataset")
parser.add_argument('--data_path_f', type=str, default="../Data/12dataset/")
parser.add_argument('--student_model_path',type=str,default='../Model/pure_vnet/iter_4000_with_label_num=12.pth')
parser.add_argument('--model',type=str,default='gta')
parser.add_argument('--seed',type=int,default=21)
parser.add_argument('--max_iterations',type=int,default=10000)
parser.add_argument('--batch_size',type=int,default=4)
parser.add_argument('--labeled_batch_size',type=int,default=2)
parser.add_argument('--label_num',type=int,default=12,help='trained samples')
parser.add_argument('--max_samples',type=int,default=123,help='all samples')
parser.add_argument('--base_lr',type=float,default=0.01)
parser.add_argument('--patch_size',type=float,default=(112,112,80))
parser.add_argument('--gpu',type=str,default='0')
parser.add_argument('--deterministic',type=bool,default=True,help='whether to use deterministic training')
parser.add_argument('--ema_decay',type=float,default=0.999,help='ema_decay')
parser.add_argument('--consistency_type',type=str,default='mse')
parser.add_argument('--consistency',type=float,default=0.1)
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--sup_only_time',type=int,default=4000)
args=parser.parse_args()

patch_size=args.patch_size
num_classes=2
data_path=args.data_path
data_path_f=args.data_path_f
label_bs=args.labeled_batch_size
base_lr=args.base_lr
label_num=args.label_num
sup_only=args.sup_only_time
os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
snapshot_path='../model/'+args.model+'/'
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

def get_current_consistency_weight(epoch):
    return args.consistency*ramps.sigmoid_rampup(epoch,args.consistency_rampup)

def update_teacher_parameters(student_model, teacher_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
        teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1 - alpha)

def update_student_feature_para_with_gta(student_model,gta_model,alpha,global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for student_param,gta_param in zip(student_model.Encoder.parameters(),gta_model.Encoder.parameters()):
        student_param.data.mul_(alpha).add_(gta_param.data,alpha=1-alpha)

if __name__ == '__main__':
    student_model=Vnet(n_channels=1,n_classes=num_classes,normalization='batchnorm',has_dropout=True).cuda()
    student_model.train()
    teacher_model=Vnet(n_channels=1,n_classes=num_classes,normalization='batchnorm',has_dropout=True).cuda()
    teacher_model.train()
    for param in teacher_model.parameters():
        param.detach_()
        param.requires_grad=False
    gta_model=Vnet(n_channels=1,n_classes=num_classes,normalization='batchnorm',has_dropout=True).cuda()
    gta_model.train()
    db_train=Getdata(base_dir=data_path,split='train',transform=transforms.Compose([RandomRotFlip(),RandomCrop(args.patch_size),ToTensor()]))
    db_train_f = Getdata(base_dir=data_path_f, split='train',transform=transforms.Compose([RandomRotFlip(), RandomCrop(args.patch_size), ToTensor()]))
    db_test=Getdata(base_dir=data_path,split='test',transform=transforms.Compose([CenterCrop(args.patch_size),ToTensor()]))
    labeled_idxs=list(range(label_num))
    unlabeled_idxs=list(range(label_num,args.max_samples))
    batch_sampler=TwoStreamBatchSampler(labeled_idxs,unlabeled_idxs,batch_size=args.batch_size,secondary_batch_size=args.batch_size-args.labeled_batch_size)
    train_loader=DataLoader(db_train,batch_sampler=batch_sampler,num_workers=4,pin_memory=True,worker_init_fn=worker_init_fn)
    train_loader_f = DataLoader(db_train_f, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,worker_init_fn=worker_init_fn)
    test_loader=DataLoader(db_test,batch_size=1,num_workers=4,pin_memory=True)
    optimizer_student=optim.SGD(student_model.parameters(),lr=args.base_lr,momentum=0.9,weight_decay=1e-4)
    optimizer_gta=optim.SGD(gta_model.parameters(),lr=args.base_lr,momentum=0.9,weight_decay=1e-4)

    log_path=snapshot_path+'log_' + str(label_num) + '/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    writer=SummaryWriter(log_path)
    iter_num=0
    # max_epochs=2*args.max_iterations//len(train_loader)+1
    lr_=base_lr
    while True:
        if iter_num < sup_only:
            for i_batch, sampled_batch in enumerate(train_loader_f):
                time1=time.time()
                iter_num+=1
                volume_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()

                student_model.train()
                student_output = student_model(volume_batch)
                loss_seg = F.cross_entropy(student_output, label_batch)
                output_soft = F.softmax(student_output, dim=1)
                loss_seg_dice = Loss.get_dice(output_soft[:, 1, :, :, :], label_batch[:] == 1)
                supervised_loss = 0.5 * loss_seg + (1 - 0.5) * (1 - loss_seg_dice)
                unsupervised_loss = 0.0
                loss=supervised_loss+unsupervised_loss
                optimizer_student.zero_grad()
                supervised_loss.backward()
                optimizer_student.step()

                writer.add_scalar('supervised_loss', supervised_loss, iter_num)
                writer.add_scalar('unsupervised_loss', unsupervised_loss, iter_num)
                writer.add_scalar('loss', loss, iter_num)
                logging.info('iteration {}:loss{:.6f} supervised loss{:.6f} unsupervised loss{:.6f}'.format(iter_num, loss,supervised_loss,unsupervised_loss))
                print('iteration {}:loss{:.6f} supervised loss{:.6f} unsupervised loss{:.6f}'.format(iter_num, loss,supervised_loss,unsupervised_loss))

                if iter_num % 50 == 0:
                    image = volume_batch[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=True)
                    writer.add_image('train/Image', grid_image, iter_num)

                    with torch.no_grad():
                        eval_output = student_model(volume_batch)
                    output_soft = F.softmax(eval_output, 1)
                    image = output_soft[0, 1:2, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=False)
                    writer.add_image('train/Predict_label', grid_image, iter_num)

                    image = label_batch[0, :, :, 20:61:10].unsqueeze(0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=False)
                    writer.add_image('train/Groundtruth_label', grid_image, iter_num)
                if iter_num % 2500 == 0:
                    lr_ = base_lr * 0.1 ** (iter_num // 2500)
                    for param_group in optimizer_student.param_groups:
                        param_group['lr'] = lr_
                    for param_group in optimizer_gta.param_groups:
                        param_group['lr'] = lr_
                if iter_num % 500 == 0:
                    train_save_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '_with_label_num=' + str(label_num) + '.pth')
                    if iter_num <= sup_only:
                        torch.save(student_model.state_dict(), train_save_path)
                    else:
                        torch.save(teacher_model.state_dict(), train_save_path)
                    logging.info('save model to {}'.format(train_save_path))
                    print('save model to {}'.format(train_save_path))
                print("this iteration costs {}".format(time.time() - time1))
        else:
            if iter_num == sup_only:
                with torch.no_grad():
                    for teacher_params, student_params in zip(teacher_model.parameters(), student_model.parameters()):
                        teacher_params.data = student_params.data
                    for gta_params, student_params in zip(gta_model.parameters(), student_model.parameters()):
                        gta_params.data = student_params.data
            for i_batch, sampled_batch in enumerate(train_loader):
                time1=time.time()
                iter_num+=1

                volume_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()
                unlabeled_volume_batch = volume_batch[label_bs:]

                noise=torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
                unlabel_input=unlabeled_volume_batch+noise
                teacher_model.eval()
                teacher_output=teacher_model(unlabel_input)
                teacher_soft_max=F.softmax(teacher_output,dim=1)
                _,fake_label=torch.max(teacher_soft_max,dim=1)

                gta_model.train()
                gta_output=gta_model(unlabel_input)
                teacher_model.train()
                with torch.no_grad():
                    teacher_output=teacher_model(unlabel_input)

                # T = 8
                # preds = []
                # teacher_model.train()
                # for i in range(T):
                #     teacher_input = unlabel_volume_batch + torch.clamp(torch.randn_like(unlabel_volume_batch) * 0.1,
                #                                                        -0.2, 0.2)
                #     with torch.no_grad():
                #         preds.append(F.softmax(teacher_model(teacher_input), dim=1))
                # preds = torch.stack(preds)
                # preds = torch.mean(preds, dim=0)

                batch_size,num_classes,h,w,d=gta_output.shape
                with torch.no_grad():
                    prob = torch.softmax(teacher_output, dim=1)
                    conf, ps_label = torch.max(prob, dim=1)
                    conf = conf.detach()
                    conf_thresh = np.percentile(conf.cpu().numpy().flatten(), 20)
                    thresh_mask = conf.le(conf_thresh).bool()
                    conf[thresh_mask] = 0
                    weight = batch_size * h * w * d / (torch.sum(thresh_mask== 0) + 1e-6)
                loss_ = weight * F.cross_entropy(gta_output, fake_label,reduction='none')
                conf = (conf + 1.0) / (conf + 1.0).sum() * (torch.sum(fake_label != 0) + 1e-6)
                unsupervised_loss = torch.mean(conf * loss_)
                unsupervised_loss*=get_current_consistency_weight(iter_num//150)
                # print("unsuploss is {}".format(unsupervised_loss))
                # T = 8
                # un_volume_batch_r = unlabeled_volume_batch.repeat(2, 1, 1, 1, 1)
                # stride = un_volume_batch_r.shape[0]
                # preds = torch.zeros([stride * T, 2, 112, 112, 80]).cuda()
                # for i in range(T // 2):
                #     teacher_input = un_volume_batch_r + torch.clamp(torch.randn_like(un_volume_batch_r) * 0.1, -0.2, 0.2)
                #     with torch.no_grad():
                #         preds[stride * i:stride * (i + 1)] = teacher_model(teacher_input)
                # preds = F.softmax(preds, dim=1)
                # preds = preds.reshape(T, stride, 2, 112, 112, 80)
                # preds = torch.mean(preds, dim=0)
                # certainty=1+1.0*torch.sum(preds*torch.log(preds+1e-6),dim=1,keepdim=True)
                # unsupervised_weight=get_current_consistency_weight(iter_num//150)
                # unsupervised_dist=F.cross_entropy(gta_output,fake_label)
                # threshold=1-(0.75+0.25*ramps.sigmoid_rampup(iter_num,args.max_iterations))*np.log(2)
                # mask=(certainty>threshold).float()
                # weight_mutrix=(tau+certainty)*mask.float()
                # weight_mutrix/=(2*torch.sum(weight_mutrix)+1e-6)
                # unsupervised_loss=unsupervised_weight*torch.sum(weight_mutrix*unsupervised_dist)
                optimizer_gta.zero_grad()
                unsupervised_loss.backward()
                optimizer_gta.step()
                student_model.eval()
                with torch.no_grad():
                    update_student_feature_para_with_gta(student_model,gta_model,args.ema_decay,iter_num)
                student_model.train()

                student_output = student_model(volume_batch[:label_bs])
                loss_seg = F.cross_entropy(student_output, label_batch[:label_bs])
                output_soft = F.softmax(student_output, dim=1)
                loss_seg_dice = Loss.get_dice(output_soft[:, 1, :, :, :], label_batch[:label_bs] == 1)
                supervised_loss = 0.5 * loss_seg + (1 - 0.5) * (1 - loss_seg_dice)
                loss = supervised_loss + unsupervised_loss
                optimizer_student.zero_grad()
                supervised_loss.backward()
                optimizer_student.step()
                teacher_model.eval()
                with torch.no_grad():
                    update_teacher_parameters(student_model,teacher_model,args.ema_decay,iter_num)

                writer.add_scalar('supervised_loss', supervised_loss, iter_num)
                writer.add_scalar('unsupervised_loss', unsupervised_loss, iter_num)
                writer.add_scalar('loss', loss, iter_num)
                logging.info('iteration {}:loss{:.6f} supervised loss{:.6f} unsupervised loss{:.6f}'.format(iter_num, loss,supervised_loss, unsupervised_loss))
                print('iteration {}:loss{:.6f} supervised loss{:.6f} unsupervised loss{:.6f}'.format(iter_num, loss,supervised_loss, unsupervised_loss))

                if iter_num % 50 == 0:
                    image = volume_batch[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=True)
                    writer.add_image('train/Image', grid_image, iter_num)

                    with torch.no_grad():
                        eval_teacher_output=teacher_model(volume_batch)
                    output_soft = F.softmax(eval_teacher_output, 1)
                    image = output_soft[0, 1:2, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=False)
                    writer.add_image('train/Predict_label', grid_image, iter_num)

                    image = label_batch[0, :, :, 20:61:10].unsqueeze(0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=False)
                    writer.add_image('train/Groundtruth_label', grid_image, iter_num)
                if iter_num % 2500 == 0:
                    lr_ = base_lr * 0.1 ** (iter_num // 2500)
                    for param_group in optimizer_student.param_groups:
                        param_group['lr'] = lr_
                    for param_group in optimizer_gta.param_groups:
                        param_group['lr'] = lr_
                if iter_num % 500 == 0:
                    train_save_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '_with_label_num=' + str(label_num) + '.pth')
                    if iter_num<=sup_only:
                        torch.save(student_model.state_dict(), train_save_path)
                    else:
                        torch.save(teacher_model.state_dict(), train_save_path)
                    logging.info('save model to {}'.format(train_save_path))
                    print('save model to {}'.format(train_save_path))
                print("this iteration costs {}".format(time.time() - time1))
                if iter_num>=args.max_iterations:
                    break
            if iter_num >= args.max_iterations:
                break