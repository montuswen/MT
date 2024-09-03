import os
import argparse
import time
import logging
import time
import random
import numpy as np
import tqdm
from tqdm import tqdm
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from Network.VNet import Vnet
from Loss.loss import get_dice, get_dice_once
import Loss.loss as Loss
import Utils.ramps as ramps
from Dataloaders.Dataenhancement import Getdata, CenterCrop, RandomCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default="../Data/123dataset/")
parser.add_argument('--model', type=str, default='ua-mt')
parser.add_argument('--seed', type=int, default=1337)
parser.add_argument('--max_iterations', type=int, default=8000)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--labeled_batch_size', type=int, default=2)
parser.add_argument('--label_num', type=int, default=12, help='trained samples')
parser.add_argument('--max_samples', type=int, default=123, help='all samples')
parser.add_argument('--base_lr', type=float, default=0.01)
parser.add_argument('--patch_size', type=float, default=(112, 112, 80))
parser.add_argument('--gpu', type=str, default='0,1,2,3')
parser.add_argument('--deterministic', type=bool, default=True, help='whether to use deterministic training')
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str, default='mse')
parser.add_argument('--consistency', type=float, default=0.1)
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
args = parser.parse_args()

patch_size = args.patch_size
num_classes = 2
data_path = args.data_path
batch_size = args.batch_size * len(args.gpu.split(','))
label_bs = args.labeled_batch_size
base_lr = args.base_lr
label_num = args.label_num
time_str=str(time.time())
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_teacher_parameters(student_model, teacher_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
        teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1 - alpha)


if __name__ == '__main__':
    student_model = Vnet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
    student_model=nn.DataParallel(student_model).to(device)
    student_model.train()
    teacher_model = Vnet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
    teacher_model = nn.DataParallel(teacher_model).to(device)
    for param in teacher_model.parameters():
        param.detach_()
    teacher_model.train()
    db_train = Getdata(base_dir=data_path, split='train',
                       transform=transforms.Compose([RandomRotFlip(), RandomCrop(args.patch_size), ToTensor()]))
    db_test = Getdata(base_dir=data_path, split='test',
                      transform=transforms.Compose([CenterCrop(args.patch_size), ToTensor()]))
    labeled_idxs = list(range(label_num))
    unlabeled_idxs = list(range(label_num, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size=args.batch_size,
                                          secondary_batch_size=args.batch_size - args.labeled_batch_size)
    train_loader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                              worker_init_fn=worker_init_fn)
    # test_loader = DataLoader(db_test, batch_size=1, num_workers=4, pin_memory=True)
    optimizer = optim.SGD(student_model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)

    if args.consistency_type == 'mse':
        consistency_criterion = Loss.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = Loss.softmax_kl_loss

    snapshot_path = "../Model/" + args.model + '/' + time_str + '/'
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    log_path = snapshot_path + 'log_' + str(label_num) + '/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    writer = SummaryWriter(log_path)
    iter_num = 0
    lr_ = args.base_lr
    student_model.train()
    while True:
        for i_batch, sampled_batch in enumerate(train_loader):
            time1 = time.time()
            iter_num += 1
            volume_batch, label_batch = sampled_batch['image'].to(device), sampled_batch['label'].to(device)
            unlabeled_volume_batch = volume_batch[label_bs:]

            teacher_input=unlabeled_volume_batch+torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            with torch.no_grad():
                teacher_output=teacher_model(teacher_input)
            student_output=student_model(volume_batch)

            # T = 8
            # preds = []
            # teacher_model.train()
            # for i in range(T):
            #     teacher_input=unlabel_volume_batch+torch.clamp(torch.randn_like(unlabel_volume_batch) * 0.1, -0.2, 0.2)
            #     with torch.no_grad():
            #         preds.append(F.softmax(teacher_model(teacher_input),dim=1))
            # preds = torch.stack(preds)
            # preds = torch.mean(preds, dim=0)
            T = 8
            un_volume_batch_r = unlabeled_volume_batch.repeat(2, 1, 1, 1, 1)
            stride = un_volume_batch_r.shape[0]
            preds = torch.zeros([stride * T, 2, 112, 112, 80]).to(device)
            for i in range(T // 2):
                teacher_input = un_volume_batch_r + torch.clamp(torch.randn_like(un_volume_batch_r) * 0.1, -0.2, 0.2)
                with torch.no_grad():
                    preds[stride * i:stride * (i + 1)] = teacher_model(teacher_input)
            preds = F.softmax(preds, dim=1)
            preds = preds.reshape(T, stride, 2, 112, 112, 80)
            preds = torch.mean(preds, dim=0)

            loss_seg = F.cross_entropy(student_output[:label_bs], label_batch[:label_bs])
            output_soft = F.softmax(student_output, dim=1)
            loss_seg_dice = Loss.get_dice(output_soft[:label_bs, 1, :, :, :], label_batch[:label_bs] == 1)
            supervised_loss = 0.5 * loss_seg + (1 - 0.5) * (1 - loss_seg_dice)

            uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True)
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            consistency_dist = consistency_criterion(student_output[label_bs:], teacher_output)
            threshold = (0.75 + 0.25 * ramps.sigmoid_rampup(iter_num, args.max_iterations)) * np.log(2)
            mask = (uncertainty < threshold).float()
            consistency_dist = torch.sum(mask * consistency_dist) / (2 * torch.sum(mask) + 1e-16)
            consistency_loss = consistency_weight * consistency_dist
            loss = supervised_loss + consistency_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_teacher_parameters(student_model, teacher_model, args.ema_decay, iter_num)

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss_seg', loss_seg, iter_num)
            writer.add_scalar('loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('supervised_loss', supervised_loss, iter_num)
            writer.add_scalar('consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('threshold', threshold, iter_num)
            writer.add_scalar('consistency_loss', consistency_loss, iter_num)
            writer.add_scalar('consistency_dist', consistency_dist, iter_num)
            writer.add_scalar('loss', loss, iter_num)
            logging.info('iteration {}:loss{:.6f} consistency loss{:.6f}'.format(iter_num, loss, consistency_loss))
            print('iteration {}:loss{:.6f} consistency loss{:.6f}'.format(iter_num, loss, consistency_loss))
            print('consistency_weight:{:.6f} consistency_dist:{:.6f}'.format(consistency_weight, consistency_dist))

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
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            if iter_num % 500 == 0:
                train_save_path_teacher = os.path.join(snapshot_path,'teacher_iter_' + str(iter_num) + '_with_label_num=' + str(label_num) + '.pth')
                torch.save(teacher_model.state_dict(), train_save_path_teacher)
                logging.info('save model to {}'.format(train_save_path_teacher))
                print('save model to {}'.format(train_save_path_teacher))
                train_save_path_student = os.path.join(snapshot_path,'student_iter_' + str(iter_num) + '_with_label_num=' + str(label_num) + '.pth')
                torch.save(student_model.state_dict(), train_save_path_student)
                logging.info('save model to {}'.format(train_save_path_student))
                print('save model to {}'.format(train_save_path_student))
            print("this iteration costs {}".format(time.time() - time1))
            if iter_num >= args.max_iterations:
                break
        if iter_num >= args.max_iterations:
            break
    writer.close()
