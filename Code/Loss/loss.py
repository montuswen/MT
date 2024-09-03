import torch.nn.functional as F
import torch.nn as nn
import torch
from einops import rearrange

def get_dice(output,target,eps=1e-4):
    target=target.float()
    inter=torch.sum(output*target)+eps
    union=torch.sum(output*output)+torch.sum(target*target)+eps*2
    dice=2*inter/union
    return dice
def get_dice_once(output,target,eps=1e-4):
    output=torch.argmax(output,dim=1)
    inter=torch.sum(output*target)+eps
    union=torch.sum(output)+torch.sum(target)+eps*2
    dice=2*inter/union
    return dice
def softmax_kl_loss(input,target,sigmoid=False):
    assert input.size()==target.size()
    if sigmoid:
        input_log_softmax=torch.log(torch.sigmoid(input))
        target_softmax=torch.sigmoid(target)
    else:
        input_log_softmax=F.log_softmax(input,dim=1)
        target_softmax=F.softmax(target,dim=1)
    kl_div=F.kl_div(input_log_softmax,target_softmax,reduction='mean')
    return kl_div

def softmax_mse_loss(input,target):
    assert input.size()==target.size()
    input_softmax=F.softmax(input,dim=1)
    target_softmax=F.softmax(target,dim=1)
    mse_loss=F.mse_loss(input_softmax,target_softmax)
    return mse_loss
class lossmodel(nn.Module):
    def __init__(self,n_classes,alpha=0.5):
        super(lossmodel,self).__init__()
        self.n_classes=n_classes
        self.alpha=alpha

    def forward(self,input,target):
        smooth=0.001
        input1=F.softmax(input,dim=1)
        target1=F.one_hot(target,self.n_classes)
        input1=rearrange(input1,'b n h w s -> b n (h w s)')
        target1=rearrange(target1,'b h w s n -> b n (h w s)')
        #扁平化
        input1=input1[:,1:,:]
        target1=target1[:,1:,:].float()
        #上段代码的目的？存疑
        inter=torch.sum(input1*target1)+smooth
        union=torch.sum(input1*input1)+torch.sum(target1*target1)+2*smooth
        dice_loss=1-2*inter/union

        totalloss=(1-self.alpha)*F.cross_entropy(input,target)+self.alpha*dice_loss
        return totalloss
