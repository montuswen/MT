import torch
from torch import nn
import torch.nn.functional as F

class convBlock(nn.Module):
    def __init__(self,n_stages,n_filters_in,n_filters_out,normalization='none'):
        super(convBlock,self).__init__()
        ops=[]
        for i in range(n_stages):
            if i==0:
                input_channel=n_filters_in
            else:
                input_channel=n_filters_out#构造卷积块中每层卷积的进出channel大小
            ops.append(nn.Conv3d(in_channels=input_channel,out_channels=n_filters_out,kernel_size=3,padding=1))
            if normalization== 'batchnorm':#在batch尺度上的归一化
                ops.append(nn.BatchNorm3d(num_features=n_filters_out))
            elif normalization=='groupnorm':#GN归一化
                ops.append(nn.GroupNorm(num_groups=16,num_channels=n_filters_out))
            elif normalization=='instancenorm':#IN归一化
                ops.append(nn.InstanceNorm3d(num_features=n_filters_out))
            ops.append(nn.ReLU(inplace=True))
        self.conv=nn.Sequential(*ops)#实现包装
    def forward(self,x):
        x=self.conv(x)
        return x

class residualConvBlock(nn.Module):
    def __init__(self,n_stages,n_filters_in,n_filters_out,normalization='none'):
        super(residualConvBlock,self).__init__()
        ops=[]
        for i in range(n_stages):
            if i==0:
                input_channel=n_filters_in
            else:
                input_channel=n_filters_out
            ops.append(nn.Conv3d(in_channels=input_channel, out_channels=n_filters_out, kernel_size=3, padding=1))
            if normalization == 'batchnorm':  # 在batch尺度上的归一化
                ops.append(nn.BatchNorm3d(num_features=n_filters_out))
            elif normalization == 'groupnorm':  # GN归一化
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':  # IN归一化
                ops.append(nn.InstanceNorm3d(num_features=n_filters_out))
            if i!=n_stages-1:
                ops.append(nn.ReLU(inplace=True))
        self.conv=nn.Sequential(*ops)
        self.relu=nn.ReLU(inplace=True)
    def forward(self,x):
        x=(self.conv(x)+x)
        x=self.relu(x)
        return x
class DownsamplingBlock(nn.Module):
    def __init__(self,n_filters_in,n_filters_out,stride=2,normalization='none'):
        super(DownsamplingBlock,self).__init__()
        ops=[]
        if normalization!='none':
            ops.append(nn.Conv3d(in_channels=n_filters_in,out_channels=n_filters_out,kernel_size=2,stride=stride))
            if normalization=='batchnorm':
                ops.append(nn.BatchNorm3d(num_features=n_filters_out))
            elif normalization=='groupnorm':
                ops.append(nn.GroupNorm(num_groups=16,num_channels=n_filters_out))
            elif normalization=='instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
        else:
            ops.append(nn.Conv3d(in_channels=n_filters_in,out_channels=n_filters_out,kernel_size=2,stride=stride))
        ops.append(nn.ReLU(inplace=True))
        self.conv=nn.Sequential(*ops)
    def forward(self,x):
        x=self.conv(x)
        return x
class UpsamplingBlock(nn.Module):
    def __init__(self,n_filters_in,n_filters_out,stride=2,normalization='none'):
        super(UpsamplingBlock,self).__init__()
        ops=[]
        if normalization!='none':
            ops.append(nn.ConvTranspose3d(in_channels=n_filters_in,out_channels=n_filters_out,kernel_size=2,stride=stride,padding=0))
            if normalization=='batchnorm':
                ops.append(nn.BatchNorm3d(num_features=n_filters_out))
            elif normalization=='groupnorm':
                ops.append(nn.GroupNorm(num_groups=16,num_channels=n_filters_out))
            elif normalization=='instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
        else:
            ops.append(nn.ConvTranspose3d(in_channels=n_filters_in,out_channels=n_filters_out,kernel_size=2,stride=stride,padding=0))
        ops.append(nn.ReLU(inplace=True))
        self.conv=nn.Sequential(*ops)
    def forward(self,x):
        x=self.conv(x)
        return x
class Upsampling(nn.Module):
    def __init__(self,n_filters_in,n_filters_out,stride=2,normalization='none'):
        super(Upsampling,self).__init__()
        ops=[]
        ops.append(nn.Upsample(scale_factor=stride,mode='trilinear',align_corners=False))
        ops.append(nn.Conv3d(in_channels=n_filters_in,out_channels=n_filters_out,kernel_size=3,padding=1))
        if normalization=='batchnorm':
            ops.append(nn.BatchNorm3d(num_features=n_filters_out))
        elif normalization=='groupnorm':
            ops.append(nn.GroupNorm(num_groups=16,num_channels=n_filters_out))
        elif normalization=='instancenorm':
            ops.append(nn.InstanceNorm3d(num_features=n_filters_out))
        ops.append(nn.ReLU(inplace=True))
        self.conv=nn.Sequential(*ops)
    def forward(self,x):
        x=self.conv(x)
        return x
class Encoder(nn.Module):
    def __init__(self,n_channels=3,n_filters=16,normalization='none',has_dropout=False):
        super(Encoder,self).__init__()
        self.has_dropout=has_dropout
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        self.block_1=convBlock(1,n_channels,n_filters,normalization=normalization)
        self.block_1dw=DownsamplingBlock(n_filters,2*n_filters,normalization=normalization)
        self.block_2=convBlock(2,2*n_filters,2*n_filters,normalization=normalization)
        self.block_2dw=DownsamplingBlock(2*n_filters,4*n_filters,normalization=normalization)
        self.block_3=convBlock(3,4*n_filters,4*n_filters,normalization=normalization)
        self.block_3dw=DownsamplingBlock(4*n_filters,8*n_filters,normalization=normalization)
        self.block_4=convBlock(3,8*n_filters,8*n_filters,normalization=normalization)
        self.block_4dw=DownsamplingBlock(8*n_filters,16*n_filters,normalization=normalization)
        self.block_5=convBlock(3,16*n_filters,16*n_filters,normalization=normalization)

    def encoder(self,input):
        x1=self.block_1(input)
        x1_dw=self.block_1dw(x1)
        x2=self.block_2(x1_dw)
        x2_dw=self.block_2dw(x2)
        x3=self.block_3(x2_dw)
        x3_dw=self.block_3dw(x3)
        x4=self.block_4(x3_dw)
        x4_dw=self.block_4dw(x4)
        x5=self.block_5(x4_dw)
        if self.has_dropout:
            x5=self.dropout(x5)
        res=[x1,x2,x3,x4,x5]
        return res

    def forward(self,input):
        features=self.encoder(input)
        return features

class Decoder(nn.Module):
    def __init__(self,n_classes=2,n_filters=16,normalization='none',has_dropout=False):
        super(Decoder,self).__init__()
        self.has_dropout=has_dropout
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        self.block_5up=UpsamplingBlock(16*n_filters,8*n_filters,normalization=normalization)
        self.block_6=convBlock(3,8*n_filters,8*n_filters,normalization=normalization)
        self.block_6up=UpsamplingBlock(8*n_filters,4*n_filters,normalization=normalization)
        self.block_7=convBlock(3,4*n_filters,4*n_filters,normalization=normalization)
        self.block_7up=UpsamplingBlock(4*n_filters,2*n_filters,normalization=normalization)
        self.block_8=convBlock(2,2*n_filters,2*n_filters,normalization=normalization)
        self.block_8up=UpsamplingBlock(2*n_filters,n_filters,normalization=normalization)
        self.block_9=convBlock(1,n_filters,n_filters,normalization=normalization)
        self.out_conv=nn.Conv3d(n_filters,n_classes,1)

    def decoder(self,features):
        x1=features[0]
        x2=features[1]
        x3=features[2]
        x4=features[3]
        x5=features[4]
        x5_up=self.block_5up(x5)+x4
        x6=self.block_6(x5_up)
        x6_up=self.block_6up(x6)+x3
        x7=self.block_7(x6_up)
        x7_up=self.block_7up(x7)+x2
        x8=self.block_8(x7_up)
        x8_up=self.block_8up(x8)+x1
        x9=self.block_9(x8_up)
        if self.has_dropout:
            x9=self.dropout(x9)
        out=self.out_conv(x9)
        return out

    def forward(self,features):
        out=self.decoder(features)
        return out
class Vnet(nn.Module):
    def __init__(self,n_channels=3,n_classes=2,n_filters=16,normalization='none',has_dropout=False):
        super(Vnet,self).__init__()
        self.has_dropout=has_dropout
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        self.Encoder=Encoder(n_channels,n_filters,normalization,has_dropout)
        self.Decoder=Decoder(n_classes,n_filters,normalization,has_dropout)
        # self.block_1=convBlock(1,n_channels,n_filters,normalization=normalization)
        #
        # self.block_1dw=DownsamplingBlock(n_filters,2*n_filters,normalization=normalization)
        # self.block_2=convBlock(2,2*n_filters,2*n_filters,normalization=normalization)
        # self.block_2dw=DownsamplingBlock(2*n_filters,4*n_filters,normalization=normalization)
        # self.block_3=convBlock(3,4*n_filters,4*n_filters,normalization=normalization)
        # self.block_3dw=DownsamplingBlock(4*n_filters,8*n_filters,normalization=normalization)
        # self.block_4=convBlock(3,8*n_filters,8*n_filters,normalization=normalization)
        # self.block_4dw=DownsamplingBlock(8*n_filters,16*n_filters,normalization=normalization)
        # self.block_5=convBlock(3,16*n_filters,16*n_filters,normalization=normalization)
        #
        # self.block_5up=UpsamplingBlock(16*n_filters,8*n_filters,normalization=normalization)
        # self.block_6=convBlock(3,8*n_filters,8*n_filters,normalization=normalization)
        # self.block_6up=UpsamplingBlock(8*n_filters,4*n_filters,normalization=normalization)
        # self.block_7=convBlock(3,4*n_filters,4*n_filters,normalization=normalization)
        # self.block_7up=UpsamplingBlock(4*n_filters,2*n_filters,normalization=normalization)
        # self.block_8=convBlock(2,2*n_filters,2*n_filters,normalization=normalization)
        # self.block_8up=UpsamplingBlock(2*n_filters,n_filters,normalization=normalization)
        #
        # self.block_9=convBlock(1,n_filters,n_filters,normalization=normalization)
        #
        # self.out_conv=nn.Conv3d(n_filters,n_classes,1)

    # def encoder(self,input):
    #     x1=self.block_1(input)
    #     x1_dw=self.block_1dw(x1)
    #     x2=self.block_2(x1_dw)
    #     x2_dw=self.block_2dw(x2)
    #     x3=self.block_3(x2_dw)
    #     x3_dw=self.block_3dw(x3)
    #     x4=self.block_4(x3_dw)
    #     x4_dw=self.block_4dw(x4)
    #     x5=self.block_5(x4_dw)
    #     if self.has_dropout:
    #         x5=self.dropout(x5)
    #     res=[x1,x2,x3,x4,x5]
    #     return res
    # def decoder(self,features):
    #     x1=features[0]
    #     x2=features[1]
    #     x3=features[2]
    #     x4=features[3]
    #     x5=features[4]
    #     x5_up=self.block_5up(x5)+x4
    #     x6=self.block_6(x5_up)
    #     x6_up=self.block_6up(x6)+x3
    #     x7=self.block_7(x6_up)
    #     x7_up=self.block_7up(x7)+x2
    #     x8=self.block_8(x7_up)
    #     x8_up=self.block_8up(x8)+x1
    #     x9=self.block_9(x8_up)
    #     if self.dropout:
    #         x9=self.dropout(x9)
    #     out=self.out_conv(x9)
    #     return out
    def forward(self,input,turnoff_drop=False):
        if turnoff_drop:
            has_dropout=self.has_dropout
            self.has_dropout=False
        features=self.Encoder.encoder(input)
        out=self.Decoder.decoder(features)
        if turnoff_drop:
            self.has_dropout=has_dropout
        return out