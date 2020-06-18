# ——————————————————————————————————
# Author : Zhiyu Xue
# Reference :  Learning to Compare: Relation Network for Few-Shot Learning
# ——————————————————————————————————

import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable 
import torch.nn.functional as F

"------------Relative Map Network--------------"
class Phi(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(Phi,self).__init__()
        self.layer1 = nn.Sequential(
             nn.Conv2d(2,2,kernel_size=3,padding=False),
             nn.BatchNorm2d(2,momentum=1,affine=True),
             nn.LeakyReLU(),
             nn.MaxPool2d(2)
        ) 
        self.fc1 = nn.Linear(2*8*8,hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)
        
    def forward(self,x):
        out = self.layer1(x)
        out = out.view(out.size(0),-1)
        out = self.fc1(out)
        out = self.fc2(out)

        return out

class RM_Network(nn.Module):
    def __init__(self,input_size,hidden_size,channels=64,GPU=0):
        super(RM_Network,self).__init__()
        self.channels = channels
        self.GPU = GPU
        self.matching = nn.ModuleDict()

        for i in range(self.channels):
            name = "phi_"+str(i) # 命名
            self.matching[name] = Phi(input_size,hidden_size)
        # 把每一张feature map的比较结果加权相加
        self.weights = nn.Linear(channels,1)

    def forward(self,x1,x2):
        batch_size = x1.size(0)
        out = Variable(torch.FloatTensor()).cuda(self.GPU)
        for i in range(self.channels):
            inputing = torch.stack([x1[:,i],x2[:,i]],dim=1) #拼接后输入
            #inputing = inputing.unsqueeze(0)
            #print(inputing.size())
            index = "phi_"+str(i)
            model = self.matching[index] #找到与之匹配的phi
            #print(model)
            output = model(inputing)
            out = torch.cat([out,output],dim=-1)
            #print(out.size())

        out = self.weights(out)

        out = torch.sigmoid(out)
        return out
"----------------------------------------------------------------"

"------------------------Relative Position Networks----------------------------------------"
class Phi_PoTPo(nn.Module):
    def __init__(self,in_channels=64,r=2): 
        super(Phi_PoTPo,self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_channels*2,in_channels),
            nn.ReLU()
        )
        self.meta_learner = nn.Sequential(
            nn.Linear(in_channels,int(in_channels*r)),
            nn.ReLU(),
            nn.Linear(int(in_channels*r),in_channels)
        )

    def forward(self,x):
        x = self.encoder(x)
        meta_weight = self.meta_learner(x)
        out = meta_weight.mul(x)
        out = out.sum(dim=-1).unsqueeze(-1)
        return out


class RP_Network(nn.Module):
    def __init__(self,in_wide=19,in_channels=64,r=0.5,GPU=0): # is_conv 最后是否要使用卷积神经网络
        super(RP_Network,self).__init__()
        self.in_channels = in_channels
        self.r = r
        self.GPU = GPU
        self.matching = Phi_PoTPo(in_channels=self.in_channels,r=self.r)
        self.in_wide = in_wide
    
     # x -> [num_class*batch_size,2*in_channels,19,19]
    def forward(self,x):
        x = x.permute([0,2,3,1]) # x -> [num_class*batch_size,19,19,2*in_channels]
        out = self.matching(x)
        out = out.permute([0,3,1,2])
        out = torch.sigmoid(out)
        return out # out ->  [num_class*batch_size,1,19,19]       
"--------------------------------------------------------------------"




"-----------------------Feature Encoder---------------------------------------------"
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
 
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class CNNEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2)) #保留前面两个网络

        self.layer3 = BasicBlock(64,64)

        self.layer4 = BasicBlock(64,64)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out
"---------------------------------------------------------------------------"




        