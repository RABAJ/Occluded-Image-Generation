import os
import sys
import time
import datetime
import argparse
import argparse
import itertools
import random

#import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
from torchvision import transforms

'''from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal'''

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torchvision.models as models
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary
import torch.nn as nn
#from utils import weights_init_normal

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)


class Encoder1(nn.Module):
    def __init__(self, n_residual_blocks=9):
        super(Encoder1, self).__init__()


        self.d = {}
        self.re0 = nn.ReflectionPad2d(3)
        self.conv0=nn.Conv2d(3, 64, 7)
        self.insn0=nn.InstanceNorm2d(64)
        self.r0=nn.ReLU(inplace=True)

        # Downsampling
        in_features = 64
        out_features = in_features*2



        self.conv1=nn.Conv2d(in_features, out_features, 3, stride=2, padding=1)
        self.insn1 =nn.InstanceNorm2d(out_features)
        self.r1=nn.ReLU(inplace=True)
        in_features = out_features
        out_features = in_features*2
        self.conv2 = nn.Conv2d(in_features, out_features, 3, stride=2, padding=1)
        self.insn2 = nn.InstanceNorm2d(out_features)
        self.r2 = nn.ReLU(inplace=True)
        in_features = out_features
        out_features = in_features * 2
        self.conv3 = nn.Conv2d(in_features, out_features, 3, stride=2, padding=1)
        self.insn3 = nn.InstanceNorm2d(out_features)
        self.r3 = nn.ReLU(inplace=True)
        #in_features = out_features
        #out_features = in_features * 2


    def forward(self, x):
        x1 = self.re0(x)
        x2 = self.conv0(x1)
        x3 = self.insn0(x2)
        x4 = self.r0(x3)
        x5 = self.conv1(x4)
        x6 = self.insn1(x5)
        x7 = self.r1(x6)
        x8 = self.conv2(x7)
        x9 = self.insn2(x8)
        x10 = self.r2(x9)
        x11 = self.conv3(x10)
        x12 = self.insn3(x11)
        x15 = self.r3(x12)






        return x15,x10,x7,x4


class Encoder2(nn.Module):
    def __init__(self, n_residual_blocks=9):
        super(Encoder2, self).__init__()

        self.d = {}
        self.re0 = nn.ReflectionPad2d(3)
        self.conv0 = nn.Conv2d(3, 64, 7)
        self.insn0 = nn.InstanceNorm2d(64)
        self.r0 = nn.ReLU(inplace=True)

        # Downsampling
        in_features = 64
        out_features = in_features * 2

        self.conv1 = nn.Conv2d(in_features, out_features, 3, stride=2, padding=1)
        self.insn1 = nn.InstanceNorm2d(out_features)
        self.r1 = nn.ReLU(inplace=True)
        in_features = out_features
        out_features = in_features * 2
        self.conv2 = nn.Conv2d(in_features, out_features, 3, stride=2, padding=1)
        self.insn2 = nn.InstanceNorm2d(out_features)
        self.r2 = nn.ReLU(inplace=True)
        in_features = out_features
        out_features = in_features * 2
        self.conv3 = nn.Conv2d(in_features, out_features, 3, stride=2, padding=1)
        self.insn3 = nn.InstanceNorm2d(out_features)
        self.r3 = nn.ReLU(inplace=True)
        in_features = out_features
        out_features = in_features * 2

    def forward(self, x):
        x1 = self.re0(x)
        x2 = self.conv0(x1)
        x3 = self.insn0(x2)
        x4 = self.r0(x3)
        x5 = self.conv1(x4)
        x6 = self.insn1(x5)
        x7 = self.r1(x6)
        x8 = self.conv2(x7)
        x9 = self.insn2(x8)
        x10 = self.r2(x9)
        x11 = self.conv3(x10)
        x12 = self.insn3(x11)
        x15 = self.r3(x12)

        return x15,x10,x7,x4




'''encoder1=Encoder1()
encoder1=encoder1.cuda()
encoder1=torch.nn.DataParallel(encoder1)
encoder2=Encoder2()
encoder2=encoder2.cuda()
encoder2=torch.nn.DataParallel(encoder2)'''
encoder1=Encoder1()
encoder2=Encoder2()
model1 = encoder1
#model1 = torch.nn.DataParallel(model1)
#model1.load_state_dict(torch.load('/home/ws2/Downloads//Market1501_train_test/model/model.pth'))
#model1.eval()
model2 = encoder2
#model2 = torch.nn.DataParallel(model2)

#model2.load_state_dict(torch.load('/home/ws2/Downloads//Market1501_train_test/model/model.pth'))
#model2.eval()

import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator1(nn.Module):
    def __init__(self, encoder1,input_nc,output_nc, n_residual_blocks=9):
        super(Generator1, self).__init__()

        # Initial convolution block

        self.d = {}
        self.lin2conv = nn.Sequential(nn.ReflectionPad2d(3),
                                      nn.Conv2d(1, 64, 7),
                                      nn.InstanceNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.pose2conv = nn.Sequential(nn.ReflectionPad2d(3),
                                      nn.Conv2d(1, 64, 7),
                                      nn.InstanceNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.pose_conv1=nn.Sequential(  nn.Conv2d(64, 128, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(128),
                        nn.ReLU(inplace=True) )
        self.pose_conv2=nn.Sequential(  nn.Conv2d(128, 256, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(256),
                        nn.ReLU(inplace=True) )
        self.pose_conv3 = nn.Sequential(nn.Conv2d(256, 256, 3, stride=2, padding=1),
                                        nn.InstanceNorm2d(512),
                                        nn.ReLU(inplace=True))
        self.layer_bal = nn.Sequential(nn.Conv2d(256, 256, 3, stride=2, padding=1),
                                       nn.InstanceNorm2d(512),
                                       nn.ReLU(inplace=True))


        # Downsampling
        in_features = 64
        out_features = in_features*2
        self.model1=nn.Sequential( nn.ReflectionPad2d(0), nn.Conv2d(64, 128, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) )
        self.model2 = nn.Sequential(nn.ReflectionPad2d(0), nn.Conv2d(128, 256, 3, stride=2, padding=1),
                                    nn.InstanceNorm2d(out_features),
                                    nn.ReLU(inplace=True))
        self.model3 = nn.Sequential(nn.ReflectionPad2d(0), nn.Conv2d(256, 512, 3, stride=2, padding=1),
                                    nn.InstanceNorm2d(out_features),
                                    nn.ReLU(inplace=True))
        self.model4 =nn.Sequential(nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True))
        self.model5 = nn.Sequential(nn.ConvTranspose2d(768,256, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True))
        self.model6 = nn.Sequential(nn.ConvTranspose2d(384, 128, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True))
        self.model7 = nn.Sequential(nn.ReflectionPad2d(3),
                                         nn.Conv2d(192, 64, 7),
                                         nn.Tanh())

        self.model8 = nn.Sequential(nn.ReflectionPad2d(3),
                                         nn.Conv2d(64, output_nc, 7),
                                         nn.Tanh())



        '''for i in range(3):
            self.d['model'+ str(i+1)]= nn.Sequential( nn.ReflectionPad2d(0), nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) )
            self.d['model' + str(i + 1)]=self.d['model'+ str(i+1)].cuda()
            in_features = out_features
            out_features = in_features*2'''

        # Residual blocks



        # Upsampling
        '''in_features1=out_features
        out_features1 = in_features
        self.li=[0.256,128]
        for j in range(3):

            self.d['model'+ str(j+4)]= nn.Sequential(nn.ConvTranspose2d(in_features1, out_features1//2, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True))
            self.d['model' + str(j + 4)] = self.d['model' + str(j + 4)].cuda()
            in_features1 = out_features1
            out_features1 = out_features1//2'''


        # Output layer
        '''self.d['model7'] = nn.Sequential(  nn.ReflectionPad2d(3),
                    nn.Conv2d(128, 64, 7),
                    nn.Tanh() )
        self.d['model7']= self.d['model7'].cuda()
        self.d['model8'] = nn.Sequential(nn.ReflectionPad2d(3),
                                         nn.Conv2d(64, output_nc, 7),
                                         nn.Tanh())
        self.d['model8'] = self.d['model8'].cuda()'''
        self.encoder1=encoder1
        self.linearr=nn.Sequential(nn.Conv1d(27, 256,1), nn.Tanh(),nn.Conv1d(256, 1024,1),nn.Tanh(),nn.Conv1d(1024, 4096,1),nn.Tanh(),nn.Conv1d(4096, 16384,1),nn.Tanh())

        #model = nn.Sequential(*model)
        #self.model= nn.DataParallel(model)

    '''def forward(self, x):
        return self.model(x)'''
    def forward(self, x,y,z):
        '''x1 = self.d['model0'](x)
        x2 = self.d['model1'](x1)
        x3 = self.d['model2'](x2)
        x4 = self.d['model3'](x3)'''
        #en_model=self.encoder1.load_state_dict(torch.load('/home/ws2/Downloads//Market1501_train_test/model/model.pth'))
        en_model=self.encoder1
        #en_model.eval()
        x4,x3,x2,x1=en_model(x)
        z1=self.pose2conv(z)
        z2=self.pose_conv1(z1)
        z3=self.pose_conv2(z2)
        z3=self.pose_conv3(z3)
        y1 = torch.flatten(y)
        '''y2 = nn.Linear(27, 256)(y1)
        y3 = nn.Linear(256, 1024)(y2)
        y4 = nn.Linear(1024, 4096)(y3)
        y5 = nn.Linear(4096, 16384)(y4)'''
        y1 = y1.view(-1, 27, 1)
        '''y2 = nn.Conv1d(27, 256,1)(y1)
        y3 = nn.Conv1d(256, 1024,1)(y2)
        y4 = nn.Conv1d(1024, 4096,1)(y3)
        y5 = nn.Conv1d(4096, 16384,1)(y4)'''
        y5=self.linearr(y1)
        y6 = y5.view(-1, 1, 128, 128)
        y7 = self.lin2conv(y6)
        y8 = self.model1(y7)
        y9 = self.model2(y8)
        y10=self.layer_bal(y9)
        #y10 = self.d['model3'](y9)
        o1 = torch.cat([x4,z3,y10], dim=1)
        #o1=o1.cuda()
        o2 = self.model4(o1)
        o3 = torch.cat([o2, x3], dim=1)
        o4 = self.model5(o3)
        o5 = torch.cat([o4, x2], dim=1)
        o6 = self.model6(o5)
        o7 = torch.cat([o6, x1], dim=1)
        o8 = self.model7(o7)
        o8 = self.model8(o8)



        return o8


class Generator2(nn.Module):
    def __init__(self, encoder2, input_nc, output_nc, n_residual_blocks=9):
        super(Generator2, self).__init__()

        # Initial convolution block

        self.d = {}
        self.lin2conv = nn.Sequential(nn.ReflectionPad2d(3),
                                      nn.Conv2d(1, 64, 7),
                                      nn.InstanceNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.pose2conv = nn.Sequential(nn.ReflectionPad2d(3),
                                       nn.Conv2d(1, 64, 7),
                                       nn.InstanceNorm2d(64),
                                       nn.ReLU(inplace=True))
        self.pose_conv1 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1),
                                        nn.InstanceNorm2d(128),
                                        nn.ReLU(inplace=True))
        self.pose_conv2 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1),
                                        nn.InstanceNorm2d(256),
                                        nn.ReLU(inplace=True))
        self.pose_conv3 = nn.Sequential(nn.Conv2d(256, 256, 3, stride=2, padding=1),
                                        nn.InstanceNorm2d(512),
                                        nn.ReLU(inplace=True))
        self.layer_bal = nn.Sequential(nn.Conv2d(256, 256, 3, stride=2, padding=1),
                                       nn.InstanceNorm2d(512),
                                       nn.ReLU(inplace=True))

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        self.model1 = nn.Sequential(nn.ReflectionPad2d(0), nn.Conv2d(64, 128, 3, stride=2, padding=1),
                                    nn.InstanceNorm2d(out_features),
                                    nn.ReLU(inplace=True))
        self.model2 = nn.Sequential(nn.ReflectionPad2d(0), nn.Conv2d(128, 256, 3, stride=2, padding=1),
                                    nn.InstanceNorm2d(out_features),
                                    nn.ReLU(inplace=True))
        self.model3 = nn.Sequential(nn.ReflectionPad2d(0), nn.Conv2d(256, 512, 3, stride=2, padding=1),
                                    nn.InstanceNorm2d(out_features),
                                    nn.ReLU(inplace=True))
        self.model4 = nn.Sequential(nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=1, output_padding=1),
                                    nn.InstanceNorm2d(out_features),
                                    nn.ReLU(inplace=True))
        self.model5 = nn.Sequential(nn.ConvTranspose2d(768, 256, 3, stride=2, padding=1, output_padding=1),
                                    nn.InstanceNorm2d(out_features),
                                    nn.ReLU(inplace=True))
        self.model6 = nn.Sequential(nn.ConvTranspose2d(384, 128, 3, stride=2, padding=1, output_padding=1),
                                    nn.InstanceNorm2d(out_features),
                                    nn.ReLU(inplace=True))
        self.model7 = nn.Sequential(nn.ReflectionPad2d(3),
                                    nn.Conv2d(192, 64, 7),
                                    nn.Tanh())

        self.model8 = nn.Sequential(nn.ReflectionPad2d(3),
                                    nn.Conv2d(64, output_nc, 7),
                                    nn.Tanh())

        '''for i in range(3):
            self.d['model'+ str(i+1)]= nn.Sequential( nn.ReflectionPad2d(0), nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) )
            self.d['model' + str(i + 1)]=self.d['model'+ str(i+1)].cuda()
            in_features = out_features
            out_features = in_features*2'''

        # Residual blocks

        # Upsampling
        '''in_features1=out_features
        out_features1 = in_features
        self.li=[0.256,128]
        for j in range(3):

            self.d['model'+ str(j+4)]= nn.Sequential(nn.ConvTranspose2d(in_features1, out_features1//2, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True))
            self.d['model' + str(j + 4)] = self.d['model' + str(j + 4)].cuda()
            in_features1 = out_features1
            out_features1 = out_features1//2'''

        # Output layer
        '''self.d['model7'] = nn.Sequential(  nn.ReflectionPad2d(3),
                    nn.Conv2d(128, 64, 7),
                    nn.Tanh() )
        self.d['model7']= self.d['model7'].cuda()
        self.d['model8'] = nn.Sequential(nn.ReflectionPad2d(3),
                                         nn.Conv2d(64, output_nc, 7),
                                         nn.Tanh())
        self.d['model8'] = self.d['model8'].cuda()'''
        self.encoder2 = encoder2
        self.linearr = nn.Sequential(nn.Conv1d(27, 256, 1), nn.Tanh(), nn.Conv1d(256, 1024, 1), nn.Tanh(),
                                     nn.Conv1d(1024, 4096, 1), nn.Tanh(), nn.Conv1d(4096, 16384, 1), nn.Tanh())

        # model = nn.Sequential(*model)
        # self.model= nn.DataParallel(model)

    '''def forward(self, x):
        return self.model(x)'''

    def forward(self, x, y, z):
        '''x1 = self.d['model0'](x)
        x2 = self.d['model1'](x1)
        x3 = self.d['model2'](x2)
        x4 = self.d['model3'](x3)'''
        # en_model=self.encoder1.load_state_dict(torch.load('/home/ws2/Downloads//Market1501_train_test/model/model.pth'))
        en_model = self.encoder2
        # en_model.eval()
        x4, x3, x2, x1 = en_model(x)
        z1 = self.pose2conv(z)
        z2 = self.pose_conv1(z1)
        z3 = self.pose_conv2(z2)
        z3 = self.pose_conv3(z3)
        y1 = torch.flatten(y)
        '''y2 = nn.Linear(27, 256)(y1)
        y3 = nn.Linear(256, 1024)(y2)
        y4 = nn.Linear(1024, 4096)(y3)
        y5 = nn.Linear(4096, 16384)(y4)'''
        y1 = y1.view(-1, 27, 1)
        '''y2 = nn.Conv1d(27, 256,1)(y1)
        y3 = nn.Conv1d(256, 1024,1)(y2)
        y4 = nn.Conv1d(1024, 4096,1)(y3)
        y5 = nn.Conv1d(4096, 16384,1)(y4)'''
        y5 = self.linearr(y1)
        y6 = y5.view(-1, 1, 128, 128)
        y7 = self.lin2conv(y6)
        y8 = self.model1(y7)
        y9 = self.model2(y8)
        y10 = self.layer_bal(y9)
        # y10 = self.d['model3'](y9)
        o1 = torch.cat([x4, z3, y10], dim=1)
        # o1=o1.cuda()
        o2 = self.model4(o1)
        o3 = torch.cat([o2, x3], dim=1)
        o4 = self.model5(o3)
        o5 = torch.cat([o4, x2], dim=1)
        o6 = self.model6(o5)
        o7 = torch.cat([o6, x1], dim=1)
        o8 = self.model7(o7)
        o8 = self.model8(o8)

        return o8







'''def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)




encoder1=Encoder1()
encoder2=Encoder2()

class Verification_Classifier(nn.Module):

    def __init__(self,encoder1,encoder2):
        super(Verification_Classifier, self).__init__()

        self.bn1 = nn.BatchNorm2d(512)

        self.avgpool=nn.AvgPool2d(kernel_size=28)
        self.fc = nn.Linear(512 , 1)
        self.encoder1=encoder1
        self.encoder2=encoder2


    def forward(self,x,y):
        x=x.cuda()
        y=y.cuda()
        v1 =self.encoder1(x)
        v2 = self.encoder2(y)
        # s=p2-p1
        s = torch.sub(v2, v1)
        po = torch.pow(s, 2)
        pa = self.bn1(po)
        #print(pa.size())
        paa=self.avgpool(pa)
        #print(paa.size())
        x = torch.flatten(paa, 1)
        fc = self.fc(x)
        f = torch.sigmoid(fc)
        #print(f)

        return f
        
        
class verification_loss(torch.nn.Module):
    def __init__(self):
        super(verification_loss, self).__init__()


    def forward(self, c,Vdv):
        #print(type(Vdv))
        s1=Vdv.size()[0]
        vl = (-c * (torch.log(Vdv))) + ((1 - c) * (1 + torch.log(Vdv)))
        s = torch.Tensor([248])

        vl = torch.sum(vl)
        #print("verification_sum",vl)
        loss=vl/s1
        #print("loss",loss)
        #print('vl_size',type(vl))
        return loss

'''




class res1(nn.Module):
    def __init__(self, n_residual_blocks=9):
        super(res1, self).__init__()


        self.d = {}
        self.re0 = nn.ReflectionPad2d(3)
        self.conv0=nn.Conv2d(3, 64, 7)
        self.insn0=nn.InstanceNorm2d(64)
        self.r0=nn.ReLU(inplace=True)

        # Downsampling
        in_features = 64
        out_features = in_features*2



        self.conv1=nn.Conv2d(in_features, out_features, 3, stride=2, padding=1)
        self.insn1 =nn.InstanceNorm2d(out_features)
        self.r1=nn.ReLU(inplace=True)
        in_features = out_features
        out_features = in_features*2
        self.conv2 = nn.Conv2d(in_features, out_features, 3, stride=2, padding=1)
        self.insn2 = nn.InstanceNorm2d(out_features)
        self.r2 = nn.ReLU(inplace=True)
        in_features = out_features
        out_features = in_features * 2
        self.conv3 = nn.Conv2d(in_features, out_features, 3, stride=2, padding=1)
        self.insn3 = nn.InstanceNorm2d(out_features)
        self.r3 = nn.ReLU(inplace=True)
        in_features = out_features
        out_features = in_features * 2


    def forward(self, x):
        x1 = self.re0(x)
        x2 = self.conv0(x1)
        x3 = self.insn0(x2)
        x4 = self.r0(x3)
        x5 = self.conv1(x4)
        x6 = self.insn1(x5)
        x7 = self.r1(x6)
        x8 = self.conv2(x7)
        x9 = self.insn2(x8)
        x10 = self.r2(x9)
        x11 = self.conv3(x10)
        x12 = self.insn3(x11)
        x15 = self.r3(x12)






        return x15


class res2(nn.Module):
    def __init__(self, n_residual_blocks=9):
        super(res2, self).__init__()

        self.d = {}
        self.re0 = nn.ReflectionPad2d(3)
        self.conv0 = nn.Conv2d(3, 64, 7)
        self.insn0 = nn.InstanceNorm2d(64)
        self.r0 = nn.ReLU(inplace=True)

        # Downsampling
        in_features = 64
        out_features = in_features * 2

        self.conv1 = nn.Conv2d(in_features, out_features, 3, stride=2, padding=1)
        self.insn1 = nn.InstanceNorm2d(out_features)
        self.r1 = nn.ReLU(inplace=True)
        in_features = out_features
        out_features = in_features * 2
        self.conv2 = nn.Conv2d(in_features, out_features, 3, stride=2, padding=1)
        self.insn2 = nn.InstanceNorm2d(out_features)
        self.r2 = nn.ReLU(inplace=True)
        in_features = out_features
        out_features = in_features * 2
        self.conv3 = nn.Conv2d(in_features, out_features, 3, stride=2, padding=1)
        self.insn3 = nn.InstanceNorm2d(out_features)
        self.r3 = nn.ReLU(inplace=True)
        in_features = out_features
        out_features = in_features * 2

    def forward(self, x):
        x1 = self.re0(x)
        x2 = self.conv0(x1)
        x3 = self.insn0(x2)
        x4 = self.r0(x3)
        x5 = self.conv1(x4)
        x6 = self.insn1(x5)
        x7 = self.r1(x6)
        x8 = self.conv2(x7)
        x9 = self.insn2(x8)
        x10 = self.r2(x9)
        x11 = self.conv3(x10)
        x12 = self.insn3(x11)
        x15 = self.r3(x12)

        return x15




'''encoder1=Encoder1()
encoder1=encoder1.cuda()
encoder1=torch.nn.DataParallel(encoder1)
encoder2=Encoder2()
encoder2=encoder2.cuda()
encoder2=torch.nn.DataParallel(encoder2)'''
res1=res1()
res2=res2()

class Identity_Discriminator1(nn.Module):

    def __init__(self,res1,res2):
        super(Identity_Discriminator1, self).__init__()

        self.bn1 = nn.BatchNorm2d(512)

        self.avgpool=nn.AvgPool2d(kernel_size=16)
        self.fc = nn.Linear(512 , 1)
        self.res1=res1
        self.res2=res2


    def forward(self,x,y):
        x=x.cuda()
        y=y.cuda()
        v1 =self.res1(x)
        v2 = self.res2(y)
        #print("v1",v1.shape)
        # s=p2-p1
        s = torch.sub(v2, v1)
        #print("s",s.shape)
        po = torch.pow(s, 2)
        #print("po",po.shape)
        pa = self.bn1(po)
        #print("pa",pa.shape)
        #print(pa.size())
        paa=self.avgpool(pa)
        #print("paa",paa.shape)
        #print(paa.size())
        x = torch.flatten(paa, 1)
        fc = self.fc(x)
        f = torch.sigmoid(fc)
        #print(f.shape)

        return f
class Identity_Discriminator2(nn.Module):

    def __init__(self,res1,res2):
        super(Identity_Discriminator2, self).__init__()

        self.bn1 = nn.BatchNorm2d(512)

        self.avgpool=nn.AvgPool2d(kernel_size=16)
        self.fc = nn.Linear(512 , 1)
        self.res1=res1
        self.res2=res2


    def forward(self,x,y):
        x=x.cuda()
        y=y.cuda()
        v1 =self.res1(x)
        v2 = self.res2(y)
        #print("v1",v1.shape)
        # s=p2-p1
        s = torch.sub(v2, v1)
        #print("s",s.shape)
        po = torch.pow(s, 2)
        #print("po",po.shape)
        pa = self.bn1(po)
        #print("pa",pa.shape)
        #print(pa.size())
        paa=self.avgpool(pa)
        #print("paa",paa.shape)
        #print(paa.size())
        x = torch.flatten(paa, 1)
        fc = self.fc(x)
        f = torch.sigmoid(fc)
        #print(f.shape)

        return f
class Pose_Discriminator1(nn.Module):
    def __init__(self):
        super(Pose_Discriminator1, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(6, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x,y):
        o1 = torch.cat([y, x], dim=1)
        x =  self.model(o1)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

class Pose_Discriminator2(nn.Module):
    def __init__(self):
        super(Pose_Discriminator2, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(6, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x,y):
        o1 = torch.cat([y, x], dim=1)
        x =  self.model(o1)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


class Identity_Discriminator_loss(torch.nn.Module):
    def __init__(self):
        super(Identity_Discriminator_loss, self).__init__()


    def forward(self,c, Idv1,Idv2):
        #print(type(Vdv))
        #print(Idv1)
        #print(Idv2)
        s1=Idv1.size()[0]
        print("Idv1",Idv1)
        s2=Idv2.size()[0]
        tp=torch.log(Idv1)
        print(tp)
        tp1=torch.log(Idv2)
        print(tp1)
        tp2=torch.log(1-Idv1)
        print(tp2)
        tp3=torch.log(Idv2)
        print(tp3)
        vl = c*(torch.log(Idv1) + torch.log(Idv2))+(1-c)*(torch.log(1-Idv1) + torch.log(Idv2))
        #s = torch.Tensor([248])

        vl = torch.sum(vl)
        #print("verification_sum",vl)
        loss=vl/s1
        #print("loss",loss)
        #print('vl_size',type(vl))
        return loss


class Pose_Discriminator_loss(torch.nn.Module):
    def __init__(self):
        super(Pose_Discriminator_loss, self).__init__()


    def forward(self,c, Idv1,Idv2):
        #print(type(Vdv))
        s1=Idv1.size()[0]
        #s2=Idv2.size()[0]
        vl =  c*(torch.log(Idv1) + torch.log(Idv2))+(1-c)*(torch.log(1-Idv1) + torch.log(Idv2))
        #s = torch.Tensor([248])

        vl = torch.sum(vl)
        #print("verification_sum",vl)
        loss=vl/s1
        #print("loss",loss)
        #print('vl_size',type(vl))
        return loss

class Same_Pose_loss(torch.nn.Module):
    def __init__(self):
        super(Same_Pose_loss, self).__init__()


    def forward(self, Idv1,Idv2,Idv3):
        #print(type(Vdv))
        s0=Idv1.size()[0]
        s1=Idv1.size()[1]
        #s2=Idv2.size()[1]
        #vl = torch.log(Idv1) + torch.log(1-Idv2)
        #s = torch.Tensor([248])
        #s=s1*s2

        #vl = torch.sum(vl)
        pl=torch.sum(torch.add((torch.sub(Idv1, Idv3) / s1 * s1),(torch.sub(Idv2, Idv3) / s1 * s1)))

        #print("verification_sum",vl)
        loss=pl/s0
        #print("loss",loss)
        #print('vl_size',type(vl))
        return loss


class Reconstruction_loss(torch.nn.Module):
    def __init__(self):
        super(Reconstruction_loss, self).__init__()

    def forward(self, c,Idv1, Idv2,Idv3):
        # print(type(Vdv))
        s0 = Idv1.size()[0]
        s1 = Idv1.size()[1]
        s2 = Idv2.size()[1]
        # vl = torch.log(Idv1) + torch.log(1-Idv2)
        # s = torch.Tensor([248])
        s = s1 * s2

        #vl = torch.sum(vl)
        pl = torch.sum(torch.add((torch.sub(Idv1, Idv3) / s1 * s1),(torch.sub(Idv2, Idv3) / s1 * s1)))

        # print("verification_sum",vl)
        loss = pl / s0
        # print("loss",loss)
        # print('vl_size',type(vl))
        return c*(loss)+(1-c)*(1-loss)



import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
from PIL import Image
import json

'''root = '/home/ws2/Downloads/occluded_person_reidentification/AiC/crops/'
json1 = '/home/ws2/Downloads/occluded_person_reidentification/AiC/annotations.json'''

root1='/home/ws2/Downloads/paras_dec_2019/Market1501_train_test/nonocc_train_v1/'
root2='/home/ws2/Downloads/paras_dec_2019/Market1501_train_test/crops_occ_train/'
root3='/home/ws2/Downloads/paras_dec_2019/Market1501_train_test/crops_occ_diffpose_train/'
root4='/home/ws2/Downloads/paras_dec_2019/Market1501_train_test/crops_occ_diffiden_train/'
root5='/home/ws2/Downloads/paras_dec_2019/Market1501_train_test/target_nonocc_pose_train/'


class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

class AiCDataset(data.Dataset):
    """AiC Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root1,root2,root3,root4,root5, transform=None,transform1=None):
        """Set the path for images and attributes.

        Args:
            root: image directory.
            json1: aic annotation file path.

            transform: image transformer.
        """
        self.root1 = root1
        self.root2 = root2
        self.root3 = root3
        self.root4=root4
        self.root5=root5
        from scipy.io import loadmat
        a = loadmat('/home/ws2/Downloads/paras_dec_2019/market_attributes/dataset.mat')
        #b = loadmat('/home/ws2/Downloads/market_attributes/image_index.mat')
        ''''for key,values in a.items() :
            print (key)
            print("value")
            #print(values)'''
        #print(a)
        import numpy

        self.attri = numpy.asarray(a['dataset'])
        #self.ids=numpy.asarray(b['image_index'][])
        #self.ids=numpy.load('/home/ws2/Downloads/Market1501_train_test/image_index.npy')
        self.ids=numpy.load('/home/ws2/Downloads/paras_dec_2019/Market1501_train_test/market_v1_iden.npy')
        #self.ids=self.ids[:-4]
        #print(a[0].shape)
        #with open(json1) as f:
        #    annots = json.load(f)

        #self.attri = [x['attributes'] for x in annots]
        #self.ids = [x for x in range(len(self.attri))]

        self.transform = transform
        self.transform1=transform1

    def __getitem__(self, index):
        """Returns one data pair (image and attribute)."""


        #attri = self.attri

        ann_id = self.ids[index]
        attrib=self.attri[int(ann_id[0:4])]

        #attrib = attri[ann_id]
        '''attrib=attri[index]'''
        attrib=attrib-1
        if attrib[0]==0 or attrib[0]==1:
            attrib[0]=0
        elif attrib[0]==2 or attrib[0]==3:
            attrib[0]=1

        #img_id = self.ids[ann_id]
        #img_id=int(ann_id)
        img_id=ann_id

        #path1 = str(img_id) + '.jpg'
        #path2= str(img_id) +'_occ'+ '.jpg'
        path=str(img_id)+'.jpg'

        image1 = Image.open(os.path.join(self.root1, path))
        image2 = Image.open(os.path.join(self.root2, path))
        image3 = Image.open(os.path.join(self.root3, path))
        image4= Image.open(os.path.join(self.root4, path))
        image5 = Image.open(os.path.join(self.root5, path))
        if self.transform is not None:
            #print(self.transform)
            image1 = self.transform(image1)
            image2 = self.transform(image2)
            image3 = self.transform(image3)
            image4 = self.transform(image4)
            image5 = self.transform1(image5)

        #target_attrib = torch.Tensor(attrib)
        target_veri=torch.Tensor([1,0])
        return image1,image2,image3,image4,image5,attrib,target_veri


    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, attribute).

    We should build custom collate_fn rather than using default collate_fn,
    because merging attribute (including padding) is not supported in default.
    Args:
        data: list of tuple (image, attribute).
            - image: torch tensor of shape (3, 256, 256).
            - attribute: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, 24).
        lengths: list; valid length for each attribute.
    """
    # Sort a data list by attribute length (descending order).
    data.sort(key=lambda x: len(x[3]), reverse=True)
    images1,images2,images3,images4,images5,attribs,veri = zip(*data)
    #print(images1.size())

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images1 = torch.stack(images1, 0)
    images2 = torch.stack(images2, 0)
    images3 = torch.stack(images3, 0)
    images4 = torch.stack(images4, 0)
    images5 = torch.stack(images5, 0)
    attribs = torch.from_numpy(np.asarray(attribs)).float()

    # Merge attributes (from tuple of 1D tensor to 2D tensor).
    lengths_attrib = [len(cap) for cap in attribs]
    targets_attrib = torch.zeros(len(attribs), max(lengths_attrib)).long()
    lengths_veri = [len(cap) for cap in veri]
    targets_veri = torch.zeros(len(veri), max(lengths_veri)).long()
    for i, cap in enumerate(attribs):
        end = lengths_attrib[i]
        targets_attrib[i, :end] = cap[:end]
    for i, cap in enumerate(veri):
        end = lengths_veri[i]
        targets_veri[i, :end] = cap[:end]
    return images1,images2,images3,images4,images5,targets_attrib,lengths_attrib,targets_veri,lengths_veri


def get_loader(root1,root2,root3,root4,root5, transform,transform1, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom aic dataset."""

    aic = AiCDataset(root1,root2,root3,root4,root5,
                     transform=transform,transform1=transform1)

    # Data loader for AiC dataset
    # This will return (images, attributes, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # attributes: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each attribute. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=aic,
                                              batch_size=batch_size,
                                              shuffle=shuffle,

                                              collate_fn=collate_fn)
    return data_loader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Image preprocessing, normalization for the pretrained resnet
transform = transforms.Compose([
    transforms.RandomResizedCrop(128),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))])
transform1 = transforms.Compose([
    transforms.RandomResizedCrop(128),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()])

# Load vocabulary wrapper


# Build data loader
data_loader = get_loader(root1,root2,root3,root4,root5,transform,transform1, 30,
                         shuffle=True, num_workers=2)

netG_A2B = Generator1(encoder1,3, 3)
netG_C2D = Generator2(encoder2,input_nc=3, output_nc=3)
# netG_B2A = Generator(opt.output_nc, opt.input_nc)
# netD_A = Discriminator(opt.input_nc)
netD_Id1 = Identity_Discriminator1(res1,res2)
netD_Id2 = Identity_Discriminator2(res1,res2)
netPD1 = Pose_Discriminator1()
netPD2 = Pose_Discriminator2()



netG_A2B.cuda()
netG_C2D.cuda()
# netD_A.cuda()
netD_Id1.cuda()
netPD1.cuda()
netD_Id2.cuda()
netPD2.cuda()

netG_A2B.apply(weights_init_normal)
netG_C2D.apply(weights_init_normal)
# netG_B2A.apply(weights_init_normal)
# netD_A.apply(weights_init_normal)
netD_Id1.apply(weights_init_normal)
netPD1.apply(weights_init_normal)
netD_Id2.apply(weights_init_normal)
netPD2.apply(weights_init_normal)

# Lossess
'''criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()'''

# Optimizers & LR schedulers
optimizer_G_A2B = torch.optim.Adam(itertools.chain(netG_A2B.parameters()),
                               lr=0.001, betas=(0.5, 0.999))
optimizer_G_C2D = torch.optim.Adam(itertools.chain(netG_C2D.parameters()),
                               lr=0.001, betas=(0.5, 0.999))
# optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_Id1 = torch.optim.Adam(netD_Id1.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizer_D_pose1 = torch.optim.Adam(netPD1.parameters(), lr=0.01, betas=(0.5, 0.999))
optimizer_D_Id2 = torch.optim.Adam(netD_Id2.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizer_D_pose2 = torch.optim.Adam(netPD2.parameters(), lr=0.01, betas=(0.5, 0.999))

'''lr_scheduler_G_A2B = torch.optim.lr_scheduler.LambdaLR(optimizer_G_A2B,
                                                   lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)'''
a=[50]
a.extend(range(51,100,1))
lr_scheduler_G_A2B = torch.optim.lr_scheduler.MultiStepLR(optimizer_G_A2B,milestones=a,gamma=0.1)
lr_scheduler_G_C2D = torch.optim.lr_scheduler.MultiStepLR(optimizer_G_C2D,milestones=a,gamma=0.1)
lr_scheduler_D_Id1 = torch.optim.lr_scheduler.MultiStepLR(optimizer_D_Id1,milestones=a,gamma=0.1)
lr_scheduler_D_pose1 = torch.optim.lr_scheduler.MultiStepLR(optimizer_D_pose1,milestones=a,gamma=0.1)
lr_scheduler_D_Id2 = torch.optim.lr_scheduler.MultiStepLR(optimizer_D_Id2,milestones=a,gamma=0.1)
lr_scheduler_D_pose2 = torch.optim.lr_scheduler.MultiStepLR(optimizer_D_pose2,milestones=a,gamma=0.1)

# lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
'''lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B,
                                                     lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)'''

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor
input_A = Tensor(30, 3, 128, 128)
input_B = Tensor(30, 3, 128, 128)
input_C = Tensor(30, 3, 128, 128)
#target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
#target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

# fake_A_buffer = ReplayBuffer()
#fake_B_buffer = ReplayBuffer()



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Image preprocessing, normalization for the pretrained resnet
'''transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))])

# Load vocabulary wrapper


# Build data loader
data_loader = get_loader(root1,root2,root3,root4,root5,transform, 30,
                         shuffle=True, num_workers=2)'''


# Dataset loader
'''transforms_ = [transforms.Resize(int(opt.size * 1.12), Image.BICUBIC),
               transforms.RandomCrop(opt.size),
               transforms.RandomHorizontalFlip(),
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True),
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

# Loss plot
logger = Logger(n_epochs, len(dataloader))'''
###################################

##vgg_loss#####
model_vgg = models.vgg19(pretrained=True)
modules_vgg = list(model_vgg.children())[:-1]
vgg_model = nn.Sequential(*modules_vgg)
for p in vgg_model.parameters():
    p.requires_grad = False

# passing vgg features of one layer to the the layer we want and so on...

'''class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = list(vgg16(pretrained=True).features)[:23]
        # features的第3，8，15，22层分别是: relu1_2,relu2_2,relu3_3,relu4_3
        self.features = nn.ModuleList(features).eval()

    def forward(self, x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {3, 8, 15, 22}:
                results.append(x)'''

'''class ResNet50Bottom(nn.Module):
    def __init__(self, original_model):
        super(ResNet50Bottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-2])

    def forward(self, x):
        x = self.features(x)
        return x


res50_model = models.resnet50(pretrained=True)
res50_conv2 = ResNet50Bottom(res50_model)

outputs = res50_conv2(inputs)
outputs.data.shape  # => torch.Size([4, 2048, 7, 7])'''

#cnn = models.vgg19(pretrained=True).features.to(device).eval()

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)


# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


# content_layers_default = ['conv_4']
vgg_layers = ['relu1_2', 'relu2_3', 'relu3_4', 'relu4_4']


class VGGNet(nn.Module):
    def __init__(self):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGGNet, self).__init__()
        self.select = ['3', '8', '17', '26']
        self.vgg = models.vgg19(pretrained=True).features.to(device)

    def forward(self, x,normalization_mean, normalization_std):
        """Extract multiple convolutional feature maps."""
        features = []
        normalization = Normalization(normalization_mean, normalization_std).to(device)
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features

cnn=VGGNet()
import torch.nn.functional as F
def vgg_loss(c,cnn, normalization_mean, normalization_std, img1, img2, img3):
    # features_var1 = vgg_model(img1) # get the output from the last hidden layer of the pretrained resnet
    vgg_features_img1 = cnn(img1, normalization_mean, normalization_std)
    vgg_features_img2 = cnn(img2, normalization_mean, normalization_std)
    vgg_features_img3 = cnn(img3, normalization_mean, normalization_std)

    w = 128
    loss_vgg=0
    for i in range(4):
        vgg_features_img1_ = vgg_features_img1[i]
        vgg_features_img2_ = vgg_features_img2[i]
        vgg_features_img3_ = vgg_features_img3[i]
        loss_vgg += F.normalize(F.normalize(
            ((F.normalize(vgg_features_img1_, dim=1, p=2) - F.normalize(vgg_features_img3_, dim=1, p=2)) // (w * w)),
            dim=1, p=2), dim=1, p=2) + F.normalize(F.normalize(
            ((F.normalize(vgg_features_img2_, dim=1, p=2) - F.normalize(vgg_features_img3_, dim=1, p=2)) // (w * w)),
            dim=1, p=2), dim=1, p=2)
        w = w // 2
        '''features1 = features_var1.data


        norm_a1 = f.normalize(features1,dim=1,p=2)
        features_var2 = vgg_model(img2) # get the output from the last hidden layer of the pretrained resnet
        features2 = features_var2.data


        norm_a2 = f.normalize(features1,dim=1,p=2)
        loss_vgg=norm_a2-norm_a1'''
        return c*(torch.sum(loss_vgg))+(1-c)*(1-(torch.sum(loss_vgg)))

'''def get_vggfeatures_layerwise(cnn, normalization_mean, normalization_std,
                              img1, img2,img3,
                              vgglayers=vgg_layers):
    #cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0
    j = 1  # increment every time we see a conv
    vgg_features_img1 = []
    vgg_features_img2 = []
    vgg_features_img3 = []
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv{}_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu{}_{}'.format(i, j)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
            model.add_module(name, layer)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
            j += 1
            i = 0
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        # model.add_module(name, layer)
        if name in vgg_layers:
            target_feature = model(img1).detach()
            vgg_features_img1.append(target_feature)
        if name in vgg_layers:
            target_feature = model(img2).detach()
            vgg_features_img2.append(target_feature)
        if name in vgg_layers:
            target_feature = model(img3).detach()
            vgg_features_img3.append(target_feature)

       
    return vgg_features_img1, vgg_features_img2, vgg_features_img3


def vgg_loss(cnn, normalization_mean, normalization_std, img1, img2,img3, vgg_layers):
    # features_var1 = vgg_model(img1) # get the output from the last hidden layer of the pretrained resnet
    get_vggfeatures_layerwise(cnn, normalization_mean, normalization_std,
                              img1, img2,img3,
                              vgglayers=vgg_layers)
    w = 128
    for i in range(4):
        vgg_features_img1 = vgg_features_img1[i].data
        vgg_features_img2 = vgg_features_img2[i].data
        vgg_features_img3 = vgg_features_img3[i].data
        loss_vgg += f.normalise(f.normalise(
            ((f.normalise(vgg_features_img1, dim=1, p=2) - f.normalise(vgg_features_img3, dim=1, p=2)) // (w * w)),
            dim=1, p=2), dim=1, p=2)+f.normalise(f.normalise(
            ((f.normalise(vgg_features_img2, dim=1, p=2) - f.normalise(vgg_features_img3, dim=1, p=2)) // (w * w)),
            dim=1, p=2), dim=1, p=2)
        w = w // 2
        
        return loss_vgg'''


##SSIM_Loss###

import pytorch_ssim
#import kornia

def ssim_loss(c,img1, img2,img3):
    ssim_loss = pytorch_ssim.SSIM(window_size = 11)
    #ssim_loss= kornia.losses.SSIM(5, reduction='none')
    #ssim_loss = pytorch_ssim.SSIM()

    return c*(1-ssim_loss(img1, img3)+1-ssim_loss(img2, img3))+(1-c)*(1-(1-ssim_loss(img1, img3)+1-ssim_loss(img2, img3)))


## attributes loss ##

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=27, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = torch.sigmoid(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs).cuda()
    #model = ResNet(block, layers)
    '''if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)'''
    return model

def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [1, 1, 1, 1], pretrained, progress, **kwargs)

mymodel=resnet101(pretrained=False, progress=True)
model_re_att = mymodel

model_re_att.load_state_dict(torch.load('/home/ws2/Downloads/paras_dec_2019/market_model/model/model.pth'))
same_B_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

def res_att_loss(c,img1, img2,img3):
    l1 = model_re_att(img1)
    l2 = model_re_att(img2)
    return c*(torch.sum((l2 - img3))+torch.sum((l1 - img3)))+(1-c)*(1-(torch.sum((l2 - img3))+torch.sum((l1 - img3))))


'''def identity_discriminator_loss(c,Did1):
    idl = c*torch.log(Did1) + (1-c)*torch.log(1 - Did1)
    return idl'''


'''def verification_loss(c, Vdv):
    vl = (-c * (torch.log(Vdv))) - ((1 - c) * (1 - torch.log(Vdv)))
    return vl'''

model_path_netG_A2B_='/home/ws2/Downloads/paras_dec_2019/Market1501_train_test/model_gan_market_2stage/model/netG_A2B'
model_path_netG_C2D='/home/ws2/Downloads/paras_dec_2019/Market1501_train_test/model_gan_market_2stage/model/netG_C2D'
model_path_netD_Id='/home/ws2/Downloads/paras_dec_2019/Market1501_train_test/model_gan_market_2stage/model/netD_Id'
model_path_netPD='/home/ws2/Downloads/paras_dec_2019/Market1501_train_test/model_gan_market_2stage/model/netPD'
def train_model(cnn,netG_A2B,netG_C2D,netD_Id1,netD_Id2,netPD1,netPD2,optimizer_G_A2B,optimizer_G_C2D,optimizer_D_Id1,optimizer_D_pose1,optimizer_D_Id2,optimizer_D_pose2
,same_pose_loss,r_l,id_l,pd_l,lr_scheduler_G_A2B,lr_scheduler_G_C2D ,lr_scheduler_D_Id1 ,lr_scheduler_D_pose1,lr_scheduler_D_Id2 ,lr_scheduler_D_pose2, data_loader,fake_B_buffer,same_B_buffer,
                       num_epochs=100):
    since = time.time()
    total_step = len(data_loader)
    # model_path='/home/ws2/Downloads/AiC/model/'


    netG_A2B.train()
    netG_C2D.train()
    netD_Id1.train()
    netD_Id2.train()
    netPD1.train()
    netPD2.train()

    for epoch in range(num_epochs):
        iteratorr=0
        for i, (images1,images2,images3,images4,images5,attribs,lengths_attrib,veris,lengths_veri) in enumerate(data_loader):
            iteratorr=iteratorr+1
            print(iteratorr)

            # Set mini-batch dataset

            images1 = images1.cuda()
            images2 = images2.cuda()
            images3 = images3.cuda()
            images4 = images4.cuda()
            images5 = images5.cuda()

            targets_attrib = attribs.cuda()
            targets_attrib = targets_attrib.float()
            targets_veri = veris[:,0].cuda()
            targets_veri = targets_veri.float()

            real_A = Variable(input_A.copy_(images2))
            real_B = Variable(input_B.copy_(images1))
            real_Pose=Variable(input_C.copy_(images5))
            optimizer_G_A2B.zero_grad()
            same_B = netG_A2B(real_B,targets_attrib,images5)

            #loss_G_A2B=resnet_attribute_loss+vgg_loss+ssim_loss+Reconstruction_loss

            #loss_G_A2B.backward()
            #optimizer_G_A2B.step()
            #lr_scheduler_G_A2B.step()
            #real_A = Variable(input_A.copy_(images1))
            #real_B = Variable(input_B.copy_(images2))
            optimizer_G_C2D.zero_grad()
            fake_B = netG_C2D(real_A, targets_attrib, images5)
            out_D_id1 = netD_Id1(fake_B, real_B)
            out_D_id2 = netD_Id2(same_B, real_B)
            loss_id_disc = id_l(1, out_D_id1, out_D_id2)
            out_D_pose1 = netPD1(real_Pose, fake_B)
            out_D_pose2 = netPD2(real_Pose, same_B)
            loss_pose_disc = pd_l(1, out_D_pose1, out_D_pose2)
            loss_G = same_pose_loss(fake_B,same_B,real_Pose)+ res_att_loss(1,fake_B,same_B,targets_attrib)+\
                     vgg_loss(1,cnn, cnn_normalization_mean, cnn_normalization_std, fake_B, same_B,real_B) \
                     + ssim_loss(1,fake_B, same_B,real_B)+r_l(1,fake_B, same_B,real_B)+loss_id_disc
            #print(loss_G.item())
            loss_G.backward()
            optimizer_G_A2B.step()
            #lr_scheduler_G_A2B.step()
            optimizer_G_C2D.step()
            #lr_scheduler_G_C2D.step()

            optimizer_D_Id1.zero_grad()
            optimizer_D_Id2.zero_grad()
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            same_B = same_B_buffer.push_and_pop(same_B)
            out_D_id1 = netD_Id1(fake_B.detach(), real_B)
            out_D_id2=netD_Id2(same_B.detach(),real_B)
            loss_id_disc=id_l(1,out_D_id1,out_D_id2)
            loss_id_disc.backward()
            optimizer_D_Id1.step()
            #lr_scheduler_D_Id1.step()
            optimizer_D_Id2.step()

            #lr_scheduler_D_Id2.step()

            optimizer_D_pose1.zero_grad()
            optimizer_D_pose2.zero_grad()
            out_D_pose1 = netPD1(real_Pose, fake_B.detach())
            out_D_pose2 = netPD2(real_Pose, same_B.detach())
            loss_pose_disc = pd_l(1, out_D_pose1,out_D_pose2)
            loss_pose_disc.backward()
            optimizer_D_pose1.step()
            #lr_scheduler_D_pose1.step()
            optimizer_D_pose2.step()
            #lr_scheduler_D_pose2.step()

            # print(

            # Print log info
            if i % 10 == 0 and i!=0:
                print('Epoch [{}/{}], Step [{}/{}], Loss_G1: {:.4f},Loss_DId_1: {:.4f},Loss_Dpose_1: {:.4f}, Perplexity_G1: {:5.4f},Perplexity_D1: {:5.4f}'
                      .format(epoch, num_epochs, i, total_step, loss_G.item(),loss_id_disc.item(),loss_pose_disc.item(), np.exp(loss_G.item())))
                time_elapsed = time.time() - since
                print('Training complete in {:.0f}m {:.0f}s'.format(
                    time_elapsed // 60, time_elapsed % 60))

            real_A = Variable(input_A.copy_(images3))
            real_B = Variable(input_B.copy_(images1))
            real_Pose = Variable(input_C.copy_(images5))
            optimizer_G_A2B.zero_grad()
            same_B = netG_A2B(real_B, targets_attrib, images5)
            # loss_G_A2B=resnet_attribute_loss+vgg_loss+ssim_loss+Reconstruction_loss

            # loss_G_A2B.backward()
            # optimizer_G_A2B.step()
            # lr_scheduler_G_A2B.step()
            # real_A = Variable(input_A.copy_(images1))
            # real_B = Variable(input_B.copy_(images2))
            optimizer_G_C2D.zero_grad()
            fake_B = netG_C2D(real_A, targets_attrib, images5)
            out_D_id1 = netD_Id1(fake_B, real_B)
            out_D_id2 = netD_Id2(same_B, real_B)
            loss_id_disc = id_l(1, out_D_id1, out_D_id2)
            out_D_pose1 = netPD1(real_Pose, fake_B)
            out_D_pose2 = netPD2(real_Pose, same_B)
            loss_pose_disc = pd_l(1, out_D_pose1, out_D_pose2)
            loss_G = same_pose_loss(fake_B, same_B, real_Pose) + res_att_loss(1,fake_B, same_B, targets_attrib) + \
                     vgg_loss(1,cnn, cnn_normalization_mean, cnn_normalization_std, fake_B, same_B, real_B) \
                     + ssim_loss(1,fake_B, same_B, real_B) + r_l(1,fake_B, same_B, real_B)+loss_id_disc
            loss_G.backward()
            optimizer_G_A2B.step()
            #lr_scheduler_G_A2B.step()
            optimizer_G_C2D.step()
            #lr_scheduler_G_C2D.step()

            optimizer_D_Id1.zero_grad()
            optimizer_D_Id2.zero_grad()
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            same_B = same_B_buffer.push_and_pop(same_B)
            out_D_id1 = netD_Id1(fake_B.detach(), real_B)
            out_D_id2 = netD_Id2(same_B.detach(), real_B)

            loss_id_disc = id_l(1, out_D_id1, out_D_id2)
            loss_id_disc.backward()
            optimizer_D_Id1.step()
            #lr_scheduler_D_Id1.step()
            optimizer_D_Id2.step()
            #lr_scheduler_D_Id2.step()

            optimizer_D_pose1.zero_grad()
            optimizer_D_pose2.zero_grad()
            out_D_pose1 = netPD1(real_Pose, fake_B.detach())
            out_D_pose2 = netPD2(real_Pose, same_B.detach())
            loss_pose_disc = pd_l(1, out_D_pose1, out_D_pose2)
            loss_pose_disc.backward()
            optimizer_D_pose1.step()
            #lr_scheduler_D_pose1.step()
            optimizer_D_pose2.step()
            #lr_scheduler_D_pose2.step()

            # print(

            # Print log info
            if i % 10 == 0 and i != 0:
                print(
                    'Epoch [{}/{}], Step [{}/{}], Loss_G1: {:.4f},Loss_DId_1: {:.4f},Loss_Dpose_1: {:.4f}, Perplexity_G1: {:5.4f},Perplexity_D1: {:5.4f}'
                    .format(epoch, num_epochs, i, total_step, loss_G.item(), loss_id_disc.item(), loss_pose_disc.item(),
                            np.exp(loss_G,item())))
                time_elapsed = time.time() - since
                print('Training complete in {:.0f}m {:.0f}s'.format(
                    time_elapsed // 60, time_elapsed % 60))

            real_A = Variable(input_A.copy_(images4))
            real_B = Variable(input_B.copy_(images1))
            real_Pose = Variable(input_C.copy_(images5))
            optimizer_G_A2B.zero_grad()
            same_B = netG_A2B(real_B, targets_attrib, images5)
            # loss_G_A2B=resnet_attribute_loss+vgg_loss+ssim_loss+Reconstruction_loss

            # loss_G_A2B.backward()
            # optimizer_G_A2B.step()
            # lr_scheduler_G_A2B.step()
            # real_A = Variable(input_A.copy_(images1))
            # real_B = Variable(input_B.copy_(images2))
            optimizer_G_C2D.zero_grad()
            fake_B = netG_C2D(real_A, targets_attrib, images5)
            out_D_id1 = netD_Id1(fake_B, real_B)
            out_D_id2 = netD_Id2(same_B, real_B)
            loss_id_disc = id_l(1, out_D_id1, out_D_id2)
            out_D_pose1 = netPD1(real_Pose, fake_B)
            out_D_pose2 = netPD2(real_Pose, same_B)
            loss_pose_disc = pd_l(1, out_D_pose1, out_D_pose2)
            loss_G = same_pose_loss(fake_B, same_B, real_Pose) + res_att_loss(0,fake_B, same_B, targets_attrib) + \
                     vgg_loss(0,cnn, cnn_normalization_mean, cnn_normalization_std, fake_B, same_B, real_B) \
                     + ssim_loss(0,fake_B, same_B, real_B) + r_l(0,fake_B, same_B, real_B)+loss_id_disc
            #print(loss_G)
            loss_G.backward()
            optimizer_G_A2B.step()
            #lr_scheduler_G_A2B.step()
            optimizer_G_C2D.step()
            #lr_scheduler_G_C2D.step()

            optimizer_D_Id1.zero_grad()
            optimizer_D_Id2.zero_grad()
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            same_B = same_B_buffer.push_and_pop(same_B)
            out_D_id1 = netD_Id1(fake_B.detach(), real_B)
            out_D_id2 = netD_Id2(same_B.detach(), real_B)
            loss_id_disc = id_l(0, out_D_id1, out_D_id2)
            loss_id_disc.backward()
            optimizer_D_Id1.step()
            #lr_scheduler_D_Id1.step()
            optimizer_D_Id2.step()
            #lr_scheduler_D_Id2.step()

            optimizer_D_pose1.zero_grad()
            optimizer_D_pose2.zero_grad()
            out_D_pose1 = netPD1(real_Pose, fake_B.detach())
            out_D_pose2 = netPD2(real_Pose, same_B.detach())
            loss_pose_disc = pd_l(0, out_D_pose1, out_D_pose2)
            loss_pose_disc.backward()
            optimizer_D_pose1.step()
            #lr_scheduler_D_pose1.step()
            optimizer_D_pose2.step()
            #lr_scheduler_D_pose2.step()

            print("loss_G",loss_G.item())
            print("loss_id_disc",loss_id_disc.item())
            print("loss_pose_disc",loss_pose_disc.item())

            # Print log info
            if i % 10 == 0 and i != 0:
                print(
                    'Epoch [{}/{}], Step [{}/{}], Loss_G1: {:.4f},Loss_DId_1: {:.4f},Loss_Dpose_1: {:.4f}, Perplexity_G1: {:5.4f},Perplexity_D1: {:5.4f}'
                    .format(epoch, num_epochs, i, total_step, loss_G.item(), loss_id_disc.item(), loss_pose_disc.item(),
                            np.exp(loss_G.item())))
                time_elapsed = time.time() - since
                print('Training complete in {:.0f}m {:.0f}s'.format(
                    time_elapsed // 60, time_elapsed % 60))
        lr_scheduler_G_A2B.step()
        lr_scheduler_G_C2D.step()
        lr_scheduler_D_Id1.step()
        lr_scheduler_D_Id2.step()
        lr_scheduler_D_pose1.step()
        lr_scheduler_D_pose2.step()





'''netG_A2B=torch.nn.DataParallel(netG_A2B.cuda())
netG_C2D=torch.nn.DataParallel(netG_C2D.cuda())
# netG_B2A.apply(weights_init_normal)
# netD_A.apply(weights_init_normal)
netD_Id=torch.nn.DataParallel(netD_Id.cuda())
netPD=torch.nn.DataParallel(netPD.cuda())'''

netG_A2B=netG_A2B.cuda()
netG_A2B=torch.nn.DataParallel(netG_A2B)
netG_C2D=netG_C2D.cuda()
netG_C2D=torch.nn.DataParallel(netG_C2D)
# netG_B2A.apply(weights_init_normal)
# netD_A.apply(weights_init_normal)
netD_Id1=netD_Id1.cuda()
netD_Id1=torch.nn.DataParallel(netD_Id1)
netD_Id2=netD_Id2.cuda()
netD_Id2=torch.nn.DataParallel(netD_Id2)
netPD1=netPD1.cuda()
netPD1=torch.nn.DataParallel(netPD1)
netPD2=netPD2.cuda()
netPD2=torch.nn.DataParallel(netPD2)

same_pose_loss=Same_Pose_loss().cuda()
r_l=Reconstruction_loss().cuda()
id_l=Identity_Discriminator_loss().cuda()
pd_l=Pose_Discriminator_loss().cuda()

from torch.optim import lr_scheduler

#exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
# with torch.enable_grad():
model_ft = train_model(cnn,netG_A2B,netG_C2D,netD_Id1,netD_Id2,netPD1,netPD2,optimizer_G_A2B,optimizer_G_C2D ,optimizer_D_Id1,optimizer_D_pose1,optimizer_D_Id2,optimizer_D_pose2
,same_pose_loss,r_l,id_l,pd_l,lr_scheduler_G_A2B,lr_scheduler_G_C2D ,lr_scheduler_D_Id1 ,lr_scheduler_D_pose1,lr_scheduler_D_Id2 ,lr_scheduler_D_pose2, data_loader,fake_B_buffer,same_B_buffer,
                       num_epochs=100)

if not os.path.exists(model_path_netG_A2B):
    os.makedirs(model_path_netG_A2B)
if not os.path.exists(model_path_netG_C2D):
    os.makedirs(model_path_netG_C2D)
if not os.path.exists(model_path_netD_Id):
    os.makedirs(model_path_netD_Id)
if not os.path.exists(model_path_netPD):
    os.makedirs(model_path_netPD)
torch.save(netG_A2B.module.state_dict(), os.path.join(model_path_netG_A2B, 'model.pth'))
torch.save(netG_C2D.module.state_dict(), os.path.join(model_path_netG_C2D, 'model.pth'))
torch.save(netD_Id.module.state_dict(), os.path.join(model_path_netD_Id, 'model.pth'))
torch.save(netPD.module.state_dict(), os.path.join(model_path_netPD, 'model.pth'))

###### Training ######

