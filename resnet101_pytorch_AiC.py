#!/usr/bin/env python
# coding: utf-8

# https://github.com/mortezamg63/Accessing-and-modifying-different-layers-of-a-pretrained-model-in-pytorch
# for adding manipulating pretrained model layers and custom loss and transformations

# In[ ]:


from __future__ import division
'''
from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate

from terminaltables import AsciiTable
'''
import os
import sys
import time
import datetime
import argparse

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
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
'''
class myModel(torch.nn.Module):
    def __init__(self, num_classes = 2):
        super(myModel,self).__init__()
        original_model = models.densenet121()
        self.features = torch.nn.Sequential(*list(original_model.children())[:-1])
        self.classifier = (torch.nn.Linear(1024, 24))

    def forward(self, x):
        f = self.features(x)
        f = F.relu(f, inplace=True)
        f = F.avg_pool2d(f, kernel_size=7).view(f.size(0), -1)
        y = self.classifier(f)
        out18 = torch.sigmoid(y)
        return out18

'''
#resnet101 implementation from scratch
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
    model = ResNet(block, layers, **kwargs)
    #model = ResNet(block, layers)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [1, 1, 1, 1], pretrained, progress, **kwargs)






'''

class myModel(torch.nn.Module):
    def __init__(self):
        super(myModel,self).__init__()
        #model = Darknet('/home/ws2/Downloads/PyTorch-YOLOv3-master/yolov3.cfg').to(device)
        #model.apply(weights_init_normal)

        #model.load_darknet_weights('/home/ws2/Downloads/PyTorch-YOLOv3-master/darknet53.conv.74')
        self.model = models.densenet121()
        #for param in self.resnet101.parameters():
          #  param.requires_grad = False
        num_ftrs = self.model.fc.in_features
        
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, self.model.fc.in_features//2)
        #self.layer1=torch.nn.Linear(self.model.fc.in_features, self.model.fc.in_features)
        #self.layer2=torch.nn.Linear(self.model.fc.in_features, self.model.fc.in_features//2)
        #self.layer3=torch.nn.Linear(self.model.fc.in_features//2, self.model.fc.in_features//2)
        self.layer4=torch.nn.Linear(self.model.fc.in_features//2, self.model.fc.in_features//4)
        self.layer5=torch.nn.Linear(self.model.fc.in_features//4, 24)
    def forward(self,x):
        #print(self.model.fc.in_features)
        out1 = self.model(x)
        out2=torch.nn.BatchNorm1d(num_features=self.model.fc.in_features//2).cuda()
        out3=out2(out1)
        out4 = F.relu(out3)
        out5=self.layer4(out4)
        out6=torch.nn.BatchNorm1d(num_features=self.model.fc.in_features//4).cuda()
        out7=out6(out5)
        out8 = F.relu(out7)
        out9=F.dropout(out8,0.4)
        out10=self.layer5(out9)
        out11=torch.nn.BatchNorm1d(num_features=24).cuda()
        out12=out11(out10)




        
        out2=torch.nn.BatchNorm1d(num_features=self.model.fc.in_features).cuda()
        out2=out2(out1)
        out3 = F.relu(out2)
        #print(out2)
        #out3=F.softmax(out2)
        #print(out3)
        out4=self.layer1(out3)

        out5=torch.nn.BatchNorm1d(num_features=self.model.fc.in_features).cuda()
        out5=out5(out4)
        out6=F.relu(out5)
        out7=self.layer2(out6)

        out8=torch.nn.BatchNorm1d(num_features=self.model.fc.in_features//2).cuda()
        out8=out8(out7)
        out9=F.relu(out8)
        out10=self.layer3(out9)

        out11=torch.nn.BatchNorm1d(num_features=self.model.fc.in_features//2).cuda()
        out11=out11(out10)
        out12=F.relu(out11)
        out13=self.layer4(out12)

        out14=torch.nn.BatchNorm1d(num_features=self.model.fc.in_features//4).cuda()
        out14=out14(out13)
        out15=F.relu(out14)
        out16=self.layer5(out15)

        out17=torch.nn.BatchNorm1d(num_features=24).cuda()
        out17=out17(out16)
        out18=torch.sigmoid(out12)
        #t = torch.cuda.FloatTensor([0.5])
        #out = (out2 > t).float()
        return out18
        '''
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
from PIL import Image
import json


root='/home/ws2/Downloads/paras_dec_2019/occluded_person_reidentification/AiC/crops/'
json1='/home/ws2/Downloads/paras_dec_2019/occluded_person_reidentification/AiC/annotations.json'
'''class AiCDataset(data.Dataset):
    """AiC Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, json1, transform=None):
        """Set the path for images and attributes.
        
        Args:
            root: image directory.
            json1: aic annotation file path.
            
            transform: image transformer.
        """
        self.root = root
        with open(json1) as f:
            annots = json.load(f)
        
        self.attri=[x['attributes'] for x in annots]
        self.ids=[ x for x in range(len(self.attri))]
        
        
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image and attribute)."""
        attri=self.attri
        
        ann_id = self.ids[index]
        
        attrib=attri[ann_id]
        
        img_id=self.ids[ann_id]
        
        path=str(img_id)+'.jpg'

        image = Image.open(os.path.join(self.root, path))
        if self.transform is not None:
            image = self.transform(image)

        target = torch.Tensor(attrib)
        return image, target

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
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, attribs = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge attributes (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in attribs]
    targets = torch.zeros(len(attribs), max(lengths)).long()
    for i, cap in enumerate(attribs):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths

def get_loader(root, json1, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom aic dataset."""
    
    aic = AiCDataset(root=root,
                       json1=json1,
                       transform=transform)
    
    # Data loader for AiC dataset
    # This will return (images, attributes, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # attributes: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each attribute. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=aic, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Image preprocessing, normalization for the pretrained resnet
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(), 
    transforms.ToTensor(), 
    transforms.Normalize((0.485, 0.456, 0.406), 
                         (0.229, 0.224, 0.225))])

# Load vocabulary wrapper


# Build data loader
data_loader = get_loader(root, json1,  
                         transform, 300,
                         shuffle=True, num_workers=2) '''


root1='/home/ws2/Downloads/paras_dec_2019/Market1501_train_test/nonocc_train_v1/'
root2='/home/ws2/Downloads/paras_dec_2019/Market1501_train_test/crops_occ_train/'
root3='/home/ws2/Downloads/paras_dec_2019/Market1501_train_test/crops_occ_diffpose_train/'
root4='/home/ws2/Downloads/paras_dec_2019/Market1501_train_test/crops_occ_diffiden_train/'

class AiCDataset(data.Dataset):
    """AiC Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root1,root2,root3,root4, transform=None):
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

        from scipy.io import loadmat
        a = loadmat('/home/ws2/Downloads/paras_dec_2019/market_attributes/dataset.mat')
        b = loadmat('/home/ws2/Downloads/paras_dec_2019/market_attributes/image_index.mat')
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

        if self.transform is not None:
            #print(self.transform)
            image1 = self.transform(image1)
            image2 = self.transform(image2)
            image3 = self.transform(image3)
            image4 = self.transform(image4)


        #target_attrib = torch.Tensor(attrib)
        target_veri=torch.Tensor([1,0])
        return image1,image2,image3,image4,attrib


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
    images1,images2,images3,images4,attribs = zip(*data)
    #print(images1.size())
    attribs=torch.from_numpy(np.asarray(attribs)).float()
    # Merge images (from tuple of 3D tensor to 4D tensor).
    images1 = torch.stack(images1, 0)
    images2 = torch.stack(images2, 0)
    images3 = torch.stack(images3, 0)
    images4 = torch.stack(images4, 0)


    # Merge attributes (from tuple of 1D tensor to 2D tensor).
    lengths_attrib = [len(cap) for cap in attribs]
    targets_attrib = torch.zeros(len(attribs), max(lengths_attrib)).long()

    for i, cap in enumerate(attribs):
        end = lengths_attrib[i]
        targets_attrib[i, :end] = cap[:end]

    return images1,images2,images3,images4,targets_attrib,lengths_attrib


def get_loader(root1,root2,root3,root4, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom aic dataset."""

    aic = AiCDataset(root1,root2,root3,root4,
                     transform=transform)

    # Data loader for AiC dataset
    # This will return (images, attributes, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # attributes: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each attribute. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=aic,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Image preprocessing, normalization for the pretrained resnet
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))])

# Load vocabulary wrapper


# Build data loader
data_loader = get_loader(root1,root2,root3,root4,transform, 30,
                         shuffle=True, num_workers=2)

# Build the models

# Loss and optimizer
criterion = torch.nn.BCELoss()
mymodel=resnet101(pretrained=False, progress=True)
optimizer = torch.optim.Adam(mymodel.parameters(), lr=0.0002)
#optimizer = torch.optim.Adam(myModel().parameters(), lr=0.0002)

# Train the models

model_path='/home/ws2/Downloads/market_model/model/'
#model_path='/home/ws2/Downloads/paras_dec_2019/AiC/model/model.pth'
def train_model(model11, criterion, optimizer,scheduler, data_loader, num_epochs=100):
    since=time.time()
    total_step = len(data_loader) 
    #model_path='/home/ws2/Downloads/AiC/model/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)	
    scheduler.step()
    model11.train()
    
    for epoch in range(num_epochs):
        for i, (images1,images2,images3,images4, attributes, lengths) in enumerate(data_loader):

            # Set mini-batch dataset

            images1 = images1.cuda()
            images2=images2.cuda()
            #print(images.size())
            #summary(model11,(3,224,224))
            targets = attributes.cuda()
            targets=targets.float()
            #print(targets.size())
            #print('targets')
            #print(targets)
            #targets=torch.argmax(targets)
            #targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            # Forward, backward and optimize
            #features = encoder(images)
            #outputs = decoder(features, captions, lengths)
            optimizer.zero_grad()
            outputs=model11(images1)
            #print(outputs.size())
            #print('outputs')
            #print(outputs)
            #print('targets')
            #print(targets)
            loss = criterion(outputs, targets)
            #loss.requires_grad=True

            

            loss.backward()
            optimizer.step()
            #print(

            # Print log info
            if i % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, num_epochs, i, total_step, loss.item(), np.exp(loss.item()))) 
                time_elapsed = time.time() - since
                print('Training complete in {:.0f}m {:.0f}s'.format(
                    time_elapsed // 60, time_elapsed % 60))
            optimizer.zero_grad()
            outputs = model11(images2)
            # print('outputs')
            # print(outputs)
            # print('targets')
            # print(targets)
            loss = criterion(outputs, targets)
            # loss.requires_grad=True

            loss.backward()
            optimizer.step()
            # print(

            # Print log info
            if i % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, num_epochs, i, total_step, loss.item(), np.exp(loss.item())))
                time_elapsed = time.time() - since
                print('Training complete in {:.0f}m {:.0f}s'.format(
                    time_elapsed // 60, time_elapsed % 60))
            # Save the model checkpoints
#mymodel=myModel()
#mymodel=mymodel.to(device)
mymodel=mymodel.cuda()

#mymodel=torch.nn.DataParallel(mymodel)
print(mymodel)
from torch.optim import lr_scheduler
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
#with torch.enable_grad():
model_ft = train_model(mymodel, criterion, optimizer,exp_lr_scheduler,data_loader,
                       num_epochs=200)
torch.save(mymodel.state_dict(), os.path.join(model_path, 'model.pth'))

