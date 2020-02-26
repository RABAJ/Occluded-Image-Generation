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

        return x15




'''encoder1=Encoder1()
encoder1=encoder1.cuda()
encoder1=torch.nn.DataParallel(encoder1)
encoder2=Encoder2()
encoder2=encoder2.cuda()
encoder2=torch.nn.DataParallel(encoder2)'''
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
'''def verification_classifier(encoder1,encoder2,x,y):
    v1=Encoder1()(x)
    v2=Encoder2()(y)
    #s=p2-p1
    s=torch.sub(v2, v1)
    po=torch.pow(s,2)
    pa=torch.nn.BatchNorm2d(512)(po)
    fc = nn.Linear(512 * block.expansion, 1)(pa)
    f=torch.sigmoid(fc)
    return f'''


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

'''def verification_loss(c,Vdv):
    print("Vdv",Vdv.size())
    vl=(-c*(torch.log(Vdv)))-((1-c)*(1-torch.log(Vdv)))
    #print(vl)
    print("vl",vl.size())
    s=torch.Tensor(248)

    vl=torch.div(torch.sum(vl),s)
    return vl'''


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

root1='/home/ws2/Downloads/Market1501_train_test/nonocc_train_v1/'
root2='/home/ws2/Downloads/Market1501_train_test/crops_occ_train/'
root3='/home/ws2/Downloads/Market1501_train_test/crops_occ_diffpose_train/'
root4='/home/ws2/Downloads/Market1501_train_test/crops_occ_diffiden_train/'




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
        a = loadmat('/home/ws2/Downloads/market_attributes/dataset.mat')
        b = loadmat('/home/ws2/Downloads/market_attributes/image_index.mat')
        ''''for key,values in a.items() :
            print (key)
            print("value")
            #print(values)'''
        #print(a)
        import numpy

        #self.attri = numpy.asarray(a['dataset'])
        #self.ids=numpy.asarray(b['image_index'][])
        #self.ids=numpy.load('/home/ws2/Downloads/Market1501_train_test/image_index.npy')
        self.ids=numpy.load('/home/ws2/Downloads/Market1501_train_test/market_v1_iden.npy')
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

        #attrib = attri[ann_id]
        '''attrib=attri[index]
        attrib=attrib-1
        if attrib[0]==0 or attrib[0]==1:
            attrib[0]=0
        elif attrib[0]==2 or attrib[0]==3:
            attrib[0]=1'''

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
        return image1,image2,image3,image4,target_veri


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
    images1,images2,images3,images4,veri = zip(*data)
    #print(images1.size())

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images1 = torch.stack(images1, 0)
    images2 = torch.stack(images2, 0)
    images3 = torch.stack(images3, 0)
    images4 = torch.stack(images4, 0)

    # Merge attributes (from tuple of 1D tensor to 2D tensor).
    #lengths_attrib = [len(cap) for cap in attribs]
    #targets_attrib = torch.zeros(len(attribs), max(lengths_attrib)).long()
    lengths_veri = [len(cap) for cap in veri]
    targets_veri = torch.zeros(len(veri), max(lengths_veri)).long()
    '''for i, cap in enumerate(attribs):
        end = lengths_attrib[i]
        targets_attrib[i, :end] = cap[:end]'''
    for i, cap in enumerate(veri):
        end = lengths_veri[i]
        targets_veri[i, :end] = cap[:end]
    return images1,images2,images3,images4,targets_veri,lengths_veri


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
#criterion = torch.nn.BCELoss()
criterion=verification_loss()
#mymodel = resnet101(pretrained=False, progress=True)
mymodel=Verification_Classifier(encoder1,encoder2)
#optimizer = torch.optim.Adam(mymodel.parameters(), lr=0.1)
optimizer=torch.optim.SGD(mymodel.parameters(), lr=0.1, momentum=0.9)
# optimizer = torch.optim.Adam(myModel().parameters(), lr=0.0002)

# Train the models

model_path = '/home/ws2/Downloads//Market1501_train_test/model/'


def train_model(model11, optimizer,criterion, scheduler, data_loader, num_epochs=80):
    since = time.time()
    total_step = len(data_loader)
    # model_path='/home/ws2/Downloads/AiC/model/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    model11.train()

    for epoch in range(num_epochs):
        for i, (images1,images2,images3,images4,veris,lengths_veri) in enumerate(data_loader):

            # Set mini-batch dataset

            images1 = images1.cuda()
            images2 = images2.cuda()
            images3 = images3.cuda()
            images4 = images4.cuda()
            # print(images.size())
            # summary(model11,(3,224,224))
            #targets_attrib = attributes.cuda()
            #targets_attrib = targets_attrib.float()
            targets_veri = veris[:,0].cuda()
            targets_veri = targets_veri.float()
            # print('targets')
            # print(targets)
            # targets=torch.argmax(targets)
            # targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            # Forward, backward and optimize
            # features = encoder(images)
            # outputs = decoder(features, captions, lengths)
            optimizer.zero_grad()
            outputs = model11(images2,images1)
            # print('outputs')
            # print(outputs)
            # print('targets')
            # print(targets)
            #loss=verification_loss(1,outputs)
            #loss = criterion(outputs, targets_veri)
            #print('outputs',type(outputs))
            loss=criterion(1,outputs).cuda()
            #loss.requires_grad=True
            #print(loss)
            loss.backward()
            optimizer.step()
            scheduler.step()
            # print(

            # Print log info
            if i % 430 == 0 and i!=0:
                print('Epoch [{}/{}], Step [{}/{}], Loss_same: {:.4f}, Perplexity_same: {:5.4f}'
                      .format(epoch, num_epochs, i, total_step, loss.item(), np.exp(loss.item())))
                time_elapsed = time.time() - since
                print('Training complete in {:.0f}m {:.0f}s'.format(
                    time_elapsed // 60, time_elapsed % 60))
                targets_veri = veris[:, 0].cuda()
                targets_veri = targets_veri.float()
                # print('targets')
                # print(targets)
                # targets=torch.argmax(targets)
                # targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

                # Forward, backward and optimize
                # features = encoder(images)
                # outputs = decoder(features, captions, lengths)
                optimizer.zero_grad()
                outputs = model11(images3, images1)
                # print('outputs')
                # print(outputs)
                # print('targets')
                # print(targets)
                # loss = verification_loss(0, outputs)
                # print("outputs",outputs)
                loss = criterion(1, outputs)
                # loss = criterion(outputs, targets_veri)
                # loss.requires_grad=True

                loss.backward()
                optimizer.step()
                scheduler.step()
                # print(

                # Print log info
                if i % 430 == 0 and i != 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss_diff: {:.4f}, Perplexity_diff: {:5.4f}'
                          .format(epoch, num_epochs, i, total_step, loss.item(), np.exp(loss.item())))
                    time_elapsed = time.time() - since
                    print('Training complete in {:.0f}m {:.0f}s'.format(
                        time_elapsed // 60, time_elapsed % 60))
            targets_veri = veris[:, 1].cuda()
            targets_veri = targets_veri.float()
            # print('targets')
            # print(targets)
            # targets=torch.argmax(targets)
            # targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            # Forward, backward and optimize
            # features = encoder(images)
            # outputs = decoder(features, captions, lengths)
            optimizer.zero_grad()
            outputs = model11(images4, images1)
            # print('outputs')
            # print(outputs)
            # print('targets')
            # print(targets)
            #loss = verification_loss(0, outputs)
            #print("outputs",outputs)
            loss = criterion(0, outputs)
            #loss = criterion(outputs, targets_veri)
            # loss.requires_grad=True

            loss.backward()
            optimizer.step()
            scheduler.step()
            # print(

            # Print log info
            if i % 430 == 0 and i!=0:
                print('Epoch [{}/{}], Step [{}/{}], Loss_diff: {:.4f}, Perplexity_diff: {:5.4f}'
                      .format(epoch, num_epochs, i, total_step, loss.item(), np.exp(loss.item())))
                time_elapsed = time.time() - since
                print('Training complete in {:.0f}m {:.0f}s'.format(
                    time_elapsed // 60, time_elapsed % 60))
            # Save the model checkpoints


# mymodel=myModel()
# mymodel=mymodel.to(device)
'''try:
    state_dict = model.module.state_dict()
except AttributeError:
    state_dict = model.state_dict()'''
mymodel = mymodel.cuda()

mymodel = torch.nn.DataParallel(mymodel)
print(mymodel)
from torch.optim import lr_scheduler

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
# with torch.enable_grad():
model_ft = train_model(mymodel, optimizer,criterion, exp_lr_scheduler, data_loader,
                       num_epochs=1)
torch.save(mymodel.module.state_dict(), os.path.join(model_path, 'model.pth'))