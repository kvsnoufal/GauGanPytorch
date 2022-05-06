
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
from glob import glob

import torch
from torch import nn
from torch.utils import data
import torchvision
import torch.nn.functional as F
from  torch.nn.utils import spectral_norm
from torchvision import models
import torchvision.transforms as transforms
from config import Config as cfg

class Gen_loss(nn.Module):
    def __init__(self):
        super(Gen_loss,self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()
    def forward(self,pred):
        return self.criterion(pred,torch.ones_like(pred))
class KLD_Loss(nn.Module):
    def __init__(self):
        super(KLD_Loss,self).__init__()
    def forward(self,mu,logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss,self).__init__()
        vgg = models.vgg19(pretrained=True).to(cfg.DEVICE).features
        print(vgg)
        for param in vgg.parameters():
            param.requires_grad = False
        self.f1 = nn.Sequential(*[vgg[x] for x in range(2)])
        self.f2 = nn.Sequential(*[vgg[x] for x in range(7)])
        self.f3 = nn.Sequential(*[vgg[x] for x in range(12)])
        self.f4 = nn.Sequential(*[vgg[x] for x in range(21)])
        self.f5 = nn.Sequential(*[vgg[x] for x in range(30)])
    def forward(self,x,y):
        loss=0
        x1 = self.f1(x)
        y1 = self.f1(y)
        loss1 = F.l1_loss(x1,y1)

        x2 = self.f2(x)
        y2 = self.f2(y)
        loss2 = F.l1_loss(x2,y2)

        x3 = self.f3(x)
        y3 = self.f3(y)
        loss3 = F.l1_loss(x3,y3)

        x4 = self.f4(x)
        y4 = self.f4(y)
        loss4 = F.l1_loss(x4,y4)

        x5 = self.f5(x)
        y5 = self.f5(y)
        loss5 = F.l1_loss(x5,y5)

        loss += loss1/32 + loss2/16 + loss3/8 + loss4/4 + loss5
        return loss
class FeatureLossDisc(nn.Module):
    def __init__(self):
        super(FeatureLossDisc,self).__init__()
    def forward(self,real_disc_outputs,fake_disc_outputs):
        loss=0
        for real_disc_output,fake_disc_output in zip(real_disc_outputs,fake_disc_outputs):
            loss+= F.l1_loss(real_disc_output,fake_disc_output)
        return loss
class Disc_HingeLoss(nn.Module):
    def __init__(self):
        super(Disc_HingeLoss,self).__init__()
        self.hingleLoss = nn.HingeEmbeddingLoss()
    def forward(self,x,real=True):
        if real:
            return self.hingleLoss(x,torch.ones_like(x))
        else:
            return self.hingleLoss(x,torch.ones_like(x)*-1)             
