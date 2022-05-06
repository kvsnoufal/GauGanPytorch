
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

def init_weights_normal(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal(m.weight)
def init_weights_uniform(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)     

class SPADE(nn.Module):
    def __init__(self,filters,epsilon=1e-5):
        super(SPADE,self).__init__()
        self.epsilon = epsilon
        self.conv = nn.Sequential(nn.Conv2d(cfg.NUM_CLASSES,128,kernel_size=3,padding="same",bias=False),\
            nn.ReLU())
        self.conv_gamma = nn.Conv2d(128,filters,kernel_size=3,padding="same",bias=False)
        self.conv_beta = nn.Conv2d(128,filters,kernel_size=3,padding="same",bias=False)
        self.bn = nn.BatchNorm2d(filters,affine=False)
        self.conv.apply(init_weights_uniform)
        self.conv_gamma.apply(init_weights_uniform)
        self.conv_beta.apply(init_weights_uniform)

    def forward(self,input,segmentation_map):
        mask = F.interpolate(segmentation_map, size=input.size()[2:], mode='nearest')        
        # print(mask.shape)
        x = self.conv(mask)
        gamma = self.conv_gamma(x)
        beta = self.conv_beta(x)
        normalized = self.bn(input)
        # print(gamma.shape,normalized.shape,beta.shape,input.shape)
        output = gamma* normalized + beta
        return output

class ResBlock(nn.Module):
    def __init__(self,num_features_in,num_features_out):
        super(ResBlock,self).__init__()
        self.spade_1 = SPADE(num_features_in)
        self.spade_2 = SPADE(num_features_out)
        self.conv_1 = nn.Conv2d(num_features_in,num_features_out,3,padding="same",bias=False)
        self.conv_2 = nn.Conv2d(num_features_out,num_features_out,3,padding="same",bias=False)

        if num_features_in==num_features_out:
            self.has_conv_skip = False
        else:
            self.has_conv_skip = True
            self.spade_skip = SPADE(num_features_in)
            self.conv_skip = nn.Conv2d(num_features_in,num_features_out,3,padding="same",bias=False)
            self.conv_skip.apply(init_weights_uniform)
        self.conv_1.apply(init_weights_uniform)
        self.conv_2.apply(init_weights_uniform)
        


    def forward(self,input,segmentation_map):
        x = self.spade_1(input,segmentation_map)
        x = self.conv_1(F.leaky_relu(x,0.2))
        x = self.spade_2(x,segmentation_map)
        x = self.conv_2(F.leaky_relu(x,0.2))
        if self.has_conv_skip==False:
            skip = input
        else:
            skip = self.conv_skip(F.leaky_relu(self.spade_skip(input,segmentation_map),0.2))
        output = skip + x
        return output            

def downsample(in_channels,out_channels,kernels,strides=2,apply_norm=True,apply_activation=True,\
    apply_dropout=False):
    block = [nn.Conv2d(in_channels,out_channels,kernels,stride=strides,\
        padding=1,bias=False)]  
    if apply_norm:
        block.append(nn.InstanceNorm2d(out_channels))
    if apply_activation:
        block.append(nn.LeakyReLU(0.2))
    if apply_dropout:
        block.append(nn.Dropout(0.5))
    block = nn.Sequential(*block)
       
    block.apply(init_weights_normal)
    return block
class Encoder(nn.Module):
    def __init__(self,):
        super(Encoder,self).__init__()
        self.layer1 = downsample(3,64,3,apply_norm=False)
        self.layer2 = downsample(64,64*2,3)
        self.layer3 = downsample(64*2,64*4,3)
        self.layer4 = downsample(64*4,64*8,3)
        self.layer5 = downsample(64*8,64*8,3)
        self.mu = nn.Linear(32768,cfg.LATENT_DIM)
        self.var = nn.Linear(32768,cfg.LATENT_DIM)
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = nn.Flatten()(x)
        mu = self.mu(x)
        var = self.var(x)
        return mu,var
    def get_latent_vector(self,mu,var):
        epsilon = torch.randn(mu.size(),device=cfg.DEVICE)
        latent_vec = mu  + torch.exp((var*0.5)) * epsilon  
        return latent_vec

# enc =  Encoder()
# inp = torch.randn(1,3,256,256)
# op = enc(inp)
# print(op[0].shape,op[1].shape)


class Generator(nn.Module):
    def __init__(self,):
        super(Generator,self).__init__()
        self.linear1 = nn.Linear(cfg.LATENT_DIM,16384)

        self.block1 = ResBlock(num_features_in=1024,num_features_out=1024)
        self.block2 = ResBlock(num_features_in=1024,num_features_out=1024)
        self.block3 = ResBlock(num_features_in=1024,num_features_out=1024)
        self.block4 = ResBlock(num_features_in=1024,num_features_out=512)
        self.block5 = ResBlock(num_features_in=512,num_features_out=256)
        self.block6 = ResBlock(num_features_in=256,num_features_out=128)
        self.conv_out = nn.Conv2d(in_channels=128,out_channels=3,kernel_size=4,padding="same",bias=False)
        self.linear1.apply(init_weights_uniform)
        self.conv_out.apply(init_weights_uniform) 

    def forward(self,latent_vec,segmentation_map):
        x = self.linear1(latent_vec)
        x = x.reshape(-1,1024,4,4)
        x = self.block1(x,segmentation_map)
        x = F.interpolate(x,scale_factor=2)
        x = self.block2(x,segmentation_map)
        x = F.interpolate(x,scale_factor=2)
        x = self.block3(x,segmentation_map)
        x = F.interpolate(x,scale_factor=2)
        x = self.block4(x,segmentation_map)
        x = F.interpolate(x,scale_factor=2)
        x = self.block5(x,segmentation_map)
        x = F.interpolate(x,scale_factor=2)
        x = self.block6(x,segmentation_map)
        x = F.interpolate(x,scale_factor=2)
        x = F.leaky_relu(x,0.2)
        output_image = F.tanh(self.conv_out(x))
        return output_image

# gen = Generator()
# lvec = torch.randn(1,256)
# seg = torch.randn(1,12,256,256)
# op = gen(lvec,seg)        
# print(op.shape)


class Discriminator(nn.Module):
    def __init__(self,):
        super(Discriminator,self).__init__()
        self.layer1 = downsample(cfg.NUM_CLASSES+3,64,4,apply_norm=False)
        self.layer2 = downsample(64,64*2,4)
        self.layer3 = downsample(64*2,64*4,4)
        self.layer4 = downsample(64*4,64*8,4,strides=1)
        self.layer5 = nn.Conv2d(64*8,1,4,bias=False)
        self.layer5.apply(init_weights_uniform)
    def forward(self,image,segmentation_map):
        concat_img = torch.concat([image,segmentation_map],dim=1)
        x1 = self.layer1(concat_img)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5  =self.layer5(x4)
        outputs = [x1,x2,x3,x4,x5]
        return outputs

# disc = Discriminator()
# img = torch.randn(1,3,256,256)        
# mask = torch.randn(1,12,256,256)        
# ops = disc(img,mask)
# print([t.shape for t in ops])

