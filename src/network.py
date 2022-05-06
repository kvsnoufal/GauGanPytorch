
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

class EncoderBlock(nn.Module):
    def __init__(self,channels_in,channels_out,with_norm=True):
        super(EncoderBlock,self).__init__()
        if with_norm:
            self.block = nn.Sequential(
                                        nn.Conv2d(in_channels=channels_in,out_channels=channels_out,\
                                                    kernel_size=3,stride=2,bias=False,padding=1),           
                                        nn.InstanceNorm2d(channels_out),
                                        nn.LeakyReLU(0.2)
                                        )
        else:
            self.block = nn.Sequential(
                                        nn.Conv2d(in_channels=channels_in,out_channels=channels_out,\
                                                    kernel_size=3,stride=2,bias=False,padding=1),
                                        nn.LeakyReLU(0.2)
                                        )
    def forward(self,x):
        return self.block(x)
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.block1 = EncoderBlock(3,64,with_norm=False)
        self.block2 = EncoderBlock(64,128)
        self.block3 = EncoderBlock(128,256)
        self.block4 = EncoderBlock(256,512)
        self.block5 = EncoderBlock(512,512)
        self.block6 = EncoderBlock(512,512)
        # self.block7 = EncoderBlock(512,512)
        self.flattening_block = nn.Conv2d(512,8192,kernel_size=1,padding=0)

        self.linear_mu_branch = nn.Linear(in_features=8192,out_features=cfg.LATENT_DIM)
        self.linear_var_branch = nn.Linear(in_features=8192,out_features=cfg.LATENT_DIM)
        

    def forward(self,x):
        x = self.block1(x)
        # print(x.shape)
        x = self.block2(x)
        # print(x.shape)
        x = self.block3(x)
        # print(x.shape)
        x = self.block4(x)
        # print(x.shape)
        x = self.block5(x)
        # print(x.shape)
        x = self.block6(x)
        # print(x.shape)
        # x = self.block7(x)
        # print(x.shape)
        x = x.reshape(x.shape[0],x.shape[1]*x.shape[2]*x.shape[3])
        # print(x.shape)
        
        
        mu = self.linear_mu_branch(x)
        var = self.linear_var_branch(x)
        
        # print(mu.shape,var.shape)
        return mu,var
        # return x
    def get_latent_vector(self,mu,var):
        epsilon = torch.randn(mu.size(),device=cfg.DEVICE)
        latent_vec = mu  + torch.exp((var*0.5)) * epsilon  
        return latent_vec
class EncoderLarge(nn.Module):
    def __init__(self):
        super(EncoderLarge,self).__init__()
        self.block1 = EncoderBlock(3,64,with_norm=False)
        self.block2 = EncoderBlock(64,128)
        self.block3 = EncoderBlock(128,256)
        self.block4 = EncoderBlock(256,512)
        self.block5 = EncoderBlock(512,512)
        self.block6 = EncoderBlock(512,512)
        self.block7 = EncoderBlock(512,512)
        self.flattening_block = nn.Conv2d(512,8192,kernel_size=1,padding=0)

        self.linear_mu_branch = nn.Linear(in_features=2048,out_features=cfg.LATENT_DIM)
        self.linear_var_branch = nn.Linear(in_features=2048,out_features=cfg.LATENT_DIM)
        

    def forward(self,x):
        x = self.block1(x)
        # print(x.shape)
        x = self.block2(x)
        # print(x.shape)
        x = self.block3(x)
        # print(x.shape)
        x = self.block4(x)
        # print(x.shape)
        x = self.block5(x)
        # print(x.shape)
        x = self.block6(x)
        # print(x.shape)
        x = self.block7(x)
        # print(x.shape)
        x = x.reshape(x.shape[0],x.shape[1]*x.shape[2]*x.shape[3])
        # print(x.shape)
        
        
        mu = self.linear_mu_branch(x)
        var = self.linear_var_branch(x)
        
        # print(mu.shape,var.shape)
        return mu,var
        # return x
    def get_latent_vector(self,mu,var):
        epsilon = torch.randn(mu.size(),device=cfg.DEVICE)
        latent_vec = mu  + torch.exp((var*0.5)) * epsilon  
        return latent_vec

class SPADE(nn.Module):
    def __init__(self,num_channels):
        super(SPADE, self).__init__()
        self.bn = nn.BatchNorm2d(num_channels,affine=False)
        self.conv_1 = nn.Sequential(spectral_norm(nn.Conv2d(cfg.NUM_CLASSES,128,kernel_size=3,padding=1)),\
                                   nn.ReLU())
        self.conv_1_1  = spectral_norm(nn.Conv2d(128, num_channels, kernel_size=3, padding=1))
        self.conv_2 = spectral_norm(nn.Conv2d(128,  num_channels, kernel_size=3, padding=1))
        
    def forward(self,x,segmentation_map):
        # print(x.shape)
        # BN
        x = self.bn(x)
        # Resize Map
        segmentation_map = F.interpolate(segmentation_map, size=x.size()[2:], mode='nearest')
        # Calc gamma and beta 
        output_shared = self.conv_1(segmentation_map)
        gamma = self.conv_1_1(output_shared)
        beta = self.conv_2(output_shared)
        # rescale
        # print(x.shape,gamma.shape,beta.shape)
        out = (x*gamma) + beta
        return out
class SPADEResBlk(nn.Module):
    def __init__(self,num_features_in,num_features_out):
        super(SPADEResBlk,self,).__init__()
        self.spade1 = SPADE(num_channels=num_features_in)
        self.conv1 = spectral_norm(nn.Conv2d(in_channels=num_features_in,\
            out_channels=num_features_out,kernel_size=3,padding=1))
        self.spade2 = SPADE(num_channels=num_features_out)
        self.conv2 = spectral_norm(nn.Conv2d(in_channels=num_features_out,\
            out_channels=num_features_out,kernel_size=3,padding=1))
        self.skip_connection_spade = SPADE(num_channels=num_features_in)
        self.skip_connection_conv = spectral_norm(nn.Conv2d(in_channels=num_features_in,\
                                                out_channels=num_features_out,\
                                                    kernel_size=1,\
                                                        bias=False))
    
    def forward(self,x,segmentation_map):
        skip_features = self.skip_connection_spade(x,segmentation_map)
        skip_features = F.leaky_relu(skip_features,0.2)
        skip_features = self.skip_connection_conv(skip_features)

        x = self.conv1(F.leaky_relu(self.spade1(x,segmentation_map),0.2))
        x = self.conv2(F.leaky_relu(self.spade2(x,segmentation_map),0.2))
        return skip_features + x   
class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.linear1 = nn.Linear(cfg.LATENT_DIM,16384)
        self.upsample = nn.Upsample(scale_factor=2)
        self.block1 = SPADEResBlk(num_features_in=1024,num_features_out=1024)
        self.block2 = SPADEResBlk(num_features_in=1024,num_features_out=1024)
        self.block3 = SPADEResBlk(num_features_in=1024,num_features_out=512)
        self.block4 = SPADEResBlk(num_features_in=512,num_features_out=256)
        self.block5 = SPADEResBlk(num_features_in=256,num_features_out=128)
        self.block6 = SPADEResBlk(num_features_in=128,num_features_out=64)
        # self.block7 = SPADEResBlk(num_features_in=64,num_features_out=32)

        self.conv = nn.Conv2d(in_channels=64,out_channels=3,kernel_size=3,padding=1)

    def forward(self,latent_vec,segmentation_map):
        x = self.linear1(latent_vec)
        x = x.reshape(-1,1024,4,4)
        x = self.block1(x,segmentation_map)
        x = self.upsample(x)
        x = self.block2(x,segmentation_map)
        x = self.upsample(x)
        x = self.block3(x,segmentation_map)
        x = self.upsample(x)
        x = self.block4(x,segmentation_map)
        x = self.upsample(x)
        x = self.block5(x,segmentation_map)
        x = self.upsample(x)
        x = self.block6(x,segmentation_map)
        x = self.upsample(x)
        # print(x.shape)        
        # x = self.block7(x,segmentation_map)
        # x = self.upsample(x)
        # print(x.shape)        
        x = F.leaky_relu(x,0.2)
        x = self.conv(x)
        x = torch.tanh(x)
        return x

class GeneratorLarge(nn.Module):
    def __init__(self):
        super(GeneratorLarge,self).__init__()
        self.linear1 = nn.Linear(cfg.LATENT_DIM,16384)
        self.upsample = nn.Upsample(scale_factor=2)
        self.block1 = SPADEResBlk(num_features_in=1024,num_features_out=1024)
        self.block2 = SPADEResBlk(num_features_in=1024,num_features_out=1024)
        self.block3 = SPADEResBlk(num_features_in=1024,num_features_out=512)
        self.block4 = SPADEResBlk(num_features_in=512,num_features_out=256)
        self.block5 = SPADEResBlk(num_features_in=256,num_features_out=128)
        self.block6 = SPADEResBlk(num_features_in=128,num_features_out=64)
        self.block7 = SPADEResBlk(num_features_in=64,num_features_out=32)

        self.conv = nn.Conv2d(in_channels=32,out_channels=3,kernel_size=3,padding=1)

    def forward(self,latent_vec,segmentation_map):
        x = self.linear1(latent_vec)
        x = x.reshape(-1,1024,4,4)
        x = self.block1(x,segmentation_map)
        x = self.upsample(x)
        x = self.block2(x,segmentation_map)
        x = self.upsample(x)
        x = self.block3(x,segmentation_map)
        x = self.upsample(x)
        x = self.block4(x,segmentation_map)
        x = self.upsample(x)
        x = self.block5(x,segmentation_map)
        x = self.upsample(x)
        x = self.block6(x,segmentation_map)
        x = self.upsample(x)
        # print(x.shape)        
        x = self.block7(x,segmentation_map)
        # x = self.upsample(x)
        # print(x.shape)        
        x = F.leaky_relu(x,0.2)
        x = self.conv(x)
        x = torch.tanh(x)
        return x        
class DiscriminatorBlock(nn.Module):
    def __init__(self,channels_in,channels_out,with_norm=True):
        super(DiscriminatorBlock,self).__init__()
        if with_norm:
            self.block = nn.Sequential(
                                        spectral_norm(nn.Conv2d(in_channels=channels_in,out_channels=channels_out,\
                                                    kernel_size=4,stride=2,bias=False,padding=1)),           
                                        nn.InstanceNorm2d(channels_out),
                                        nn.LeakyReLU(0.2)
                                        )
        else:
            self.block = nn.Sequential(
                                        spectral_norm(nn.Conv2d(in_channels=channels_in,out_channels=channels_out,\
                                                    kernel_size=4,stride=2,bias=False,padding=1)),
                                        nn.LeakyReLU(0.2)
                                        )
    def forward(self,x):
        return self.block(x)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        # change input channeldim        
        self.block1 = spectral_norm(nn.Conv2d(cfg.NUM_CLASSES+3,64,kernel_size=4,stride=2,bias=True))
        self.block2 = DiscriminatorBlock(64,128,False)
        self.block3 = DiscriminatorBlock(128,256)
        self.block4 = DiscriminatorBlock(256,512)
        self.block5 = DiscriminatorBlock(512,512)
        self.in7 = nn.InstanceNorm2d(512)
        self.conv8 = spectral_norm(nn.Conv2d(512,1,kernel_size=4))
    
    def forward(self,segmentation_map,img):
        # print(segmentation_map.shape,img.shape)
        concat_img = torch.concat([segmentation_map,img],dim=1)
        op1 = self.block2(self.block1(concat_img))
        op2 = self.block3(op1)
        op3 = self.block4(op2)
        op4 = self.block5(op3)
        op5 = self.conv8(F.leaky_relu(self.in7(op4),0.2))
        return [op1,op2,op3,op4,op5]
