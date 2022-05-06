

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
import matplotlib.pyplot as plt


def weights_init(m):
    ''' Function for initializing all model weights '''
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        

def convert_tensors_to_list_of_images(tensor_list):
    tensor_list = tensor_list.detach().cpu().numpy().astype(np.float)
    tensor_list = np.transpose(tensor_list,(0,2,3,1))
    tensor_list = (tensor_list+1)/2
    tensor_list=(tensor_list*255).astype(np.uint8)
    return tensor_list
def get_images(infer_maps,infer_images,infer_fakes,save=""):
    infer_maps_conv,infer_images_conv,infer_fakes_conv = convert_tensors_to_list_of_images(infer_maps),\
    convert_tensors_to_list_of_images(infer_images),convert_tensors_to_list_of_images(infer_fakes)
    fig,axs = plt.subplots(infer_maps_conv.shape[0],3,figsize=(20,20))
    for i,(map,img,fake) in enumerate(zip(infer_maps_conv,infer_images_conv,infer_fakes_conv)):
        # print(i)
        axs[i,0].imshow(map)
        axs[i,0].set_title("Input")
        axs[i,1].imshow(fake)
        axs[i,1].set_title("Output")
        axs[i,2].imshow(img)
        axs[i,2].set_title("Ground Truth")
    if save!="":
        fig.savefig(save)
    return fig,axs