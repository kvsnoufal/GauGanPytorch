
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

import pandas as pd

class GauGanDataset:
    def __init__(self,img_paths):
        self.img_paths = img_paths
        self.img_transforms = transforms.Compose([
            transforms.Resize((cfg.IMAGE_HEIGHT,cfg.IMAGE_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.map_transforms = transforms.Compose([
            transforms.Resize((cfg.IMAGE_HEIGHT,cfg.IMAGE_WIDTH)),
            transforms.ToTensor(),
        ])
        self.longest_side_resize = transforms.Resize(cfg.IMAGE_HEIGHT)
    def __len__(self):
        return len(self.img_paths)
    def random_crop_dimensions(self,img):
        
        xmax,ymax = img.size
        x1max = xmax-cfg.IMAGE_WIDTH
        # print(x1max)
        x1min = 0
        x1 = int(np.random.uniform(x1min,x1max))
        x2 = x1+cfg.IMAGE_WIDTH
        # print(x1,x2,x2-x1)

        y1max = ymax-cfg.IMAGE_HEIGHT
        # print(y1max)
        y1min = 0
        y1 = int(np.random.uniform(y1min,y1max))
        y2 = y1+cfg.IMAGE_HEIGHT
        # print(x1,x2,x2-x1)
        # print(y1,y2,y2-y1)
        return x1,x2,y1,y2
    def __getitem__(self,index):
        image = Image.open(self.img_paths[index]).convert("RGB")
        label = Image.open(self.img_paths[index].replace("jpg","bmp"))
        label_img = Image.open(self.img_paths[index].replace("jpg","png")).convert("RGB")
        image = self.longest_side_resize(image)
        label = self.longest_side_resize(label)
        label_img = self.longest_side_resize(label_img)
        
        x1,x2,y1,y2 = self.random_crop_dimensions(image)
        image = image.crop((x1,y1,x2,y2))
        label = label.crop((x1,y1,x2,y2))
        label_img = label_img.crop((x1,y1,x2,y2))

        image = self.img_transforms(image)
        label = self.map_transforms(label).squeeze(0)
        label = (label*255).long()
        label_img = self.img_transforms(label_img)
        label_ohe = F.one_hot(label,num_classes=cfg.NUM_CLASSES).permute(2,0,1).to(image.dtype)
        return {
            "image":image,
            "segmentation_map":label_ohe,
            "label_img": label_img

        }

FLICKRCLASSES = ["clouds" ,
                "sky-other" ,
                "mountain" ,
                "tree" ,
                "sea" ,
                "grass" ,
                "rock" ,
                "hill" ,
                "sand" ,
                "river" ,
                "water-other" ,
                "bush" ,]
def get_classtable():
        with open("../dev_flickr/deeplab-pytorch/data/datasets/cocostuff/labels.txt") as f:
            classes = {}
            for label in f:
                label = label.rstrip().split("\t")
                classes[int(label[0])] = label[1].split(",")[0]
        return classes                
def get_file_list_flickr(debug):
    OUTPUT_FOLDER = "../data/flickr/maps_raw"
    if debug:
        images = glob("../data/flickr/archive/*jpg")[:50]
    else:
        images = glob("../data/flickr/archive/*jpg")
    already_done = glob("../data/flickr/maps_raw/*png")
    final_image_list = []
    for file in images:
        output_file = os.path.join(OUTPUT_FOLDER,os.path.basename(file.replace(".jpg",".png")))
        if output_file in already_done:
            final_image_list.append(file)
        else:
            continue
    # print("all files : {}\nfinal image lists : {}".format(len(images),len(final_image_list)))
    seg_maps = [os.path.join(OUTPUT_FOLDER,os.path.basename(f.replace(".jpg",".png")))\
    for f in final_image_list ]
    df_profile = pd.DataFrame()
    for i,seg_map in tqdm(enumerate(seg_maps),total=len(seg_maps)):
        img_classes = list(np.array(Image.open(seg_map)).flatten())
        
        # df = pd.Series(img_classes).value_counts().reset_index().rename(columns={"index":"class",\
        #                                                                         0:"count"})
        df = pd.DataFrame([[int(t),len([img_ for img_ in img_classes if img_==t])] for t in np.unique(img_classes)],columns=["class","count"])                                                                       
        df["seg_path"] = seg_map
        df["img_path"] = df["seg_path"].apply(lambda x: x.replace(".png",".jpg")).\
            apply(lambda x: x.replace("maps_raw","archive"))
        if df_profile.empty:
            df_profile = df.copy()
        else:
            df_profile = df_profile.append(df.copy())
    
    class_table = get_classtable()

    df_profile["class_name"] = df_profile["class"].map(class_table)
    df_profile["fname"] = df_profile["img_path"].apply(lambda x: str(os.path.basename(x)[:-4]))
    # files_has_classes = df_profile[df_profile["class_name"].isin(FLICKRCLASSES)]["fname"].unique()
    # other_classes = df_profile[df_profile["class_name"].isin(FLICKRCLASSES)==False]["class_name"].unique()
    files_with_other_classes = df_profile[df_profile["class_name"].isin(FLICKRCLASSES)==False]["fname"].unique()

    files_with_only_classes = df_profile[df_profile["fname"].isin(files_with_other_classes)==False]
    files_with_only_classes = files_with_only_classes.reset_index(drop=True)


    
    
    for i,row in tqdm(files_with_only_classes.iterrows(),total=len(files_with_only_classes)):
        shape_x,shape_y = Image.open(row["img_path"]).size
        files_with_only_classes.loc[i,"shape_x"] = shape_x
        files_with_only_classes.loc[i,"shape_y"] = shape_y
    files_with_only_classes = files_with_only_classes[(files_with_only_classes["shape_y"]>cfg.IMAGE_WIDTH)\
        &(files_with_only_classes["shape_x"]>cfg.IMAGE_HEIGHT)].reset_index(drop=True)
    final_file_list = list(files_with_only_classes["img_path"].unique())
    return final_file_list
class flickrDataset:
    def __init__(self,img_paths):
        self.img_paths = img_paths
        self.classTable = get_classtable()
        self.classTableEncoding = {int(k):FLICKRCLASSES.index(v) for k,v in self.classTable.items() if v in FLICKRCLASSES}

        self.img_transforms = transforms.Compose([
            transforms.Resize((cfg.IMAGE_HEIGHT,cfg.IMAGE_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.map_transforms = transforms.Compose([
            transforms.Resize((cfg.IMAGE_HEIGHT,cfg.IMAGE_WIDTH)),
            transforms.ToTensor(),
        ])
        self.longest_side_resize = transforms.Resize(cfg.IMAGE_HEIGHT)

    def __len__(self):
        return len(self.img_paths)
    def random_crop_dimensions(self,img):
        
        xmax,ymax = img.size
        x1max = xmax-cfg.IMAGE_WIDTH
        # print(x1max)
        x1min = 0
        x1 = int(np.random.uniform(x1min,x1max))
        x2 = x1+cfg.IMAGE_WIDTH
        # print(x1,x2,x2-x1)

        y1max = ymax-cfg.IMAGE_HEIGHT
        # print(y1max)
        y1min = 0
        y1 = int(np.random.uniform(y1min,y1max))
        y2 = y1+cfg.IMAGE_HEIGHT
        # print(x1,x2,x2-x1)
        # print(y1,y2,y2-y1)
        return x1,x2,y1,y2
    def __getitem__(self,index):
        image = Image.open(self.img_paths[index]).convert("RGB")
        label = Image.open(self.img_paths[index].replace(".jpg",".png").replace("archive","maps_raw"))
        label = np.array(label)
        for k,v in self.classTableEncoding.items():
            label[label==k] = v
        # print(self.img_paths[index].replace(".jpg",".png").replace("archive","maps_raw"),\
            # np.unique(label))
        label = Image.fromarray(label)
        label_img = Image.open(self.img_paths[index].replace(".jpg",".png").replace("archive","maps_raw")).convert("RGB")

        image = self.longest_side_resize(image)
        label = self.longest_side_resize(label)
        label_img = self.longest_side_resize(label_img)
        
        x1,x2,y1,y2 = self.random_crop_dimensions(image)
        image = image.crop((x1,y1,x2,y2))
        label = label.crop((x1,y1,x2,y2))
        label_img = label_img.crop((x1,y1,x2,y2))

        image = self.img_transforms(image)
        label = self.map_transforms(label).squeeze(0)
        label = (label*255).long()
        label_img = self.img_transforms(label_img)
        
        label_ohe = F.one_hot(label,num_classes=cfg.NUM_CLASSES).permute(2,0,1).to(image.dtype)
        return {
            "image":image,
            "segmentation_map":label_ohe,
            "label_img": label_img

        }        