import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
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
from torch.utils.tensorboard import SummaryWriter
from config import Config as cfg
import matplotlib.pyplot as plt
import pandas as pd

import utils
import network
import loss
import dataset
import cv2
st.set_page_config(layout="wide")
# inits
img_files = glob("../data/flickr/style_images/*")[:5]
ds = dataset.flickrDataset(img_files)



encoder = network.EncoderLarge().to(cfg.DEVICE)
generator = network.GeneratorLarge().to(cfg.DEVICE)
encoder.load_state_dict(torch.load(cfg.ENCODER_WEIGHTS_EVAL))
generator.load_state_dict(torch.load(cfg.GENERATOR_WEIGHTS_EVAL))
encoder.eval()
generator.eval()
print()
latent_vecs = []
for style_image_file in img_files:
    style_image = Image.open(style_image_file).convert("RGB")
    style_image_resized = ds.longest_side_resize(style_image)

    x1,x2,y1,y2 = ds.random_crop_dimensions(style_image_resized)
    style_image_resized = style_image_resized.crop((x1,y1,x2,y2))
    style_image_resized_input = ds.img_transforms(style_image_resized).unsqueeze(0)
    
    with torch.no_grad():
        style_image_resized_input = style_image_resized_input.to(cfg.DEVICE)
        mu,var = encoder(style_image_resized_input)
        latent_vec = encoder.get_latent_vector(mu,var)
        latent_vecs.append(latent_vec)
def convert_to_label(canvas_extract):
    cond_list = [
    canvas_extract == (126.0,104.0,195.0),
    canvas_extract == (222.0,235.0,241.0),
    canvas_extract == (146.0,124.0,88.0),
    canvas_extract == (129.0,128.0,126.0),
    canvas_extract == (72.0,47.0,10.0),
    canvas_extract == (226.0,188.0,16.0),
    canvas_extract == (24.0,16.0,226.0),
    canvas_extract == (16.0,207.0,226.0),
    canvas_extract == (100.0,220.0,180.0),
    canvas_extract == (63.0,60.0,3.0),
    canvas_extract == (120.0,239.0,95.0),
    canvas_extract == (33.0,146.0,10.0),
    ]
    choices = [
    1,
    0,
    2,
    7,
    6,
    8,
    4,
    9,
    10,
    3,
    5,
    11,
    ]
    converted = np.select(cond_list,choices,1)[:,:,0]
    return converted.astype(np.uint8)

def get_predictions(canvas_extract):
    label = convert_to_label(canvas_extract)
    label = Image.fromarray(label)
    label = ds.map_transforms(label).squeeze(0)
    label = (label*255).long()
    label_ohe = F.one_hot(label,num_classes=cfg.NUM_CLASSES).permute(2,0,1).to(torch.float32).unsqueeze(0)
    fake_images = []
    for latent_vec in latent_vecs:
        with torch.no_grad():
            latent_vec = latent_vec.to(cfg.DEVICE)
            label_ohe = label_ohe.to(cfg.DEVICE)
            fake_image = generator(latent_vec=latent_vec,segmentation_map=label_ohe).squeeze(0)
            fake_image = fake_image.detach().cpu().numpy()
        
        fake_image = np.transpose(fake_image,(1,2,0))
        fake_image = (fake_image+1)/2
        fake_image = (fake_image*255).astype(np.uint8)
        fake_image = cv2.resize(fake_image, (400,400), interpolation=cv2.INTER_CUBIC)
        fake_images.append(fake_image)

    return fake_images


color_map = {
    "sky":"#7EB8C3",
    "clouds":"#DEEBF1",
    "mountain":"#927C58",
    "hill":"#81807E",
    "rock":"#482F0A",
    "sand":"#E2BC10",
    "sea":"#1810E2",
    "river":"#10CFE2",
    "water":"#64DCB4",
    "tree":"#3F3C03",
    "grass":"#78EF5F",
    "bush":"#21920A"
}

# init 



drawing_mode = "freedraw"
# stroke_width = 10
stroke_width = st.sidebar.slider("Stroke width: ", 1, 80, 3)
# if drawing_mode == 'point':
#     point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)

bg_color = st.sidebar.color_picker("Background color hex: ", color_map["sky"])
# bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
brush = st.sidebar.radio(
     "Select brush",
     ("clouds" ,"sky" ,"mountain" ,"hill" ,"rock" , "sand" , "sea" , "river" , "water" ,"tree" ,"grass" ,"bush" ))

stroke_color = st.sidebar.color_picker("Stroke color hex: ",color_map[brush])
# print(bg_color)
realtime_update = st.sidebar.checkbox("Update in realtime", True)

    

# Create a canvas component
# with c1:
st.write("draw here: ")
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    background_image= None,
    update_streamlit=realtime_update,
    height=450,
    width=450,
    drawing_mode=drawing_mode,
    point_display_radius=0,
    key="canvas",
)
st.write("model output")
# c1, c2, c3, c4 ,c5 = st.columns((1, 1, 1, 1,1))
c1, c2 = st.columns((1, 1))
# Do something interesting with the image data and paths
# mapping_dict = {
# 	1: {"name":'sky',"color" : (126,104,195)},
# 	0: {"name":'clouds',"color" : (222,235,241)},
# 	2: {"name":'mountain',"color" : (146,124,88)},
# 	7: {"name":'hill',"color" : (129,128,126)},
# 	6: {"name":'rock',"color" : (72,47,10)},
# 	8: {"name":'sand',"color" : (226,188,16)},
# 	4: {"name":'sea',"color" : (24,16,226)},
# 	9: {"name":'river',"color" : (16,207,226)},
# 	10: {"name":'water',"color" : (100,220,180)},
# 	3: {"name":'tree',"color" : (63,60,3)},
# 	5: {"name":'grass',"color" : (120,239,95)},
# 	11: {"name":'bush',"color" : (33,146,10)},
# }

print(canvas_result.image_data[:,:,:3][0])
canvas_extract = canvas_result.image_data[:,:,:3].astype(np.float32)

results = get_predictions(canvas_extract)

# st.write(str(converted[0]))

if canvas_result.image_data is not None:
    c1.image(results[0])
if canvas_result.image_data is not None:
    c2.image(results[3])
# if canvas_result.image_data is not None:
#     c3.image(results[2])
# if canvas_result.image_data is not None:
#     c4.image(results[3])  
# if canvas_result.image_data is not None:
#     c5.image(results[4])                
if canvas_result.json_data is not None:
    objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow
    for col in objects.select_dtypes(include=['object']).columns:
        objects[col] = objects[col].astype("str")
    st.dataframe(objects)