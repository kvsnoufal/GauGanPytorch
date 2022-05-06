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
import network_keras as network
import loss_keras as loss
import dataset

DEBUG_MODE = False

if __name__=="__main__":
    if DEBUG_MODE == False:
        img_files = glob(cfg.INPUT_FILES)
        np.random.shuffle(img_files)
        train_index = int((1-cfg.TEST_SPLIT) * len(img_files))
        train_img_files,test_img_files = img_files[:train_index],img_files[train_index:]
    else:
        print("Debug Mode")
        img_files = glob(cfg.INPUT_FILES)[:8]
        train_img_files,test_img_files = img_files,img_files
    print("TRAIN SIZE {} TEST SIZE {}".format(len(train_img_files),len(test_img_files)))

    train_dataset=dataset.GauGanDataset(train_img_files)
    train_loader=torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=True,
            num_workers=12
        )
    eval_dataset=dataset.GauGanDataset(test_img_files)
    eval_loader=torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
            num_workers=1
        )

    gen_loss = loss.Gen_loss().to(cfg.DEVICE)
    kld_loss = loss.KLD_Loss().to(cfg.DEVICE)
    vgg_loss = loss.VGGLoss().to(cfg.DEVICE)
    feat_loss = loss.FeatureLossDisc().to(cfg.DEVICE)
    disc_loss = loss.Disc_HingeLoss().to(cfg.DEVICE)

    encoder = network.Encoder().to(cfg.DEVICE)
    generator = network.Generator().to(cfg.DEVICE)
    discriminator = network.Discriminator().to(cfg.DEVICE)

    gen_optimizer = torch.optim.Adam(list(generator.parameters()) + list(encoder.parameters()),\
                                    lr=1e-4,weight_decay=0.0001,betas=(0.0,0.999)) 
    disc_optimizer = torch.optim.Adam(discriminator.parameters(),lr=4e-4,weight_decay=0.0001,betas=(0.0,0.999))        

    writer = SummaryWriter("{}/runs/{}".format(cfg.OUTPUT_FOLDER,cfg.RUN_ID))
    # train step
    for epoch in tqdm(range(cfg.EPOCHS),total=cfg.EPOCHS):
        generator.train()
        discriminator.train()
        encoder.train()
        d_losses=0
        g_losses=0

        for batch in train_loader:
            image = batch["image"].to(cfg.DEVICE)
            segmentation_map = batch["segmentation_map"].to(cfg.DEVICE)
            mu,var = encoder(image)
            latent_vec = encoder.get_latent_vector(mu,var)
            # train discriminator
            disc_optimizer.zero_grad()
            fake_images = generator(latent_vec.detach(),segmentation_map)
            pred_fake = discriminator(fake_images.detach(),segmentation_map)[-1]
            pred_real = discriminator(image,segmentation_map)[-1]
            d_loss_fake = disc_loss(pred_fake,False)
            d_loss_real = disc_loss(pred_real,True)
            d_loss = 0.5*d_loss_fake + 0.5*d_loss_real
            d_loss.backward()
            disc_optimizer.step()

            # train gen
            real_d_outputs = discriminator(image,segmentation_map)
            fake_images = generator(latent_vec,segmentation_map)
            preds_fake = discriminator(fake_images,segmentation_map)
            gen_loss_batch = gen_loss(preds_fake[-1]) # gen should try to target disc output 1
            kld_loss_batch = kld_loss(mu,var)
            vgg_loss_batch = vgg_loss(image,fake_images)
            feat_loss_batch = feat_loss(real_d_outputs,preds_fake)
            g_loss = gen_loss_batch * (1) + \
                    kld_loss_batch * (0.1)+ \
                    vgg_loss_batch * (0.1) + \
                    feat_loss_batch* (10)
            g_loss.backward()
            gen_optimizer.step()

            d_losses+=d_loss.item()
            g_losses+=g_loss.item()
        d_losses = d_losses/len(train_loader)
        g_losses = g_losses/len(train_loader)
        writer.add_scalar("gLoss/train", g_losses, epoch)
        writer.add_scalar("dLoss/train", d_losses, epoch)
        if (epoch+1)%cfg.EVAL_EVERY==0:
            # EVALUATE
            generator.eval()
            discriminator.eval()
            encoder.eval()
            eval_d_losses = 0
            eval_g_losses = 0
            eval_batch_counter = 0
            infer_maps = []
            infer_images = []
            infer_fakes = []
            with torch.no_grad():
                for batch in eval_loader:
                    image = batch["image"].to(cfg.DEVICE)
                    segmentation_map = batch["segmentation_map"].to(cfg.DEVICE)
                    mu,var = encoder(image)
                    latent_vec = encoder.get_latent_vector(mu,var)
                    fake_image = generator(latent_vec=latent_vec,segmentation_map=segmentation_map)
                    fake_disc_output = discriminator(fake_image,segmentation_map)
                    real_disc_output= discriminator(image,segmentation_map)

                    gen_loss_batch = gen_loss(fake_disc_output[-1])
                    kld_loss_batch = kld_loss(mu,var)
                    vgg_loss_batch = vgg_loss(image,fake_image)
                    feat_loss_batch = feat_loss(real_disc_output,fake_disc_output)

                    g_loss = gen_loss_batch * (1.0) + \
                            kld_loss_batch * (0.1)+ \
                            vgg_loss_batch * (0.1) + \
                            feat_loss_batch* (10)
                    real_disc_output_for_d = discriminator(image.detach(),segmentation_map)
                    fake_disc_output_for_d = discriminator(fake_image.detach(),segmentation_map)
                    real_d_loss = disc_loss(real_disc_output_for_d[-1],True) *0.5
                    fake_d_loss = disc_loss(fake_disc_output_for_d[-1],False)*0.5
                    d_loss = real_d_loss + fake_d_loss
                    eval_d_losses+=d_loss.item()
                    eval_g_losses+=g_loss.item()

                    if eval_batch_counter<1:
                        infer_maps.append(batch["label_img"].detach())
                        infer_images.append(image.detach())
                        infer_fakes.append(fake_image.detach())
                    eval_batch_counter+=1
            eval_d_losses = eval_d_losses/len(eval_loader)
            eval_g_losses = eval_g_losses/len(eval_loader)
            print("EPOCH {}: train g {} d {} eval g {} d {}".format(epoch,g_losses,d_losses,\
                eval_g_losses,eval_d_losses))
            
            writer.add_scalar("gLoss/eval", eval_g_losses, epoch)
            writer.add_scalar("dLoss/eval", eval_d_losses, epoch)


            infer_maps = torch.concat(infer_maps,dim=0)
            infer_images = torch.concat(infer_images,dim=0)
            infer_fakes = torch.concat(infer_fakes,dim=0)
            # break
            infer_maps_conv,infer_images_conv,infer_fakes_conv = utils.convert_tensors_to_list_of_images(infer_maps),\
                                                                utils.convert_tensors_to_list_of_images(infer_images),\
                                                                utils.convert_tensors_to_list_of_images(infer_fakes)
            fig,axs = plt.subplots(infer_maps_conv.shape[0],3,figsize=(20,20))
            for i,(map,img,fake) in enumerate(zip(infer_maps_conv,infer_images_conv,infer_fakes_conv)):
                # print(i)
                axs[i,0].imshow(map)
                axs[i,0].set_title("Input")
                axs[i,1].imshow(fake)
                axs[i,1].set_title("Output")
                axs[i,2].imshow(img)
                axs[i,2].set_title("Ground Truth")
            save_folder = os.path.join(cfg.OUTPUT_FOLDER,cfg.SAVE_LOGS_FOLDER,cfg.RUN_ID)
            os.makedirs(save_folder,exist_ok=True)
            save_path = os.path.join(save_folder,"{}.png".format(epoch))
            fig.savefig(save_path)   
            im = Image.open(save_path)
            im = torchvision.transforms.ToTensor()(im)
            writer.add_image('images', im, epoch)
        if (epoch+1) %cfg.SAVE_EVERY==0:
            print("saving..... {}".format(epoch))
            save_folder = os.path.join(cfg.OUTPUT_FOLDER,cfg.SAVE_MODEL_FOLDER,cfg.RUN_ID)
            os.makedirs(save_folder,exist_ok=True)
            save_path = os.path.join(save_folder,"enc_{}.pt".format(epoch))
            torch.save(encoder.state_dict(), save_path)
            save_path = os.path.join(save_folder,"gen_{}.pt".format(epoch))
            torch.save(generator.state_dict(), save_path)
            save_path = os.path.join(save_folder,"disc_{}.pt".format(epoch))
            torch.save(discriminator.state_dict(), save_path)

        writer.flush()
    writer.close()