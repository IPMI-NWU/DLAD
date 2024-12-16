#!/usr/bin/python3
import sys
sys.path.append('../')
import torch
import os
import time
import copy
import numpy as np
import pandas as pd
import argparse
import itertools
import torch.nn as nn
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from my_dataset import TrainDataset, TestDataset
from network import Encoder, Decoder, Discriminator, Latent_Dis
import torchvision.utils as vutils
# from sklearn import preprocessing
from torch.autograd import Variable, grad
from utils import Logger, LambdaLR, ReplayBuffer
from tensorboardX import SummaryWriter
from sklearn import metrics
from MS_SSIM_L1_loss import MS_SSIM_L1_LOSS

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

TIME = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))

if not os.path.exists('./model/{0}_Crop'.format(TIME)):
    os.makedirs('./model/{0}_Crop'.format(TIME))
if not os.path.exists('./runs/{0}_Crop'.format(TIME)):
    os.makedirs('./runs/{0}_Crop'.format(TIME))
if not os.path.exists('./output/{0}_Crop'.format(TIME)):
    os.makedirs('./output/{0}_Crop'.format(TIME))

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--batchSize', type=int, default=16, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0005, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--latent_size', type=int, default=128, help='latent_size')
parser.add_argument('--nu', type=float, default=0.9, help='proportion of allowed anomaly')
parser.add_argument('--n_critic', type=int, default=2)
parser.add_argument('--clamp', type=float, default=0.01)
opt = parser.parse_args()

# Networks
net_Enc = Encoder(opt.input_nc, opt.latent_size)
net_Enc.to(device)
net_Dec = Decoder(opt.output_nc, opt.latent_size)
net_Dec.to(device)
net_D = Discriminator(opt.input_nc)
net_D.to(device)
net_LD = Latent_Dis(opt.latent_size)
net_LD.to(device)

# Lossess
criterion_GAN = torch.nn.BCELoss()
# criterion_Rec = torch.nn.L1Loss()
criterion_Rec = MS_SSIM_L1_LOSS()
criterion_Vec_Clf = torch.nn.BCELoss()

# Optimizers & LR schedulers
optimizer_Enc = torch.optim.Adam(net_Enc.parameters(), lr=opt.lr, betas=(0.5, 0.9))
optimizer_Dec = torch.optim.Adam(net_Dec.parameters(), lr=opt.lr, betas=(0.5, 0.9))
optimizer_D = torch.optim.Adam(net_D.parameters(), lr=opt.lr, betas=(0.5, 0.9))
optimizer_LD = torch.optim.Adam(net_LD.parameters(), lr=opt.lr, betas=(0.5, 0.9))


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha = 1 - decay)


def sample_data(img_size):
    transform_ = transforms.Compose([transforms.Grayscale(),
                                    transforms.Resize((img_size, img_size)),
                                    transforms.ToTensor()])
    normal_image_dir = '/home/niuyi/dataset/original_dataset/CXR_ipmi/normal_crop'
    abnormal_image_dir = '/home/niuyi/dataset/original_dataset/CXR_ipmi/abnormal_crop'
    train_list_dir = './split_dataset/train_set.txt'
    valid_normal_list = './split_dataset/valid_normal.txt'
    valid_abnormal_list = './split_dataset/valid_abnormal.txt'
    train_dataset = TrainDataset(image_dir=normal_image_dir, train_list_dir=train_list_dir, transform=transform_)
    train_loader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True,drop_last=True)
    valid_dataset = TestDataset(normal_image_dir, valid_normal_list, abnormal_image_dir, valid_abnormal_list, transform=transform_ )
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True, drop_last=True)

    return train_loader, valid_loader

def train(encoder, decoder, discriminator, latent_dis):
    step = 0
    train_loader, valid_loader = sample_data(4 * 2 ** step)
    pbar = tqdm(range(650000))

    # alpha = 0
    one = torch.tensor(1, dtype=torch.float).to(device)
    one = one * 100
    mone = one * -1
    iteration = 0
    stabilize = False

    disc_loss_val = 0
    enc_loss_val = 0
    dec_loss_val = 0
    ld_loss_val = 0

    for i in pbar:

        alpha = min(1, 0.000025 * iteration)

        if stabilize is False and iteration > 40000:
            train_loader, valid_loader = sample_data(4 * 2 ** step)
            stabilize = True

        if iteration > 80000:
        # if iteration > 2000:
            alpha = 0
            iteration = 0
            step += 1
            stabilize = False
            if step > 6:
                alpha = 1
                step = 6
            train_loader, valid_loader = sample_data(4 * 2 ** step)

        real_image, _, _ = next(iter(train_loader))
        valid_z = Variable(torch.ones(real_image.size(0), dtype=torch.float32), requires_grad=False).to(device)    
        fake_z = Variable(torch.zeros(real_image.size(0), dtype=torch.float32), requires_grad=False).to(device)

        iteration += 1

        b_size = real_image.size(0)
        real_image = Variable(real_image).to(device)
        
        optimizer_D.zero_grad()
        real_predict= discriminator(real_image, step, alpha)
        real_predict = real_predict.mean()
        real_predict.backward(mone)

        gen_image = decoder(Variable(torch.randn(b_size, opt.latent_size)).to(device), step, alpha)
        gen_predict = discriminator(gen_image, step, alpha)
        gen_predict = gen_predict.mean()
        gen_predict.backward(one)

        real_out = encoder(real_image, step, alpha)
        real_latent = real_out[-1]
        rec_image = decoder(real_latent, step, alpha)
        rec_predict = discriminator(rec_image, step, alpha)
        rec_predict = rec_predict.mean()
        disc_loss_val = (real_predict - gen_predict - rec_predict).item()
        rec_predict.backward(one)
        optimizer_D.step()

        optimizer_LD.zero_grad()
        guss_latent = Variable(torch.randn(real_image.size(0), opt.latent_size)).to(device)
        clf_real = latent_dis(torch.squeeze(real_latent).detach())
        clf_guss = latent_dis(torch.squeeze(guss_latent).detach()) 
        # optim_LD
        loss_LD_real = criterion_Vec_Clf(torch.squeeze(clf_real), valid_z)
        loss_LD_guss = criterion_Vec_Clf(torch.squeeze(clf_guss), fake_z)
        loss_LD = loss_LD_guss + loss_LD_real
        ld_loss_val = loss_LD.item()
        loss_LD.backward()
        optimizer_LD.step()
        # accumulate(LD_running, latent_dis)

        for _ in range(opt.n_critic):
            requires_grad(encoder, True)
            requires_grad(decoder, True)
            requires_grad(discriminator, False)
            requires_grad(latent_dis, False)

            optimizer_Enc.zero_grad()
            real_out = encoder(real_image, step, alpha)
            real_latent = real_out[-1]
            rec_image = decoder(real_latent, step, alpha)
            loss_Enc_rec = criterion_Rec(rec_image, real_image)
            clf_real = latent_dis(torch.squeeze(real_latent))
            loss_Enc_Vec_Clf = criterion_Vec_Clf(torch.squeeze(clf_real), fake_z)
            loss_Enc_GAN = -torch.mean(discriminator(rec_image, step, alpha).view(-1))
            loss_Enc = 5*loss_Enc_rec + loss_Enc_Vec_Clf + loss_Enc_GAN
            enc_loss_val = loss_Enc.item()
            loss_Enc.backward(retain_graph=True)
            optimizer_Enc.step()
            
            optimizer_Dec.zero_grad()
            rec_image = decoder(real_latent.detach(), step, alpha)
            loss_Dec_rec = criterion_Rec(rec_image, real_image)
            guss_latent = Variable(torch.randn(real_image.size(0), opt.latent_size)).to(device)
            gen_image = decoder(guss_latent, step, alpha)
            pre_rec = discriminator(rec_image, step, alpha).view(-1)
            pre_gen = discriminator(gen_image, step, alpha).view(-1)
            loss_Dec_GAN_rec = -torch.mean(pre_rec)
            loss_Dec_GAN_gen = -torch.mean(pre_gen)
            loss_Dec = 5*loss_Dec_rec + loss_Dec_GAN_gen + loss_Dec_GAN_rec
            dec_loss_val = loss_Dec.item()
            loss_Dec.backward()
            optimizer_Dec.step()

            requires_grad(encoder, False)
            requires_grad(decoder, False)
            requires_grad(discriminator, True)
            requires_grad(latent_dis, True)

        if (i + 1) % 1000 == 0:
            v_real_image, _, _ = next(iter(valid_loader))
            v_real_image = Variable(v_real_image).to(device)
            v_latent_out = encoder(v_real_image, step, alpha)
            v_latent_vec = v_latent_out[-1]
            v_guss_vec = Variable(torch.randn(v_real_image.size(0), opt.latent_size)).to(device)
            v_rec_image = decoder(v_latent_vec, step, alpha)
            v_gen_image = decoder(v_guss_vec, step, alpha)
            # if (epoch + 1) % 3 == 0:
            vutils.save_image(torch.cat((v_real_image.cpu(), v_rec_image.cpu(), torch.abs(v_real_image - v_rec_image).cpu(), v_gen_image.cpu()), dim=0),
                            './output/{0}_Crop/{1}_output.png'.format(TIME, i))
            with SummaryWriter('./runs/{0}_Crop'.format(TIME)) as writer:
                writer.add_scalar('scalar/loss_Enc', enc_loss_val, i)
                writer.add_scalar('scalar/loss_Dec', dec_loss_val , i)
                writer.add_scalar('scalar/loss_D', disc_loss_val, i)
                writer.add_scalar('scalar/loss_LD', ld_loss_val, i)
                writer.add_scalar('scalar/loss_rec', loss_Enc_rec.item(), i)
                writer.close()


        if (i + 1) % 1000 == 0 and alpha == 1:
            torch.save(encoder, './model/{0}_Crop/net_Enc_epoch{1}_expand{2}.pth'.format(TIME, i, step))
            torch.save(decoder, './model/{0}_Crop/net_Dec_epoch{1}_expand{2}.pth'.format(TIME, i, step))
            torch.save(discriminator, './model/{0}_Crop/net_D_epoch{1}_expand{2}.pth'.format(TIME, i, step))
            torch.save(latent_dis, './model/{0}_Crop/net_LD_epoch{1}_expand{2}.pth'.format(TIME, i, step))


        pbar.set_description(
            (f'{i + 1}; Enc: {enc_loss_val:.5f}; Dec: {dec_loss_val:.5f}; Dis: {disc_loss_val:.5f};'
             f' LD: {ld_loss_val:.5f}; Alpha: {alpha:.3f}'))
        


if __name__ == '__main__':
    args = parser.parse_args()

    train(encoder=net_Enc, decoder=net_Dec, discriminator=net_D, latent_dis=net_LD)
