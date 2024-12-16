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
from torch.nn import functional as F
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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

def Perceptual_loss(x, y):
    loss = torch.zeros(1, dtype=torch.float32).to(device)
    criterion = nn.MSELoss()
    for i in range(len(x)):
        loss += criterion(x[i], y[i])

    return loss

def Vec_loss(x, y):
    loss = torch.zeros(1, dtype=torch.float32).to(device)
    criterion = nn.MSELoss()
    for i in range(len(x)):
        loss += criterion(x[i], y[i])

    return loss


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=1600, help='number of epochs of training')
    parser.add_argument('--batchSize', type=int, default=10, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.0005, help='initial learning rate')
    parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
    parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
    parser.add_argument('--latent_size', type=int, default=128, help='latent_size')
    parser.add_argument('--nu', type=float, default=0.9, help='proportion of allowed anomaly')
    parser.add_argument('--n_critic', type=int, default=1)
    parser.add_argument('--clamp', type=float, default=0.01)
    opt = parser.parse_args()
    print(opt)
    return opt

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)

if __name__ =="__main__":

    TIME = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))

    if not os.path.exists('./model/{0}_Crop'.format(TIME)):
        os.makedirs('./model/{0}_Crop'.format(TIME))
    if not os.path.exists('./runs/{0}_Crop'.format(TIME)):
        os.makedirs('./runs/{0}_Crop'.format(TIME))
    if not os.path.exists('./output/{0}_Crop'.format(TIME)):
        os.makedirs('./output/{0}_Crop'.format(TIME))

    opt = parse_args()

    ###### Definition of variables ######

    # Inputs & targets memory allocation

    # Dataset loader
    transform_ = transforms.Compose([transforms.Grayscale(),
                                    transforms.Resize((256, 256)),
                                    transforms.ToTensor()])
    normal_image_dir = '/home/niuyi/dataset/original_dataset/CXR_ipmi/normal_crop'
    abnormal_image_dir = '/home/niuyi/dataset/original_dataset/CXR_ipmi/abnormal_crop'
    train_list_dir = './split_dataset/train_set.txt'
    train_dataset_1 = TrainDataset(image_dir=normal_image_dir, train_list_dir=train_list_dir, transform=transform_)
    train_loader_1 = DataLoader(train_dataset_1, batch_size=opt.batchSize, shuffle=True,drop_last=True)
    valid_normal_list = './split_dataset/valid_normal.txt'
    valid_abnormal_list = './split_dataset/valid_abnormal.txt'
    # Loss plot
    logger = Logger(opt.n_epochs, len(train_loader_1))

    ###################################

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
    criterion_Rec = torch.nn.L1Loss()
    criterion_Vec_Clf = torch.nn.BCELoss()

    # Optimizers & LR schedulers
    optimizer_Enc = torch.optim.Adam(net_Enc.parameters(), lr=opt.lr, betas=(0.5, 0.9))
    optimizer_Dec = torch.optim.Adam(net_Dec.parameters(), lr=opt.lr, betas=(0.5, 0.9))
    optimizer_D = torch.optim.Adam(net_D.parameters(), lr=opt.lr, betas=(0.5, 0.9))
    optimizer_LD = torch.optim.Adam(net_LD.parameters(), lr=opt.lr, betas=(0.5, 0.9))


    # lr_scheduler_Enc = torch.optim.lr_scheduler.LambdaLR(optimizer_Enc, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    # lr_scheduler_Dec = torch.optim.lr_scheduler.LambdaLR(optimizer_Dec, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    # lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    # lr_scheduler_LD = torch.optim.lr_scheduler.LambdaLR(optimizer_LD, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

    step=0
    alpha=0
    expand=0
    iters = 0
    stabilize= False
    next_idx = 1
    # next_expand_epoch = [0, 80, 250, 500, 800, 1200, 1700, 2400]
    next_expand_epoch = [0, 50, 150, 300, 500, 750, 1050, 1400]
    for epoch in range(opt.n_epochs):
        print("================epoch:{0}================".format(epoch+1))
        record_loss_Enc = 0.
        record_loss_Enc_Rec = 0.
        record_loss_Enc_Vec_Clf = 0.
        record_loss_Dec = 0.
        record_loss_Dec_GAN_Rec = 0.
        record_loss_Dec_GAN_Gen = 0.
        record_loss_Dec_Rec = 0.
        record_loss_D = 0.
        record_loss_LD = 0.
        record_loss_Perceptual = 0.
        record_loss_Vec = 0.
        n_batch = 0
        net_Enc.train()
        net_Dec.train()
        net_D.train()
        net_LD.train()
        if epoch == next_expand_epoch[next_idx]:
        # if epoch>0 and epoch %1 == 0:
            alpha = 0
            iters = 0
            expand += 1
            stabilize = False
            if next_idx <= 6:
                next_idx += 1
        if expand > 6:
            alpha = 1
            expand = 6
            stabilize = True
        transform_ = transforms.Compose([transforms.Grayscale(),
                                    transforms.Resize((2**(expand+2), 2**(expand+2))),
                                    transforms.ToTensor()])
        train_dataset = TrainDataset(image_dir=normal_image_dir, train_list_dir=train_list_dir, transform=transform_)
        train_loader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True,drop_last=True)
        valid_dataset = TestDataset(normal_image_dir, valid_normal_list, abnormal_image_dir, valid_abnormal_list, transform=transform_ )
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True, drop_last=True)
        
        for i, (image, _, _) in enumerate(train_loader):
            b_size = image.size(0)
            gap_epoch = (next_expand_epoch[next_idx]-next_expand_epoch[next_idx-1])
            fade_in_iters = (gap_epoch - int(gap_epoch*0.2))* len(train_loader)
            if iters > fade_in_iters:
                stabilize = True
            alpha = min(1, iters * round(1/fade_in_iters, 6)) if stabilize==False else 1
            iters += 1
            valid_z = Variable(torch.ones(image.size(0), dtype=torch.float32), requires_grad=False).to(device)    
            fake_z = Variable(torch.zeros(image.size(0), dtype=torch.float32), requires_grad=False).to(device)
            valid_img = Variable(torch.zeros(image.size(0), dtype=torch.float32), requires_grad=False).to(device)      
            fake_img = Variable(torch.ones(image.size(0), dtype=torch.float32), requires_grad=False).to(device)
            # Set model input
            real_image = image.to(device)

            #######################################

            optimizer_D.zero_grad()
            real_Enc_out = net_Enc(real_image, expand, alpha)
            real_latent = real_Enc_out[-1]
            rec_image = net_Dec(real_latent, expand, alpha)
            pre_rec = net_D(rec_image, expand, alpha).view(-1)
            guss_latent = Variable(torch.randn(real_image.size(0), opt.latent_size)).to(device)
            gen_image = net_Dec(guss_latent, expand, alpha)
            pre_gen = net_D(gen_image, expand, alpha).view(-1)
            pre_real = net_D(real_image, expand, alpha).view(-1)
            #  optim_D
            loss_D_real = -torch.mean(pre_real)
            loss_D_rec = torch.mean(pre_rec)
            loss_D_gen = torch.mean(pre_gen)
            # loss_D_real = criterion_GAN(torch.squeeze(pre_real), valid_img)
            # loss_D_rec = criterion_GAN(torch.squeeze(pre_rec), fake_img)
            # loss_D_gen = criterion_GAN(torch.squeeze(pre_gen), fake_img)

            ### gradient penalty for D
            # eps = torch.rand(b_size, 1, 1, 1).to(device)
            # x_hat = eps * real_image.data + (1 - eps) * gen_image.detach().data
            # x_hat.requires_grad = True
            # hat_predict = net_D(x_hat, expand, alpha)
            # grad_x_hat = grad(
            #     outputs=hat_predict.sum(), inputs=x_hat, create_graph=True)[0]
            # grad_penalty = ((grad_x_hat.view(grad_x_hat.size(0), -1)
            #              .norm(2, dim=1) - 1)**2).mean()
            # loss_D = loss_D_real + 0.5 * (loss_D_rec + loss_D_gen) + 10 * grad_penalty
            loss_D = loss_D_real + loss_D_rec + loss_D_gen
            loss_D.backward()
            optimizer_D.step()
            # accumulate(D_running, net_D)
            # for p in net_D.parameters():
            #     p.data.clamp_(-opt.clamp, opt.clamp)

            optimizer_LD.zero_grad()
            guss_latent = Variable(torch.randn(real_image.size(0), opt.latent_size)).to(device)
            clf_real = net_LD(torch.squeeze(real_latent).detach())
            clf_guss = net_LD(torch.squeeze(guss_latent).detach()) 
            # optim_LD
            loss_LD_real = criterion_Vec_Clf(torch.squeeze(clf_real), valid_z)
            loss_LD_guss = criterion_Vec_Clf(torch.squeeze(clf_guss), fake_z)
            loss_LD = loss_LD_guss + loss_LD_real
            loss_LD.backward()
            optimizer_LD.step()
            # accumulate(LD_running, net_LD)

            ##################################
            optimizer_Enc.zero_grad()
            real_Enc_out = net_Enc(real_image, expand, alpha)
            rec_image = net_Dec(real_Enc_out[-1], expand, alpha)
            loss_Enc_rec = criterion_Rec(rec_image, real_image)
            rec_Enc_out = net_Enc(rec_image, expand, alpha)
            loss_Perceptual = Perceptual_loss(rec_Enc_out[:-1], real_Enc_out[:-1])
            loss_Vec = Vec_loss(rec_Enc_out[-1], real_Enc_out[-1])
            clf_real = net_LD(torch.squeeze(real_Enc_out[-1]))
            loss_Enc_Vec_Clf = criterion_Vec_Clf(torch.squeeze(clf_real), fake_z)
            loss_Enc_GAN = -torch.mean(net_D(rec_image, expand, alpha).view(-1))
            # loss_Enc_GAN = criterion_GAN(net_D(rec_image, expand, alpha).view(-1), valid_img)
            loss_Enc = 20 *loss_Enc_rec + loss_Enc_Vec_Clf +  4 * loss_Enc_GAN + loss_Perceptual + 8 * loss_Vec
            # loss_Enc = 0.5 * loss_Enc_rec + loss_Enc_Vec_Clf + loss_Enc_GAN
            loss_Enc.backward()
            optimizer_Enc.step()
            # accumulate(Enc_running, net_Enc)
            
            optimizer_Dec.zero_grad()
            real_Enc_out = net_Enc(real_image, expand, alpha)
            rec_image = net_Dec(real_Enc_out[-1], expand, alpha)
            loss_Dec_rec = criterion_Rec(rec_image, real_image)
            rec_Enc_out = net_Enc(rec_image, expand, alpha)
            loss_Perceptual = Perceptual_loss(rec_Enc_out[:-1], real_Enc_out[:-1])
            loss_Vec = Vec_loss(rec_Enc_out[-1], real_Enc_out[-1])
            guss_latent = Variable(torch.randn(real_image.size(0), opt.latent_size)).to(device)
            gen_image = net_Dec(guss_latent, expand, alpha)
            pre_rec = net_D(rec_image, expand, alpha).view(-1)
            pre_gen = net_D(gen_image, expand, alpha).view(-1)
            loss_Dec_GAN_rec = -torch.mean(pre_rec)
            loss_Dec_GAN_gen = -torch.mean(pre_gen)
            # loss_Dec_GAN_rec = criterion_GAN(torch.squeeze(pre_rec), valid_img)
            # loss_Dec_GAN_gen = criterion_GAN(torch.squeeze(pre_gen), valid_img)
            loss_Dec = 20 * loss_Dec_rec + 4 * (loss_Dec_GAN_gen + loss_Dec_GAN_rec)+ loss_Perceptual + 8 * loss_Vec

            # loss_Enc.backward(retain_graph=True)
            loss_Dec.backward()
            # optimizer_Enc.step()
            optimizer_Dec.step()
            # accumulate(Dec_running, net_Dec)

            record_loss_Enc += loss_Enc.detach().cpu().numpy()
            record_loss_Enc_Rec += loss_Enc_rec.detach().cpu().numpy()
            record_loss_Enc_Vec_Clf += loss_Enc_Vec_Clf.detach().cpu().numpy()
            record_loss_Dec += loss_Dec.detach().cpu().numpy()
            record_loss_Dec_Rec += loss_Dec_rec.detach().cpu().numpy()
            record_loss_Dec_GAN_Gen += loss_Dec_GAN_gen.detach().cpu().numpy()
            record_loss_Dec_GAN_Rec += loss_Dec_GAN_rec.detach().cpu().numpy()
            record_loss_D += loss_D.detach().cpu().numpy()
            record_loss_LD += loss_LD.detach().cpu().numpy()
            record_loss_Perceptual += loss_Perceptual.detach().cpu().numpy()
            record_loss_Vec += loss_Vec.detach().cpu().numpy()
            n_batch += 1
            print("  ")

            logger.log({'loss_Enc': loss_Enc, 'loss_Enc_rec': loss_Enc_rec, 'loss_Enc_Vec_Clf':loss_Enc_Vec_Clf, 
            'loss_Dec': loss_Dec, 'loss_Dec_rec': loss_Dec_rec, 'loss_Dec_GAN_Gen':loss_Dec_GAN_gen, 'loss_Dec_GAN_Rec':loss_Dec_GAN_rec, 
            'loss_D': loss_D, 'loss_LD':loss_LD, 'loss_Perceptual':loss_Perceptual, 'loss_Vec':loss_Vec})

        with torch.no_grad():
            if (epoch + 1) % 10 == 0:
                Enc_out = net_Enc(real_image, expand, alpha)
                guss_vec = Variable(torch.randn(real_image.size(0), opt.latent_size)).to(device)
                rec_image = net_Dec(Enc_out[-1], expand, alpha)
                gen_image = net_Dec(guss_vec, expand, alpha)
            
                vutils.save_image(torch.cat((real_image, rec_image, torch.abs(real_image - rec_image), gen_image), dim=0),
                            './output/{0}_Crop/{1}_output.png'.format(TIME, epoch))
        with SummaryWriter('./runs/{0}_Crop'.format(TIME)) as writer:
            writer.add_scalar('scalar/loss_Enc', record_loss_Enc / n_batch, epoch)
            writer.add_scalar('scalar/loss_Enc_Rec', record_loss_Enc_Rec / n_batch, epoch)
            writer.add_scalar('scalar/loss_Enc_Vec_Clf', record_loss_Enc_Vec_Clf / n_batch, epoch)
            writer.add_scalar('scalar/loss_Dec', record_loss_Dec / n_batch, epoch)
            writer.add_scalar('scalar/loss_Dec_Rec', record_loss_Dec_Rec / n_batch, epoch)
            writer.add_scalar('scalar/loss_Dec_GAN_Gen', record_loss_Dec_GAN_Gen / n_batch, epoch)
            writer.add_scalar('scalar/loss_Dec_GAN_Rec', record_loss_Dec_GAN_Rec / n_batch, epoch)
            writer.add_scalar('scalar/loss_D', record_loss_D / n_batch, epoch)
            writer.add_scalar('scalar/loss_LD', record_loss_LD / n_batch, epoch)
            writer.add_scalar('scalar/loss_Perceptual', record_loss_Perceptual / n_batch, epoch)
            writer.add_scalar('scalar/loss_Vec', record_loss_Vec / n_batch, epoch)
            writer.close()
        
        # Update learning rates
        # lr_scheduler_Enc.step()
        # lr_scheduler_Dec.step()
        # lr_scheduler_D.step()
        # lr_scheduler_LD.step()


        with torch.no_grad():
            Anomaly_score = []
            true_label = []
            pred_label = []
            net_Enc.eval()
            net_Dec.eval()
            for i, (data, label, _) in enumerate(valid_loader):
                real_data = data.to(device)
                v_real_out = net_Enc(real_data, expand, alpha)
                v_rec = net_Dec(v_real_out[-1], expand, alpha)
                vutils.save_image(torch.cat((real_data, v_rec), dim=0),'./output.png')
                Anomaly_score.append(torch.squeeze(criterion_Rec(real_data, v_rec)).item())
                true_label.extend(label)
            th = np.quantile(Anomaly_score, 1 - opt.nu)
            pred_label = [1 if Anomaly_score[i] >= th else 0 for i in range(len(Anomaly_score))]
            matrix = metrics.confusion_matrix(true_label, pred_label)
            sensitivity = matrix[1, 1] / (matrix[1, 0] + matrix[1, 1])
            specificity = matrix[0, 0] / (matrix[0, 0] + matrix[0, 1])
            AUC = metrics.roc_auc_score(true_label, Anomaly_score)
            ACC = metrics.accuracy_score(true_label, pred_label)
            print(matrix)
            print("AUC: {0}, ACC:{1}, sensitivity:{2}, specificity:{3}".format(AUC, ACC, sensitivity, specificity))
            
            with SummaryWriter('./runs/{0}_Crop'.format(TIME)) as writer:
                writer.add_scalar('scalar/AUC', AUC, epoch)
                writer.close()

        if alpha == 1:
            torch.save(net_Enc.state_dict(), './model/{0}_Crop/net_Enc_epoch{1}_expand{2}.pth'.format(TIME, epoch, expand))
            torch.save(net_Dec.state_dict(), './model/{0}_Crop/net_Dec_epoch{1}_expand{2}.pth'.format(TIME, epoch, expand))
            torch.save(net_D.state_dict(), './model/{0}_Crop/net_D_epoch{1}_expand{2}.pth'.format(TIME, epoch, expand))
            torch.save(net_LD.state_dict(), './model/{0}_Crop/net_LD_epoch{1}_expand{2}.pth'.format(TIME, epoch, expand))