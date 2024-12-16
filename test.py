#!/usr/bin/python3
import sys
sys.path.append('../')
import torch
import os
import time
import random
import numpy as np
import pandas as pd
import argparse
import itertools
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from my_dataset import TrainDataset, TestDataset
from network import Encoder, Decoder, Discriminator, Latent_Dis
import torchvision.utils as vutils
# from sklearn import preprocessing
from torch.autograd import Variable
from utils import Logger, LambdaLR, ReplayBuffer
from tensorboardX import SummaryWriter
from sklearn import metrics, manifold
import matplotlib.pyplot as plt

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=1000, help='number of epochs of training')
    parser.add_argument('--batchSize', type=int, default=10, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.0005, help='initial learning rate')
    parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
    parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
    parser.add_argument('--latent_size', type=int, default=128, help='latent_size')
    parser.add_argument('--nu', type=float, default=0.5, help='proportion of allowed anomaly')
    parser.add_argument('--n_critic', type=int, default=1)
    parser.add_argument('--clamp', type=float, default=0.01)
    opt = parser.parse_args()
    print(opt)
    return opt

def getImageNameList(path):
    select_image = []
    file = open(path, encoding='utf-8')
    for line in file:
        select_image.append(line.strip('\n'))
    return select_image


def getSelectSet(dataset, select_list):
    index_list = []
    for idx, row in dataset.iterrows():
        if row['image_name'] not in select_list:
            index_list.append(idx)

    return dataset.drop(index_list)

if __name__ =="__main__":

    opt = parse_args()


    # Networks
    net_Enc = Encoder(opt.input_nc, opt.latent_size)
    net_Enc = torch.load("./model/20230401_211608_Crop/net_Enc_epoch978_expand6.pth")
    net_Enc.to(device)
    net_Dec = Decoder(opt.output_nc, opt.latent_size)
    net_Dec = torch.load("./model/20230401_211608_Crop/net_Dec_epoch978_expand6.pth")
    net_Dec.to(device)


    criterion_Rec = torch.nn.L1Loss()
    expand=6
    alpha=1
    
    # Dataset loader
    transform_ = transforms.Compose([transforms.Grayscale(),
                                    transforms.Resize((2**(expand+2), 2**(expand+2))),
                                    transforms.ToTensor()])
    normal_image_dir = '/home/niuyi/dataset/original_dataset/CXR_ipmi/normal_crop'
    abnormal_image_dir = '/home/niuyi/dataset/original_dataset/CXR_ipmi/abnormal_crop'
    # test_normal_list = './split_dataset/valid_normal.txt'
    # test_abnormal_list = './split_dataset/valid_abnormal.txt'
    test_normal_list = './split_dataset/test_normal.txt'
    test_abnormal_list = './split_dataset/test_abnormal.txt'
    test_dataset = TestDataset(normal_image_dir, test_normal_list, abnormal_image_dir, test_abnormal_list, transform=transform_ )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=True)

    output = {'image_name':[], 'true_label':[], 'anomaly_score':[]}
    features = []
    with torch.no_grad():
        Anomaly_score = []
        true_label = []
        pred_label = []
        net_Enc.eval()
        net_Dec.eval()
        for i, (data, label, image_name) in enumerate(test_loader):
            real_data = data.to(device)
            b, c, _, _ = real_data.size()
            real_out = net_Enc(real_data, expand, alpha)
            rec_image = net_Dec(real_out[-1], expand, alpha)
            # vutils.save_image(real_data,'./output/20230401_211608_Crop/epoch_978_res/real_img/{0}.png'.format(image_name))
            # vutils.save_image(rec_image, './output/20230401_211608_Crop/epoch_978_res/rec_img/{0}.png'.format(image_name))
            
            anomaly_score = torch.squeeze(criterion_Rec(real_data, rec_image)).item()
            Anomaly_score.append(anomaly_score)
            true_label.extend(label)

            output['image_name'].extend(image_name)
            output['true_label'].extend(label.detach().numpy())
            output['anomaly_score'].append(anomaly_score)
            features.append(torch.squeeze(real_out[-1]).detach().cpu().numpy())

        df_outfea = pd.DataFrame(np.array(features))
        df_outfea['image_name']=output['image_name']
        df_outfea['true_label']=output['true_label']
        df_outfea.to_csv('./output/20230401_211608_Crop/epoch_978_res/test_features.csv', index=False)
        
        df_output = pd.DataFrame(output)
        df_output.to_csv('./output/20230314_220603_Crop/epoch_978_res/test_res.csv', index=False)

        th = np.quantile(Anomaly_score, 1 - opt.nu)
        pred_label = [1 if Anomaly_score[i] >= th else 0 for i in range(len(Anomaly_score))]
        matrix = metrics.confusion_matrix(true_label, pred_label)
        sensitivity = matrix[1, 1] / (matrix[1, 0] + matrix[1, 1])
        specificity = matrix[0, 0] / (matrix[0, 0] + matrix[0, 1])
        AUC = metrics.roc_auc_score(true_label, Anomaly_score)
        ACC = metrics.accuracy_score(true_label, pred_label)
        print(matrix)
        print("AUC: {0}, ACC:{1}, sensitivity:{2}, specificity:{3}".format(AUC, ACC, sensitivity, specificity))

    # df_test_res = pd.read_csv('./output/20230401_211608_Crop/epoch_978_res/test_res.csv', header=0)
    # df_test_normal = df_test_res.groupby('true_label').get_group(0.0)
    # df_test_abnormal = df_test_res.groupby('true_label').get_group(1.0)

    # test_normal_root = './split_dataset/test_normal.txt'
    # test_abnormal_root = './split_dataset/test_abnormal.txt'

    # test_normal_list = getImageNameList(test_normal_root)
    # test_abnormal_list = getImageNameList(test_abnormal_root)
    
    # all_res = {'auc': [], 'sensitivity': [], 'specificity': []}
    # for i in range(50):
    #     random.shuffle(test_normal_list)
    #     random.shuffle(test_abnormal_list)
    #     cur_normal_list = test_normal_list[:500]
    #     cur_abnormal_list = test_abnormal_list[:25]

    #     df_select_normal = getSelectSet(df_test_normal, cur_normal_list)
    #     df_select_abnormal = getSelectSet(df_test_abnormal, cur_abnormal_list)

    #     df_select_test = pd.concat((df_select_normal, df_select_abnormal), axis=0)
    #     true_label = df_select_test['true_label'].values
    #     Anomaly_score = df_select_test['anomaly_score'].values
    #     th = np.quantile(Anomaly_score, 1 - 0.5)
    #     pred_label = [1 if Anomaly_score[i] >= th else 0 for i in range(len(Anomaly_score))]
    #     matrix = metrics.confusion_matrix(true_label, pred_label)
    #     sensitivity = matrix[1, 1] / (matrix[1, 0] + matrix[1, 1])
    #     specificity = matrix[0, 0] / (matrix[0, 0] + matrix[0, 1])
    #     AUC = metrics.roc_auc_score(true_label, Anomaly_score)
    #     ACC = metrics.accuracy_score(true_label, pred_label)
    #     print(matrix)
    #     print("AUC: {0}, ACC:{1}, sensitivity:{2}, specificity:{3}".format(AUC, ACC, sensitivity, specificity))

    #     all_res['auc'].append(AUC)
    #     all_res['sensitivity'].append(sensitivity)
    #     all_res['specificity'].append(specificity)
    # df_all_res = pd.DataFrame(all_res)
    # print(df_all_res.describe())
