import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
# from einops import rearrange, reduce, repeat


class DISENTANGLE_MODEL(nn.Module):
    def __init__(self, zdim, ch_num):
        super().__init__()
        self.zdim = zdim
        self.ch_num = ch_num
        hidden_units = 512

        self.encoder = nn.Sequential(
                    nn.Conv2d(1, 32, 4, 2, 1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 128, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, 4, 1, 1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
        )
        self.fc_c = nn.Sequential(
                    nn.Linear(256*7*7, hidden_units),
                    # nn.ReLU(inplace=True),
                    nn.Dropout(p=0.5),
                    nn.Linear(hidden_units, int(zdim/2)),
                    # nn.ReLU(inplace=True)
                    )
        self.fc_f = nn.Sequential(
                    nn.Linear(256*7*7, hidden_units),
                    # nn.ReLU(inplace=True),
                    nn.Dropout(p=0.5),
                    nn.Linear(hidden_units, int(zdim/2)),
                    # nn.ReLU(inplace=True)
                    )
        self.classifier_c = nn.Sequential(
                            nn.Linear(int(zdim/2),ch_num),
#                             nn.Softmax(dim=1)
        )
        self.upsample = nn.Sequential(
            nn.Linear(zdim, 256*7*7),
            nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, layer_key):
        # # x = nn.Sigmoid()(x)
        # c1 = 2.0 # 傾き
        # c2 = 0.5 # 変曲点
        # x = 1.0 / (1 + torch.exp(-c1*(x - c2)))

        middle_outputs = {}

        conv_cnt = 0
        for layer in self.encoder.children():
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                conv_cnt += 1
            elif isinstance(layer, nn.ReLU):
                key = 'conv{}'.format(conv_cnt)
                middle_outputs[key] = x

        z = x.view(x.shape[0], -1)
        z_c = self.fc_c(z)
        z_f = self.fc_f(z)
        # output_c = self.classifier_c(z_c)

        return [middle_outputs[k] for k in layer_key], z_c, z_f
