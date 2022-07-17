import random
from os import path

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.optim import lr_scheduler

from utility.vgg_network_with_top import VGG, Model
from utility.utility import load_mono_images


def make_path_list():
    DATA_CSV_PATH = './data.csv'
    data_df = pd.read_csv(DATA_CSV_PATH)
    train_list = data_df[data_df['type'] == 'train']['dirname'].to_list()
    val_list = data_df[data_df['type'] == 'valid']['dirname'].to_list()
    test_list = data_df[data_df['type'] == 'test']['dirname'].to_list()

    char_list = [chr(c_code) for c_code in range(ord('A'), ord('Z')+1)]
    # train_list = [[path.join('../font2img/image', d, char+'.png') for char in char_list] for d in ['AbhayaLibre-ExtraBold']]
    train_list = [[path.join('../font2img/image', d, char+'.png') for char in char_list] for d in train_list]
    val_list = [[path.join('../font2img/image', d, char+'.png') for char in char_list] for d in val_list]
    test_list = [[path.join('../font2img/image', d, char+'.png') for char in char_list] for d in test_list]

    train_list = sum(train_list, [])
    val_list = sum(val_list, [])
    test_list = sum(test_list, [])

    print('TRAIN SIZE: ', len(train_list))
    print('VAL SIZE: ', len(val_list))
    print('TEST SIZE: ', len(test_list))

    return train_list, val_list, test_list


class LoadDataset(data.Dataset):
    def __init__(self, file_list, device, img_size):
        self.file_list = file_list
        self.device = device
        self.img_size = img_size

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        invert = 1
        filepath = self.file_list[index]
        img = load_mono_images(filepath, self.img_size, self.device, invert)
        alphabet = path.basename(filepath)[:-4]
        label = ord(alphabet) - ord('A')

        return img, label


def make_dataloader(device):
    img_size = 256
    batch_size = 128

    train_path, val_path, test_path = make_path_list()

    train_dataset = LoadDataset(train_path, device, img_size)
    val_dataset = LoadDataset(val_path, device, img_size)
    test_dataset = LoadDataset(test_path, device, img_size)

    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}


def train(device):
    layers = ['fc3']
    epochs = 100
    n_classes = 26
    vgg_filename = 'resnet18.pth'
    device = 'cuda:0' if torch.cuda.is_available else 'cpu'

    # vgg = VGG(n_classes)

    # vgg = Model()

    # vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', init_weights=False)

    from torchvision import models
    # vgg = models.vgg19()
    # vgg.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
    # vgg.avgpool = nn.AdaptiveAvgPool2d((8, 8))
    # vgg.classifier[0] = nn.Linear(512*8*8, 4096)
    # vgg.classifier[6] = nn.Linear(4096, 26)

    vgg = models.resnet18(pretrained=False)
    vgg.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
    vgg.fc = nn.Linear(512, 26)

    optimizer = optim.Adam(vgg.parameters(), lr=0.001)
    # optimizer = optim.SGD(vgg.parameters(), lr=0.001, momentum=0.9)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    dataloader = make_dataloader(device)

    vgg = vgg.to(device)

    best_val_loss = 1e18
    epoch_loss_dic = {}
    epoch_acc_dic = {}

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch+1, epochs))
        print('-------------')

        for phase in ['train', 'val']:
            if phase == 'train':
                vgg.train()
            else:
                vgg.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            for inputs, labels in tqdm(dataloader[phase]):
                # inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # outputs = vgg(inputs, layers)[0]
                    outputs = vgg(inputs)
                    # print(phase, 'fc3', outputs)

                    _, preds = torch.max(outputs, 1)
                    # print(phase, 'preds', preds, 'labels', labels)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                epoch_loss += loss.item() * inputs.size(0)
                epoch_corrects += torch.sum(preds == labels.data)

            epoch_loss = epoch_loss / len(dataloader[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloader[phase].dataset)

            epoch_loss_dic.setdefault(phase, []).append(epoch_loss)
            epoch_acc_dic.setdefault(phase, []).append(epoch_acc.detach().item())

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                torch.save(vgg.state_dict(), vgg_filename)
                print('model is saved !!!!')

    return epoch_loss_dic, epoch_acc_dic


def save_history(his, title):
    fig = plt.figure()
    plt.plot(his['train'], label="{} for training".format(title))
    plt.plot(his['val'], label="{} for validation".format(title))

    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel(title)
    plt.legend(loc='best')

    fig.savefig(title + '.png')

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("using ", device)

    history_loss, history_acc = train(device)

    save_history(history_loss, 'loss')
    save_history(history_acc, 'acc')

if __name__ == '__main__':
    SEED = 7777
    torch.cuda.manual_seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    main()