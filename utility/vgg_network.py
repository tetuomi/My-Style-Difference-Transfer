import torch
import torch.nn as nn
import torch.nn.functional as F

class Normalization(nn.Module):
    def __init__(self, mean, std, device):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1).to(device)
        self.std = torch.tensor(std).view(-1, 1, 1).to(device)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

# VGG definition that conveniently let's you grab the outputs from any layer
class VGG(nn.Module):
    def __init__(self, device):
        super(VGG, self).__init__()
        # data augmentation
        self.prep = Normalization(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], device=device)
        #vgg modules
        self.conv_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu_1 = nn.ReLU(inplace=False)
        self.conv_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu_2 = nn.ReLU(inplace=False)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu_3 = nn.ReLU(inplace=False)
        self.conv_4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu_4 = nn.ReLU(inplace=False)
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu_5 = nn.ReLU(inplace=False)
        self.conv_6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu_6 = nn.ReLU(inplace=False)
        self.conv_7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu_7 = nn.ReLU(inplace=False)
        self.conv_8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu_8 = nn.ReLU(inplace=False)
        self.pool_8 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_9 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu_9 = nn.ReLU(inplace=False)
        self.conv_10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu_10 = nn.ReLU(inplace=False)
        self.conv_11 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu_11 = nn.ReLU(inplace=False)
        self.conv_12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu_12 = nn.ReLU(inplace=False)
        self.pool_12 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_13 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu_13 = nn.ReLU(inplace=False)
        self.conv_14 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu_14 = nn.ReLU(inplace=False)
        self.conv_15 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu_15 = nn.ReLU(inplace=False)
        self.conv_16 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu_16 = nn.ReLU(inplace=False)
        self.pool_16 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, out_keys):
        out = {}
        prep_x = self.prep(x)
        out['r11'] = self.relu_1(self.conv_1(prep_x))
        out['r12'] = self.relu_2(self.conv_2(out['r11']))
        out['p1'] = self.pool_2(out['r12'])

        out['r21'] = self.relu_3(self.conv_3(out['p1']))
        out['r22'] = self.relu_4(self.conv_4(out['r21']))
        out['p2'] = self.pool_4(out['r22'])

        out['r31'] = self.relu_5(self.conv_5(out['p2']))
        out['r32'] = self.relu_6(self.conv_6(out['r31']))
        out['r33'] = self.relu_7(self.conv_7(out['r32']))
        out['r34'] = self.relu_8(self.conv_8(out['r33']))

        out['p3'] = self.pool_8(out['r34'])
        out['r41'] = self.relu_9(self.conv_9(out['p3']))
        out['r42'] = self.relu_10(self.conv_10(out['r41']))
        out['r43'] = self.relu_11(self.conv_11(out['r42']))
        out['r44'] = self.relu_12(self.conv_12(out['r43']))
        out['p4'] = self.pool_12(out['r44'])

        out['r51'] = self.relu_13(self.conv_13(out['p4']))
        out['r52'] = self.relu_14(self.conv_14(out['r51']))
        out['r53'] = self.relu_15(self.conv_15(out['r52']))
        out['r54'] = self.relu_16(self.conv_16(out['r53']))
        out['p5'] = self.pool_16(out['r54'])

        return [out[key] for key in out_keys]