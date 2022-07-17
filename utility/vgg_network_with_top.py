import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG(nn.Module):
    def __init__(self, n_classes, pool='max'):
        super(VGG, self).__init__()
        # feature extractor
        self.conv1_1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

        # avgpool
        self.avgpool = nn.AdaptiveAvgPool2d((8, 8))

        # FC
        self.fc1 = nn.Linear(512*8*8, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, n_classes)

        self.dropout = nn.Dropout(p=0.5)

        # Initalization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, out_keys):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])

        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])

        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])

        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])

        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])

        out['p6'] = self.avgpool(out['p5'])

        flattened_out = torch.flatten(out['p6'], 1)
        out['fc1'] = F.relu(self.fc1(flattened_out))
        out_fc1 = self.dropout(out['fc1'])
        out['fc2'] = F.relu(self.fc2(out_fc1))
        out_fc2 = self.dropout(out['fc2'])
        out['fc3'] = self.fc3(out_fc2)

        return [out[key] for key in out_keys]


class Model(nn.Module):

    def __init__(self):
        # スーパークラス（Module クラス）の初期化メソッドを実行
        super().__init__()

        self.c0 = nn.Conv2d(in_channels=1,    # 入力は3チャネル
                            out_channels=16,  # 出力は16チャネル
                            kernel_size=3,    # カーネルサイズは3*3
                            stride=2,         # 1pix飛ばしでカーネルを移動
                            padding=1)        # 画像の外側1pixを埋める

        self.c1 = nn.Conv2d(in_channels=16,   # 入力は16チャネル
                            out_channels=32,  # 出力は32チャネル
                            kernel_size=3,    # カーネルサイズは3*3
                            stride=2,         # 1pix飛ばしでカーネルを移動
                            padding=1)        # 画像の外側1pixを埋める

        self.c2 = nn.Conv2d(in_channels=32,   # 入力は32チャネル
                            out_channels=64,  # 出力は64チャネル
                            kernel_size=3,    # カーネルサイズは3*3
                            stride=2,         # 1pix飛ばしでカーネルを移動
                            padding=1)        # 画像の外側1pixを埋める

        self.bn0 = nn.BatchNorm2d(num_features=16)   # c0用のバッチ正則化
        self.bn1 = nn.BatchNorm2d(num_features=32)   # c1用のバッチ正則化
        self.bn2 = nn.BatchNorm2d(num_features=64)   # c2用のバッチ正則化

        self.fc = nn.Linear(in_features=64 * 32 * 32,   # 入力サイズ
                            out_features=26)             # 各クラスに対応する4次元のベクトルに変換

    def forward(self, x): # 入力から出力を計算するメソッドを定義
        h = F.relu(self.c0(x))
        h = F.relu(self.c1(h))
        h = F.relu(self.c2(h))
        h = h.view(-1, 64 * 32 * 32)
        y = self.fc(h)     # 全結合層
        return y