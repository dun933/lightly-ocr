import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.asrn import ASRN
from modules.attention import Attention
from modules.morn import MORN
from modules.resnet import ResNet
from modules.sequence import BidirectionalLSTM
from modules.transform import TPS_STN
from modules.vgg_bn import init_weights, vgg16_bn


class UpConv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(UpConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1), nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
                                  nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class VGG_UNet(nn.Module):
    def __init__(self, pretrained=False, freeze=False):
        super(VGG_UNet, self).__init__()
        """ Base network """
        self.basenet = vgg16_bn(pretrained, freeze)
        """ U network """
        self.upconv1 = UpConv(1024, 512, 256)
        self.upconv2 = UpConv(512, 256, 128)
        self.upconv3 = UpConv(256, 128, 64)
        self.upconv4 = UpConv(128, 64, 32)

        num_class = 2
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_class, kernel_size=1),
        )

        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_cls.modules())

    def forward(self, x):
        # basenet
        sources = self.basenet(x)

        # UNet
        y = torch.cat([sources[0], sources[1]], dim=1)
        y = self.upconv1(y)

        y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[2]], dim=1)
        y = self.upconv2(y)

        y = F.interpolate(y, size=sources[3].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[3]], dim=1)
        y = self.upconv3(y)

        y = F.interpolate(y, size=sources[4].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[4]], dim=1)
        feature = self.upconv4(y)

        y = self.conv_cls(feature)

        return y.permute(0, 2, 3, 1), feature


class MORAN(nn.Module):
    def __init__(self, nc, num_classes, nh, height, width, bidirectional=False, input_data_type='torch.cuda.FloatTensor', max_batch=256):
        super(MORAN, self).__init__()
        self.MORN = MORN(nc, height, width, input_data_type, max_batch)
        self.ASRN = ASRN(height, nc, num_classes, nh, bidirectional)

    def forward(self, x, length, text, text_rev, test=False, debug=False):
        if debug:
            x_rectified, img = self.MORN(x, test, debug=debug)
            preds = self.ASRN(x_rectified, length, text, text_rev, test)
            return preds, img
        else:
            x_rectified = self.MORN(x, test, debug=debug)
            preds = self.ASRN(x_rectified, length, text, text_rev, test)
            return preds


class CRNN(nn.Module):
    # TPS - ResNet - biLSTM - Attn/CTC
    def __init__(self, config):
        super(CRNN, self).__init__()
        self.config = config
        if config['transform'] == 'TPS':
            self.transform = TPS_STN(F=config['num_fiducial'],
                                     im_size=(config['height'], config['width']),
                                     im_rectified=(config['height'], config['width']),
                                     num_channels=config['input_channel'])
        else:
            print('not using TPS')

        if config['backbone'] == 'ResNet':
            self.backbone = ResNet(config['input_channel'], config['output_channel'])
        else:
            raise Exception('No backbone specified')
        self.backbone_outputs = config['output_channel']
        self.adaptivepool = nn.AdaptiveAvgPool2d((None, 1))

        if config['sequence'] == 'biLSTM':
            self.sequence = nn.Sequential(BidirectionalLSTM(self.backbone_outputs, config['hidden_size'], config['hidden_size']),
                                          BidirectionalLSTM(config['hidden_size'], config['hidden_size'], config['hidden_size']))
            self.sequence_outputs = config['hidden_size']

        if config['prediction'] == 'CTC':
            self.prediction = nn.Linear(self.sequence_outputs, config['num_classes'])
        elif config['prediction'] == 'Attention':
            self.prediction = Attention(self.sequence_outputs, config['hidden_size'], config['num_classes'])
        else:
            raise Exception('prediction needs to be either CTC or attention-based sequence prediction')

    def forward(self, inputs, text, training=True):
        if not self.config['transform'] == 'None':
            inputs = self.transform(inputs)

        x = self.backbone(inputs)
        x = self.adaptivepool(x.permute(0, 3, 1, 2))  # [b,c,h,w]-> [b,w,c,h]
        x = x.squeeze(3)

        if self.config['sequence'] == 'biLSTM':
            x = self.sequence(x)
        if self.config['prediction'] == 'CTC':
            pred = self.prediction(x.contiguous())
        else:
            pred = self.prediction(x.contiguous(), text, training, batch_max_len=self.config['batch_max_len'])
        return pred
