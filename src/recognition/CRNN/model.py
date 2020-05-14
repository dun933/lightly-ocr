import torch
import torch.nn as nn
from modules.transform import TPS_STN
from modules.backbone import ResNet
from modules.sequence import Attention, biLSTM



class CRNN(nn.Module):
    # TPS - ResNet - biLSTM - Attn/CTC
    def __init__(self,config):
        self.config = config
        if config['transform']=='TPS':
            self.transform = TPS_STN(F=config['num_fiducial'], I_size=(config['height'], config['width']),
                                     I_r_size=(config['height'], config['width']), num_channels=config['input_channel'])
        else:
            print('not using TPS')

        if config['backbone']=='ResNet':
            self.backbone = ResNet(config['input_channel'], config['output_channel'])
        else:
            raise Exception(f'No backbone specified')
        self.backbone_outputs = config['output_channel']
        self.adaptivepool = nn.AdaptiveAvgPool2d((None, 1))

        if config['sequence']=='biLSTM':
            self.sequence = nn.Sequential(biLSTM(self.backbone_outputs, config['hidden_size'], config['hidden_size']),
                                          biLSTM(config['hidden_size'], config['hidden_size'], config['hidden_size']))
            self.sequence_outputs = config['hidden_size']

        if config['prediction'] == 'CTC':
            self.prediction = nn.Linear(self.sequence_outputs, config['num_classes'])
        elif config['prediction'] == 'Attention':
            self.prediction = Attention(self.sequence_outputs, config['hidden_size'], config['num_classes'])
        else:
            raise Exception('prediction needs to be either CTC or attention based')

    def forward(self, inputs, text, training=True):
        if not self.config['transform'] == 'None':
            inputs = self.transform(inputs)

        x = self.backbone(inputs)
        x = self.adaptivepool(x.permute(0,3,1,2)) # [b,c,h,w]-> [b,w,c,h]
        x = x.squeeze(3)

        if self.config['sequence']=='biLSTM':
            x = self.sequence(x)
        if self.config['prediction']=='CTC':
            pred = self.prediction(x.contiguous())
        else:
            pred = self.prediction(x.contiguous(), text, training, batch_max_len=self.config['batch_max_len'])
        return pred
