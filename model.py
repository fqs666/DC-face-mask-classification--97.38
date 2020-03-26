# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 12:42:00 2020

@author: fqsfyq
"""

from torchvision.models.resnet import Bottleneck
from torchvision.models.resnet import ResNet
from torch import nn
import torch

class ResNet_Modified(ResNet):
    """
    modified version of resnet in torchvision, the modified part is that 
    adding extra mutilple FC layer which followed bn and dropout layer to the backbone.
    
    Args:
        layers(list): num block of each stage
        num_classes(int): number of class
        feature_only(bool): just ouput the feature which was just computed by backbone, not forward by fc layer
        frozen_stages(int): number of stage to be freezed
        dp_rate(float): the dropout rate of the dp layer
    """
    def __init__(self, layers, fc_channel, num_classes, feature_only=False, frozen_stages=-1, dp_rate=0.2, **kwargs):
        super(ResNet_Modified, self).__init__(Bottleneck, layers, num_classes=num_classes, **kwargs)
        self.__delattr__('fc') #del default fc layer
        in_channel = 2048
        self.frozen_stages = frozen_stages
        fc_channel += [num_classes]
        self.fc_channel = fc_channel
        self.feature_only = feature_only
        for i in range(len(fc_channel)):
            self.add_module('fc'+str(i+1), nn.Linear(in_channel, fc_channel[i]))
            in_channel = fc_channel[i]
        self.relu = nn.ReLU(inplace=True)
        for i in range(len(self.fc_channel) - 1):
            self.add_module('fc_bn'+str(i+1), nn.BatchNorm1d(self.fc_channel[i]))
            self.add_module('fc_dp'+str(i+1), nn.Dropout(p=dp_rate))
            
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.feature_only:
            return x
        for i in range(len(self.fc_channel)):
            fc = self.__getattr__('fc'+str(i+1))
            if i < len(self.fc_channel) - 1:
                bn = self.__getattr__('fc_bn'+str(i+1))
                dp = self.__getattr__('fc_dp'+str(i+1))
                x = fc(x)
                x = self.relu(bn(x))
                x = dp(x)
            else:
                x = fc(x)
        return x
    
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.bn1.eval()
            for m in [self.conv1, self.bn1]:
                for param in m.parameters():
                    param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
    
    #load pretrain model except fc layer
    def load_pretrain_model(self, file):    
        state_dict = torch.load(file)
        for key in state_dict.keys():
            if 'fc' in key:
                del state_dict[key]
        self.load_state_dict(state_dict, strict=False)
