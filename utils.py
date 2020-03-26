""" helper function

author baiyu
"""

import sys

import torch
from torch.optim.lr_scheduler import _LRScheduler
from model import ResNet_Modified
from conf import settings

#from dataset import CIFAR100Train, CIFAR100Test

def get_network(args, num_classes=1000, use_gpu=True):
    """ return given network
    """
    if args.net == 'resnet50':
        net = ResNet_Modified([3, 4, 6, 3], settings.FC_CHANNEL, num_classes, frozen_stages=settings.FREE_STAGES)
    elif args.net == 'resnet101':
        net = ResNet_Modified([3, 4, 23, 3], settings.FC_CHANNEL, num_classes, frozen_stages=settings.FREE_STAGES)
    elif args.net == 'resnext50':
        net = ResNet_Modified([3, 4, 6, 3], settings.FC_CHANNEL, num_classes, 
                              frozen_stages=settings.FREE_STAGES, groups=32, width_per_group=4)
    elif args.net == 'resnext101':
        net = ResNet_Modified([3, 4, 23, 3], settings.FC_CHANNEL, num_classes, 
                              frozen_stages=settings.FREE_STAGES, groups=32, width_per_group=8)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()
    
    if use_gpu:
        net = net.cuda()
    
    #load the pretrain_model except fc weights and bias 
    if args.pretrain_path is not None:
        state_dict = torch.load(args.pretrain_path)
        if num_classes != 1000:
            ded_keys = []
            for key in state_dict.keys():
                if 'fc' in key:
                    ded_keys.append(key)
            for key in ded_keys:
                del state_dict[key]
        net.load_state_dict(state_dict, strict=False)
        
    return net

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
