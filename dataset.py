# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 16:54:29 2020

@author: fqsfyq
"""
import os
import random
import numpy as np

import transform
from transform import Resize
from transform import Compose

loading = transform.LoadImgFile
resize = Resize([224, 320, 416, 512])

#noise
AdditiveGaussianNoise = transform.AdditiveGaussianNoise(scale_limit=(10, 30))
AdditiveLaplaceNoise = transform.AdditiveLaplaceNoise()
AdditivePoissonNoise = transform.AdditivePoissonNoise()
#cotout
CoarseDropout = transform.CoarseDropout()
CoarseSaltAndPepper = transform.CoarseSaltAndPepper()
Cutout = transform.Cutout()
#color
HueSat = transform.AddToHueAndSaturation()
DropChannel = transform.DropChannel()
#geometry
elastic = transform.ElasticTransformation(sigma=9)
Rotate_Shear = transform.Rotate_Shear()
#value
add = transform.Add()
mul = transform.Mul()
Solarize = transform.Solarize()
JpegCompression = transform.JpegCompression()

#blend
BlendAlpha = transform.BlendAlpha()
BlendAlphaElementwise = transform.BlendAlphaElementwise()
#blur
GaussianBlur = transform.GaussianBlur()
AverageBlur = transform.AverageBlur()
MedianBlur = transform.MedianBlur()
MotionBlur = transform.MotionBlur()

flip = transform.Flip()
flip_1 = transform.Flip(p=1)
normalize = transform.Normalize()
Pad_And_Shift = transform.Pad_And_Shift(max_size=512)

train_pipeline = [resize, [[CoarseDropout, CoarseSaltAndPepper, Cutout],
                     [AdditiveGaussianNoise, AdditiveLaplaceNoise, AdditivePoissonNoise],
                     [HueSat, DropChannel],
                     [elastic, Rotate_Shear],
                     [add, mul, JpegCompression]], 
                     flip, normalize, Pad_And_Shift]

test_pipeline = [[Resize([224]), normalize, Pad_And_Shift],
                  [Resize([320]), normalize, Pad_And_Shift],
                  [Resize([416]), normalize, Pad_And_Shift],
                  [Resize([512]), normalize, Pad_And_Shift]]

Class2Label = {'neg': 0, 'pos': 1}

class Dataset(object):
    def __init__(self, data_root, pipeline=train_pipeline, split_num=10, split_id=0, mode='train'):
        assert mode == 'train' or mode == 'val' or mode == 'test' or mode == 'train_val'
        self.data_dir = os.path.join(data_root, 'train') 
        self.test_dir = os.path.join(data_root, 'toPredict')
        self.__idx__ = split_id
        self.class2label = Class2Label
        self.img_infos = []
        self.spilt_imginfos = []
        self.train_imginfos = []
        self.val_imginfos = []
        self.split_num = split_num
        self.__mode__ = mode
        
        if self.__mode__ != 'test':
            self.img_infos = self.get_img_infos(self.data_dir)
            self.split_data()
            self.get_train_val_data()
        else:
            self.test_imginfos = self.get_Testimg_infos(self.test_dir)
            
        self.pipeline = Compose(pipeline)
    
    
    def get_img_infos(self, data_dir):
        img_infos = []
        idx = 0
        for filedir in os.listdir(data_dir):
            class_dir = os.path.join(data_dir, filedir)
            if os.path.isdir(class_dir):
                if filedir in self.class2label.keys():
                    for img_name in os.listdir(class_dir):
                        if '.DS_Store' in img_name:
                            continue
                        img_infos.append({'path': os.path.join(class_dir, img_name),
                                          'label': self.class2label[filedir],
                                          'id': idx})
                        idx += 1
        random.seed(888)
        random.shuffle(img_infos)
        return img_infos
    
    def get_Testimg_infos(self, test_dir):
        img_infos = []
        for img_name in os.listdir(test_dir):
            img_infos.append({'path': os.path.join(test_dir, img_name)})
        return img_infos
    
    def split_data(self):
        assert len(self.img_infos) != 0
        num_samples = len(self.img_infos)
        num = int(np.ceil(num_samples / self.split_num))
        self.spilt_imginfos = [self.img_infos[i*num : (i+1)*num] for i in range(self.split_num)]
    
    def get_train_val_data(self):
        tmp_train = []
        for i in range(self.split_num):
            if i == self.__idx__:
                self.val_imginfos = self.spilt_imginfos[i]
            else:
                tmp_train += self.spilt_imginfos[i]
        self.train_imginfos = tmp_train
            
    def convert_mode(self, mode):
        assert mode == 'train' or mode == 'val'
        assert mode != 'test' or self.__mode__ != 'test'
        self.__mode__ = mode
    
    def convert_id(self, idx):
        assert idx < self.split_num
        self.__idx__ = idx
        self.get_train_val_data()
        
    def __getitem__(self, idx):
        if self.__mode__ == 'train':
            img_infos = self.train_imginfos
        elif self.__mode__ == 'val':
            img_infos = self.val_imginfos
        elif self.__mode__ == 'test':
            img_infos = self.test_imginfos
        else:
            img_infos = self.img_infos
            
        img = loading(img_infos[idx]['path'])
        img = self.pipeline(img)
        img = img.transpose(2, 0, 1)
        if self.__mode__ != 'test':
            label = img_infos[idx]['label']
            idx = img_infos[idx]['id']
            return img, label, idx
        else:
            idx = os.path.splitext(
                    os.path.basename(img_infos[idx]['path']))[0]
            idx = int(idx)
            return img, idx
    
    def __len__(self):
        if self.__mode__ == 'train':
            return len(self.train_imginfos)
        elif self.__mode__ == 'val':
            return len(self.val_imginfos)
        else:
            return len(self.test_imginfos)
