# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 17:38:54 2020

@author: fqsfyq
"""
import imageio
import numpy as np
import cv2
from imgaug import augmenters as iaa
import random

def LoadImgFile(filepath):
    img = imageio.imread(filepath)
    if len(img.shape) < 3:
        img = img[:, :, None]
        img = np.repeat(img, 3, axis=2)
    return img

class Resize(object):
    def __init__(self, image_scale, determint=True):
        self.image_scale = image_scale
        self.determint = determint
    
    def sample_scale(self):
        idx = np.random.randint(0, len(self.image_scale))
        return self.image_scale[idx]
    
    def __call__(self, img):
        scale = self.sample_scale()
        scale = (scale, scale)
        h, w = img.shape[:2]
        resized_img = cv2.resize(
        img, scale, interpolation=cv2.INTER_LINEAR)
        return resized_img

class Rotate_Shear(object):
    def __init__(self, angle=25, shear=(-20, 20), determint=False):
        self.angle = angle
        self.shear = shear
        self.determint = determint
        self.func = iaa.Affine(rotate=(-self.angle, self.angle), shear=shear)
    
    def __call__(self, img):
        return self.func(image=img)

class CoarseDropout(object):
    def __init__(self, limit=(0.0, 0.1), size_percent=(0.02, 0.2), determint=False):
        self.limit = limit
        self.size_percent = size_percent
        self.determint = determint
        self.func = iaa.CoarseDropout(limit, size_percent=size_percent)
    
    def __call__(self, img):
        return self.func(image=img)
    
class BlendAlpha(object):
    def __init__(self, limit=(0.0, 1.0), rotate=(-20, 20), per_channel=0.5, determint=False):
        self.limit = limit
        self.rotate = rotate
        self.per_channel = per_channel
        self.determint = determint
        self.func = iaa.BlendAlpha(limit, iaa.Affine(rotate=rotate), per_channel=per_channel)
    
    def __call__(self, img):
        return self.func(image=img)

class BlendAlphaElementwise(object):
    def __init__(self, limit=(0.0, 1.0), rotate=(-20, 20), per_channel=0.5, determint=False):
        self.limit = limit
        self.rotate = rotate
        self.per_channel = per_channel
        self.determint = determint
        self.func = iaa.BlendAlphaElementwise(limit, iaa.Affine(rotate=rotate), per_channel=per_channel)
    
    def __call__(self, img):
        return self.func(image=img)
    

class GaussianBlur(object):
    def __init__(self, sigma=(0.0, 3.0), determint=False):
        self.sigma = sigma
        self.determint = determint
        self.func = iaa.GaussianBlur(sigma=sigma)
    
    def __call__(self, img):
        return self.func(image=img)

class AverageBlur(object):
    def __init__(self, k=(2, 11), determint=False):
        self.k = k
        self.determint = determint
        self.func = iaa.AverageBlur(k=k)
    
    def __call__(self, img):
        return self.func(image=img)

class RandAugment(object):
    def __init__(self, n=2, m=9, determint=False):
        self.n = n
        self.m = m
        self.determint = determint
        self.func = iaa.RandAugment(n=n, m=m)
    
    def __call__(self, img):
        return self.func(image=img)
    

class MedianBlur(object):
    def __init__(self, k=(3, 11), determint=False):
        self.k = k
        self.determint = determint
        self.func = iaa.MedianBlur(k=k)
    
    def __call__(self, img):
        return self.func(image=img)

class MotionBlur(object):
    def __init__(self, k=(3, 12), determint=False):
        self.k = k
        self.determint = determint
        self.func = iaa.MotionBlur(k=k)
    
    def __call__(self, img):
        return self.func(image=img)
    
    
class Add(object):
    def __init__(self, value=20, determint=False):
        self.value = value
        self.determint = determint
        self.func = iaa.Add((-self.value, self.value))
    
    def __call__(self, img):
        return self.func(image=img)

class Cutout(object):
    def __init__(self, nb_iterations=(3, 8), size=0.1, determint=False):
        self.nb_iterations = nb_iterations
        self.size = size
        self.determint = determint 
        self.func = iaa.Cutout(nb_iterations=nb_iterations, size=size, squared=False, fill_mode="gaussian")
    
    def __call__(self, img):
        return self.func(image=img)

class Mul(object):
    def __init__(self, limit=(0.9, 1.1), determint=False):
        self.limit = limit
        self.determint = determint
        self.func = iaa.Multiply(limit)
    
    def __call__(self, img):
        return self.func(image=img)

class DropChannel(object):
    def __init__(self, p=0.5, determint=False):
        self.p = p
        self.determint = determint
        self.func = iaa.Dropout2d(p=p)
    
    def __call__(self, img):
        return self.func(image=img)
    
class Solarize(object):
    def __init__(self, p=0.5, threshold=(64, 128), determint=False):
        self.p = p
        self.threshold = threshold
        self.determint = determint
        self.func = iaa.Solarize(p, threshold=threshold)
        
    def __call__(self, img):
        return self.func(image=img)
    
class AdditiveGaussianNoise(object):
    def __init__(self, scale_limit=(10, 50), determint=False):
        assert len(scale_limit) == 2
        self.scale_limit = scale_limit
        self.determint = determint
        self.func = iaa.AdditiveGaussianNoise(scale=(scale_limit[0], scale_limit[1]))
    
    def __call__(self, img):
        return self.func(image=img)

class AdditiveLaplaceNoise(object):
    def __init__(self, scale_limit=(0, 0.1*255), determint=False):
        assert len(scale_limit) == 2
        self.scale_limit = scale_limit
        self.determint = determint
        self.func = iaa.AdditiveLaplaceNoise(scale=(scale_limit[0], scale_limit[1]))
    
    def __call__(self, img):
        return self.func(image=img)

class AdditivePoissonNoise(object):
    def __init__(self, scale_limit=(0, 20), determint=False):
        assert len(scale_limit) == 2
        self.scale_limit = scale_limit
        self.determint = determint
        self.func = iaa.AdditivePoissonNoise((scale_limit[0], scale_limit[1]))
    
    def __call__(self, img):
        return self.func(image=img)

class AddToHueAndSaturation(object):
    def __init__(self, limit=(-40, 40), determint=False):
        assert len(limit) == 2
        self.limit = limit
        self.determint = determint
        self.func = iaa.AddToHueAndSaturation(limit)
    
    def __call__(self, img):
        return self.func(image=img)

class JpegCompression(object):
    def __init__(self, compression=(70, 99), determint=False):
        self.compression = compression
        self.determint = determint
        self.func = iaa.JpegCompression(compression=compression)
    
    def __call__(self, img):
        return self.func(image=img)

class CoarseSaltAndPepper(object):
    def __init__(self, p=0.05, size_percent=(0.01, 0.1), determint=False):
        self.p = p
        self.size_percent = size_percent
        self.determint = determint
        self.func = iaa.CoarseSaltAndPepper(p, size_percent=size_percent)
    
    def __call__(self, img):
        return self.func(image=img)
    
    
class ElasticTransformation(object):
    def __init__(self, alpha=90, sigma=9, determint=False):
        self.alpha = alpha
        self.sigma = sigma
        self.determint = determint
        self.func = iaa.ElasticTransformation(alpha=alpha, sigma=sigma)
    
    def __call__(self, img):
        return self.func(image=img)

class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], dtype=np.float32, determint=True):
        self.mean = mean
        self.std = std
        self.determint = determint
        self.dtype = dtype
    
    def __call__(self, img):
        img = img.astype(self.dtype)
        img /= 255.0
        img = (img - self.mean) / self.std
        return img

class Flip(object):
    def __init__(self, p=0.5, determint=True):
        self.p = p
        self.determint = determint
        self.func = iaa.Fliplr(p)
    
    def __call__(self, img):
        return self.func(image=img)

class Pad_And_Shift(object):
    def __init__(self, max_size=512, shift=True, determint=True):
        self.max_size = max_size
        self.determint = determint
        self.shift = shift
    
    def __call__(self, img):
        h, w = img.shape[:2]
        assert h <= self.max_size  and w <= self.max_size
        if self.shift and w < self.max_size and h < self.max_size:
            shift_w = np.random.randint(0, self.max_size - w)
            shift_h = np.random.randint(0, self.max_size - h)
        else:
            shift_w, shift_h = int((self.max_size - w)/2.0), int((self.max_size - h)/2.0)
        pad_img = np.zeros((self.max_size, self.max_size, 3), dtype=np.float32)
        pad_img[shift_h : shift_h + h, shift_w : shift_w + w] = img
        return pad_img
        

    
class Compose(object):
    def __init__(self, transforms, aug_pro=0.9):
        self.transforms = transforms
        self.aug_pro = aug_pro
        for transform in transforms:
            if isinstance(transform, list):
                self.random_transforms = transform
    
    def __call__(self, img):
        for transform in self.transforms:
            if not isinstance(transform, list) and transform.determint:
                img = transform(img)
            elif random.random() <= self.aug_pro:
                if random.random() < 0.5:
                    func = random.choice(transform[0])
                    img = func(img)
                if random.random() < 0.8:
                    func_list = random.choice(transform[1:])
                    func = random.choice(func_list)
                    img = func(img)
        return img
        



        
    
    
