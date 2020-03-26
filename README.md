**DC-face-mask-classification-单模97.38**

 - 比赛网址: https://www.dcjingsai.com/common/cmpt/AI%E6%88%98%E7%96%AB%C2%B7%E5%8F%A3%E7%BD%A9%E4%BD%A9%E6%88%B4%E6%A3%80%E6%B5%8B%E5%A4%A7%E8%B5%9B_%E7%AB%9E%E8%B5%9B%E4%BF%A1%E6%81%AF.html
 
 - 提示：该份代码从以下的baseline中修改过来
https://github.com/weiaicunzai/pytorch-cifar100

 - 使用的数据增强方法：多尺度，cutout，shear，rotate，noise，elastic，huesat....等等，详情可参考dataset.py和transform.py
 - 预测的trick：同一张图片多个尺度预测，选取总和最高的那一类作为预测标签，详情可参考test.py.
 - 训练使用的模型：resnext101_32x8d
 - baseline支持模型：resnet50， resnet101， resnext50_32x4d， resnext101-32x8d
 - 预训练模型下载网址:  
 resnet50: https://download.pytorch.org/models/resnet50-19c8e357.pth  
 resnet101: https://download.pytorch.org/models/resnet101-5d3b4d8f.pth  
 resnext50_32x4d: 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth  
 resnext101_32x8d: 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth  
 

