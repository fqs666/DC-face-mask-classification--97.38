#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""

import argparse
#from dataset import *

#from skimage import io


import torch
from torch.utils.data import DataLoader
from dataset import Dataset, test_pipeline

from utils import get_network
import numpy as np
import csv

Label2Class = {0: 'neg', 1: 'pos'}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-c', type=int, default=1000, help='number of class')
    parser.add_argument('-data_dir', type=str, default=None, help='path of the data')
    parser.add_argument('-pretrain_path', type=str, default=None, help='path of the pretrain model')
    args = parser.parse_args()

    net = get_network(args, num_classes=args.c)

    test_dataset = [Dataset(args.data_dir, mode='test', pipeline=p) for p in test_pipeline]
    test_loader = [DataLoader(d, num_workers=args.w, batch_size=args.b, shuffle=args.s) 
                    for d in test_dataset]

    net.load_state_dict(torch.load(args.weights), args.gpu)
    net.eval()
    
    result = np.zeros((len(test_dataset[0]), len(test_dataset), args.c))
    with torch.no_grad():
        for i, loader in enumerate(test_loader):
            for n_iter, (images, idxs) in enumerate(loader):
                images = images.cuda()
                pred = net(images).cpu().numpy()
                idxs = idxs.numpy()
                result[idxs, i, :] = pred
            print('it has finished %dth loader prediction!' % (i))
        pre_labels = np.argmax(np.sum(result, axis=1), axis=1)
        np.save('result.npy', result)
    with open('submit.csv', mode='w') as f:
        csv_writer = csv.writer(f, delimiter=',')
        csv_writer.writerow(['ID', 'Label'])
        for idx in range(pre_labels.shape[0]):
            csv_writer.writerow([idx, Label2Class[pre_labels[idx]]])
                
        
