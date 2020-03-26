# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
#from dataset import *
from torch.autograd import Variable

from tensorboardX import SummaryWriter

from dataset import Dataset

from conf import settings
from utils import get_network, WarmUpLR

from copy import deepcopy

def train(epoch):

    net.train()
    total_loss = 0
    total_acc = 0
    for batch_index, (images, labels, _) in enumerate(training_loader):
        
        if epoch <= args.warm:
            warmup_scheduler.step()

        images = Variable(images)
        labels = Variable(labels)

        labels = labels.cuda()
        images = images.cuda().half()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        _, pred = outputs.max(1)
        acc = np.mean(pred.eq(labels).cpu().numpy())
        
        total_loss += loss.item()
        total_acc += acc
        
        n_iter = (epoch - 1) * len(training_loader) + batch_index + 1

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if not para.requires_grad:
                continue
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}\tAccuracy: {:0.4f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            acc,
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(training_loader.dataset)
        ))

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)
    
    print('Training Epoch: {epoch}\tTotal Loss: {:0.4f}\tTotal Accuracy: {:0.4f}'.format(
            total_loss/(batch_index+1),
            total_acc/(batch_index+1),
            epoch=epoch))
    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

def eval_training(epoch):
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels, _) in val_loader:
        images = Variable(images)
        labels = Variable(labels)

        images = images.cuda().half()
        labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        test_loss / len(val_loader.dataset),
        correct.float() / len(val_loader.dataset)
    ))
    print()

    #add informations to tensorboard
    writer.add_scalar('Test/Average loss', test_loss / len(val_loader.dataset), epoch)
    writer.add_scalar('Test/Accuracy', correct.float() / len(val_loader.dataset), epoch)

    return correct.float() / len(val_loader.dataset), test_loss / len(val_loader.dataset)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-data_dir', type=str, default=None, help='path of the data')
    parser.add_argument('-pretrain_path', type=str, default=None, help='path of pretrain model')
    parser.add_argument('-weights', type=str, default=None, help='path of weights')
    parser.add_argument('-split', type=int, default=0, help='id of split')
    parser.add_argument('-c', type=int, default=1000, help='number of classes')
    args = parser.parse_args()

    net = get_network(args, use_gpu=args.gpu, num_classes=args.c)
    net._freeze_stages()
    net.half()

    for name, param in net.named_parameters():
        print(name, param.size(), param.requires_grad)
    if args.weights is not None:
        net.load_state_dict(torch.load(args.weights), args.gpu)
        
    train_dataset = Dataset(args.data_dir, split_id=args.split)
    val_dataset = deepcopy(train_dataset)
    val_dataset.convert_mode('val')
    
    training_loader = DataLoader(train_dataset,
                                 num_workers=args.w,
                                 batch_size=args.b,
                                 shuffle=args.s)
    
    val_loader = DataLoader(val_dataset,
                            num_workers=args.w,
                            batch_size=6,
                            shuffle=args.s)
    
    loss_function = nn.CrossEntropyLoss()
    #loss_function = FocalLoss(alpha=[1, 1.6])
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    #optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.6) #learning rate decay
    iter_per_epoch = len(training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, str(args.split), args.net)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, str(args.split), args.net, settings.TIME_NOW))
    input_tensor = torch.Tensor(12, 3, 32, 32).cuda().half()
    writer.add_graph(net, Variable(input_tensor, requires_grad=True))

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}-{loss}.pth')

    #torch.backends.cudnn.benchmark = True
    best_loss = 1000.0
    for epoch in range(1, settings.EPOCH):

        train(epoch)
        acc, loss = eval_training(epoch)
        
        
        if epoch > args.warm:
            train_scheduler.step(epoch)
        
        #start to save best performance model
        if loss < best_loss:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best', loss=str(loss)))
            best_loss = loss
            continue
        
        if not epoch % settings.SAVE_EPOCH or loss < 0.0010:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular', loss=str(loss)))
    
    writer.close()
