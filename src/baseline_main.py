#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import datetime
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import get_dataset
from options import args_parser
from update import test_inference
from models import CNNMnist, CNNFashion_Mnist, CNNCifar
from resnet import ResNet18

def adjust_learning_rate(optimizer, epoch):
    update_list = [150, 250, 350]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return

if __name__ == '__main__':
    args = args_parser()
    multi_gpus = False
    if ',' in args.gpu:
        gpu_ids = [int(id) for id in args.gpu.split(',')]
        multi_gpus = True
    else:
        gpu_ids = [int(args.gpu)]
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load datasets
    train_dataset, test_dataset, _ = get_dataset(args)

    # BUILD MODEL
    # Convolutional neural netork
    if args.dataset == 'mnist':
        global_model = CNNMnist(args=args)
    elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
#        global_model = MobileNetV2(1, 10, alpha =1)
    elif args.dataset == 'cifar':
#        global_model = CNNCifar(args=args)
        global_model = ResNet18()
#        global_model = nn.DataParallel(MobileNetV2(3, 10, alpha = 1), device_ids=gpu_ids)
#            global_model = nn.DataParallel(MobileNet(), device_ids=gpu_ids)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    print(global_model)

    # Training
    # Set optimizer and criterion
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
                                    momentum=0.9, nesterov=True)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr,
                                     weight_decay=1e-4)

    trainloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    criterion = nn.CrossEntropyLoss().to(device)
    epoch_loss = []
    test_loss = []
    train_accs = []
    test_accs = []

    start_time = datetime.datetime.now()

    for epoch in range(args.epochs):
        batch_loss = []
        total,correct = 0, 0

        # Train
        adjust_learning_rate(optimizer, epoch)
        global_model.train()
        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = global_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()       

            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx * len(images), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), loss.item()))
            batch_loss.append(loss.item())

        elapsed_time = datetime.datetime.now() - start_time
        train_acc = correct / total
        train_accs.append(train_acc)

        # Test
        test_acc, test_los = test_inference(args, global_model, test_dataset)
        
        loss_avg = sum(batch_loss)/len(batch_loss)
#        print('Train Loss:', loss_avg
        print('\nTime [%s] : Train loss:%.5f, Acc:%.4f%% ----- Test loss:%.5f, Acc:%.4f%%\n' % (elapsed_time, loss_avg, train_acc*100,test_los, test_acc*100))
        epoch_loss.append(loss_avg)
        test_loss.append(test_los)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
#    np.save('save/train_loss.npy', epoch_loss)
#    np.save('save/test_loss.npy', test_loss)
#    np.save('save/train_acc.npy', train_accs)
#    np.save('save/test_acc.npy', test_accs)
    # Plot loss
#    plt.figure()
#    plt.plot(range(len(epoch_loss)), epoch_loss)
#    plt.xlabel('epochs')
#    plt.ylabel('Train loss')
#    plt.savefig('../save/nn_{}_{}_{}.png'.format(args.dataset, args.model,args.epochs))

    # testing
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    print('Test on', len(test_dataset), 'samples')
    print("Test Accuracy: {:.2f}%".format(100*test_acc))
    
    torch.save(global_model.state_dict(), 'whole_model.pkl')
