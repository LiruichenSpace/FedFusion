#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details
from VAE_optimizer import VAE_optimizer
from resnet import ResNet18
from mobilenetv2 import MobileNetV2
from densenet import densenet_cifar


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    # logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    if args.gpu:
        torch.cuda.set_device(int(args.gpu))
#    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)


    # BUILD MODEL
    # Convolutional neural netork
    if args.dataset == 'mnist':
        global_model = CNNFashion_Mnist(args=args)
    elif args.dataset == 'fmnist':
        global_model = CNNFashion_Mnist(args=args)
    elif args.dataset == 'cifar':
#            global_model = CNNCifar(args=args)
        if args.model == 'resnet':
            global_model = ResNet18()
        elif args.model == 'densenet':
            global_model = densenet_cifar()
        elif args.model == 'mobilenet':
            global_model = MobileNetV2()
        elif args.model == '4lCNN':
            global_model = CNNCifar()

    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.cuda()
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, local_test_acc = [], []
    val_acc_list, net_list = [], []
    test_loss, test_acc = [], []
    print_every = 1
    val_loss_pre, counter = 0, 0
    best_test_acc, best_epoch = 0, 0
    max_local_Tacc = 0.0
    
    for epoch in range(args.epochs):
        local_weights, local_losses = [], []
        print('\n | Global Training Round : {%d} |\n' % (epoch+1))

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx])
            w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch+1)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))


        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx])
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        avg_lacc = sum(list_acc)/len(list_acc)
        max_local_Tacc = avg_lacc if avg_lacc > max_local_Tacc else max_local_Tacc
        local_test_acc.append(avg_lacc)

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(' \nAvg Training Stats after {%d} global rounds:' % (epoch+1))
            print('Training Loss : {%f}' % np.mean(np.array(train_loss)))
            print('Local Test Accuracy: {:.2f}% \n'.format(100*local_test_acc[-1]))

            # Test inference after completion of training
            _test_acc, _test_loss = test_inference(args, global_model, test_dataset)
            if _test_acc > best_test_acc:
                best_test_acc = _test_acc
                best_epoch = epoch
            test_acc.append(_test_acc)
            test_loss.append(_test_loss)
            print(' \n Results after {%d} global rounds of training:' % (epoch + 1))
            print("|---- Avg Local Test Accuracy: {:.2f}%".format(100*local_test_acc[-1]))
            print("|---- Test Accuracy: {:.2f}%".format(100*_test_acc))

    # Saving the objects train_loss and local_test_acc:
    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
    print('Maximum acc is {:.2f}% in {:}'.format(100*best_test_acc, best_epoch))

    np.save('../save/fed_{}_{}_{}_C[{}]_M[{}]_iid[{}]_E[{}]_B[{}]_test_acc.npy'.
                format(args.dataset, args.model, args.epochs, args.num_users, args.component_num,
                       args.iid, args.local_ep, args.local_bs), test_acc)
    np.save('../save/fed_{}_{}_{}_C[{}]_M[{}]_iid[{}]_E[{}]_B[{}]_l_test_acc.npy'.
                format(args.dataset, args.model, args.epochs, args.num_users, args.component_num,
                       args.iid, args.local_ep, args.local_bs), local_test_acc)
    np.save('../save/fed_{}_{}_{}_C[{}]_M[{}]_iid[{}]_E[{}]_B[{}]_test_loss.npy'.
                format(args.dataset, args.model, args.epochs, args.num_users, args.component_num,
                       args.iid, args.local_ep, args.local_bs), test_loss)
    np.save('../save/fed_{}_{}_{}_C[{}]_M[{}]_iid[{}]_E[{}]_B[{}]_train_loss.npy'.
                format(args.dataset, args.model, args.epochs, args.num_users, args.component_num,
                       args.iid, args.local_ep, args.local_bs), train_loss)

    # PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    # Plot Loss curve
    plt.figure()
    plt.plot(train_loss, color='b', label='Training Loss')
    plt.plot(test_loss, color='orange', label='Test Loss')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/fed_{}_{}_{}_C[{}]_M[{}]_iid[{}]_E[{}]_S[{}]_loss.png'.
                format(args.dataset, args.model, args.epochs, args.num_users, args.component_num,
                       args.iid, args.local_ep, args.vae_step))
    #
    # # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.plot(range(len(local_test_acc)), local_test_acc, color='g')
    plt.plot(range(len(test_acc)), test_acc, color='b')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Training Rounds')
    plt.savefig('../save/fed_{}_{}_{}_C[{}]_M[{}]_iid[{}]_E[{}]_S[{}]_acc.png'.
                format(args.dataset, args.model, args.epochs, args.num_users, args.component_num,
                       args.iid, args.local_ep, args.vae_step))
