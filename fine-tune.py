"""Train CIFAR with PyTorch."""
from __future__ import print_function

import argparse
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from helpers.consts import *
from helpers.ImageFolderCustomClass import ImageFolderCustomClass
from helpers.loaders import *
from helpers.utils import re_initializer_layer
from trainer import test, train, trigger_subsets_test

def main():
    parser = argparse.ArgumentParser(description='Fine-tune CIFAR10 models.')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--train_db_path', default='./data', help='the path to the root folder of the traininng data')
    parser.add_argument('--test_db_path', default='./data', help='the path to the root folder of the traininng data')
    parser.add_argument('--dataset', default='cifar10', help='the dataset to train on [cifar10]')
    parser.add_argument('--wm_path', default='./data/trigger_set/', help='the path the wm set')
    parser.add_argument('--wm_lbl', default='labels-cifar.txt', help='the path the wm random labels')
    parser.add_argument('--batch_size', default=100, type=int, help='the batch size')
    parser.add_argument('--max_epochs', default=20, type=int, help='the maximum number of epochs')
    parser.add_argument('--load_path', default='./checkpoint/model.t7', help='the path to the pre-trained model')
    parser.add_argument('--save_dir', default='./checkpoint/', help='the path to the model dir')
    parser.add_argument('--save_model', default='finetune.t7', help='model name')
    parser.add_argument('--log_dir', default='./log', help='the path the log dir')
    parser.add_argument('--runname', default='finetune', help='the exp name')
    parser.add_argument('--tunealllayers', action='store_true', help='fine-tune all layers')
    parser.add_argument('--reinitll', action='store_true', help='re initialize the last layer')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    LOG_DIR = args.log_dir
    if not os.path.isdir(LOG_DIR):
        os.mkdir(LOG_DIR)
    logfile = os.path.join(LOG_DIR, 'log_' + str(args.runname) + '.txt')
    confgfile = os.path.join(LOG_DIR, 'conf_' + str(args.runname) + '.txt')

    # save configuration parameters
    with open(confgfile, 'w') as f:
        for arg in vars(args):
            f.write('{}: {}\n'.format(arg, getattr(args, arg)))

    trainloader, testloader, n_classes = getdataloader(
        args.dataset, args.train_db_path, args.test_db_path, args.batch_size)

    # load watermark images
    print('Loading watermark images')
    wmloader, wmloader_noshuffle = getwmloader(args.wm_path, args.batch_size, args.wm_lbl)

    # Loading model.
    print('==> loading model...')
    checkpoint = torch.load(args.load_path)
    net = checkpoint['net']
    acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    net = net.to(device)
    # support cuda
    if device == 'cuda':
        print('Using CUDA')
        print('Parallel training on {0} GPUs.'.format(torch.cuda.device_count()))
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    # re initialize and re train the last layer
    private_key = net.module.linear
    if args.reinitll:
        net, _ = re_initializer_layer(net, n_classes)

    if device == 'cuda': # is -> ==
        net.module.unfreeze_model()
    else:
        net.unfreeze_model()

    criterion = nn.KLDivLoss()
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    
    # 学習曲線の描画用
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []
    wm_loss_list = []
    wm_acc_list = []
    wm_subset_acc_list = []

    # start training loop
    print("WM acc:")
    test_loss, test_acc = test(net, criterion, logfile, wmloader, device)
    wm_subset_acc = trigger_subsets_test(net, criterion, logfile, wmloader_noshuffle, device, args.wm_path, args.wm_lbl, subset_size=5)
    print("Test acc:")
    test_loss, test_acc = test(net, criterion, logfile, testloader, device)

    # start training
    for epoch in range(start_epoch, start_epoch + args.max_epochs):
        train_loss, train_acc = train(epoch, net, criterion, optimizer, logfile,
                trainloader, device, wmloader=False, tune_all=args.tunealllayers)

        print("Test acc:")
        test_loss, test_acc = acc = test(net, criterion, logfile, testloader, device)

        # replacing the last layer to check the wm resistance
        new_layer = net.module.linear
        net, _ = re_initializer_layer(net, 0, private_key)
        print("WM acc:")
        wm_loss, wm_acc = test(net, criterion, logfile, wmloader, device)
        print("WM set acc:")
        wm_subset_acc = trigger_subsets_test(net, criterion, logfile, wmloader_noshuffle, device, args.wm_path, args.wm_lbl, subset_size=5)

        # plugging the new layer back
        net, _ = re_initializer_layer(net, 0, new_layer)
        
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc.item())
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc.item())
        wm_loss_list.append(wm_loss)
        wm_acc_list.append(wm_acc.item())
        wm_subset_acc_list.append(wm_subset_acc)

        print('Saving..')
        state = {
            'net': net.module if device == 'cuda' else net, # is -> ==
            'acc': test_acc,
            'epoch': epoch,
        }
        if not os.path.isdir(args.save_dir):
            os.mkdir(args.save_dir)
        torch.save(state, os.path.join(args.save_dir, str(args.runname) + str(args.save_model)))
    
    print(f"{train_acc_list=}")
    print(f"{test_acc_list=}")
    print(f"{wm_acc_list=}")
    print(f"{wm_subset_acc_list=}")
    
    # 学習曲線の描画
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_list, label='Train Loss', marker='o')
    plt.plot(test_loss_list, label='Test Loss', marker='o')
    plt.plot(wm_loss_list, label='WM Loss', marker='o')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_list, label='Train Accuracy', marker='o')
    plt.plot(test_acc_list, label='Test Accuracy', marker='o')
    plt.plot(wm_acc_list, label='WM Accuracy', marker='o')
    plt.plot(wm_subset_acc_list, label='WM Subset Accuracy', marker='o')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("learning_curve_finetune_" + args.runname + ".png")


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()

