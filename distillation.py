# added to check distillation resistance
from __future__ import print_function

import argparse
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from helpers.loaders import *
from helpers.utils import adjust_learning_rate
from models import ResNet18
from trainer import test, train, distill, trigger_subsets_test

def main():
    parser = argparse.ArgumentParser(description='Train CIFAR-10 models with watermaks.')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--train_db_path', default='./data', help='the path to the root folder of the traininng data')
    parser.add_argument('--test_db_path', default='./data', help='the path to the root folder of the traininng data')
    parser.add_argument('--dataset', default='cifar10', help='the dataset to train on [cifar10]')
    parser.add_argument('--wm_path', default='./data/trigger_set/', help='the path the wm set')
    parser.add_argument('--wm_lbl', default='labels-cifar.txt', help='the path the wm random labels')
    parser.add_argument('--batch_size', default=100, type=int, help='the batch size')
    parser.add_argument('--wm_batch_size', default=2, type=int, help='the wm batch size')
    parser.add_argument('--max_epochs', default=60, type=int, help='the maximum number of epochs')
    parser.add_argument('--lradj', default=20, type=int, help='multiple the lr by 0.1 every n epochs')
    parser.add_argument('--save_dir', default='./checkpoint/', help='the path to the model dir')
    parser.add_argument('--save_model', default='model.t7', help='model name')
    parser.add_argument('--load_path', default='./checkpoint/ckpt.t7', help='the path to the pre-trained model, to be used with resume flag')
    parser.add_argument('--log_dir', default='./log', help='the path the log dir')
    parser.add_argument('--runname', default='train', help='the exp name')

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

    wmloader = None
    print('Loading watermark images')
    wmloader, wmloader_noshuffle = getwmloader(args.wm_path, args.wm_batch_size, args.wm_lbl)

    # create the model
    # Load checkpoint.
    print('==> Loading from checkpoint..')
    assert os.path.exists(args.load_path), 'Error: no checkpoint found!'
    checkpoint = torch.load(args.load_path)
    teacher_net = checkpoint['net']
    # acc = checkpoint['acc']
    # start_epoch = checkpoint['epoch']
    
    print('==> Building model..')
    student_net = ResNet18(num_classes=n_classes)

    teacher_net = teacher_net.to(device)
    student_net = student_net.to(device)
    # support cuda
    if device == 'cuda':
        print('Using CUDA')
        print('Parallel training on {0} GPUs.'.format(torch.cuda.device_count()))
        teacher_net = torch.nn.DataParallel(teacher_net, device_ids=range(torch.cuda.device_count()))
        student_net = torch.nn.DataParallel(student_net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.KLDivLoss()
    optimizer = optim.SGD(student_net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    
    # 学習曲線の描画用
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []
    wm_loss_list = []
    wm_acc_list = []
    wm_subset_acc_list = []

    # loading wm examples
    # if args.wmtrain:
    print("WM acc:")
    test_loss, test_acc = test(student_net, criterion, logfile, wmloader, device)
    wm_subset_acc = trigger_subsets_test(student_net, criterion, logfile, wmloader_noshuffle, device, args.wm_path, args.wm_lbl, subset_size=5)

    print('Start training..') ######## added

    # distillation parameters
    T = 2
    alpha = 0.5
    
    # start training
    for epoch in range(start_epoch, start_epoch + args.max_epochs):
        # adjust learning rate
        adjust_learning_rate(args.lr, optimizer, epoch, args.lradj)

        train_loss, train_acc = distill(epoch, teacher_net, student_net, None, optimizer, logfile,
            trainloader, device, T, alpha, wmloader=None)

        print("Test acc:")
        test_loss, test_acc = test(student_net, criterion, logfile, testloader, device)

        # if args.wmtrain:
        print("WM acc:")
        wm_loss, wm_acc = test(student_net, criterion, logfile, wmloader, device)
        print("WM set acc:")
        wm_subset_acc = trigger_subsets_test(student_net, criterion, logfile, wmloader_noshuffle, device, args.wm_path, args.wm_lbl, subset_size=5)

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
        wm_loss_list.append(wm_loss)
        wm_acc_list.append(wm_acc)
        wm_subset_acc_list.append(wm_subset_acc)

        print('Saving..')
        state = {
            'net': student_net.module if device == 'cuda' else student_net, #### is -> ==
            'acc': test_acc,
            'epoch': epoch,
        }
        if not os.path.isdir(args.save_dir):
            os.mkdir(args.save_dir)
        torch.save(state, os.path.join(args.save_dir, args.save_model))
    
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
    plt.savefig("learning_curve_distillation.png")


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
