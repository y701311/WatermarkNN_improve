import os
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn

from helpers.utils import progress_bar

# Train function
def train(epoch, net, criterion, optimizer, logfile, loader, device, wmloader=False, tune_all=True):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    iteration = -1
    wm_correct = 0
    print_every = 5
    l_lambda = 1.2

    # update only the last layer
    if not tune_all:
        if type(net) is torch.nn.DataParallel:
            net.module.freeze_hidden_layers()
        else:
            net.freeze_hidden_layers()

    # get the watermark images
    wminputs, wmtargets = [], []
    if wmloader:
        for wm_idx, (wminput, wmtarget) in enumerate(wmloader):
            wminput, wmtarget = wminput.to(device), wmtarget.to(device)
            wminputs.append(wminput)
            wmtargets.append(wmtarget)

        # the wm_idx to start from
        wm_idx = np.random.randint(len(wminputs))
    for batch_idx, (inputs, targets) in enumerate(loader):
        iteration += 1
        inputs, targets = inputs.to(device), targets.to(device)

        # add wmimages and targets
        if wmloader:
            inputs = torch.cat([inputs, wminputs[(wm_idx + batch_idx) % len(wminputs)]], dim=0)
            targets = torch.cat([targets, wmtargets[(wm_idx + batch_idx) % len(wminputs)]], dim=0)
            
        inputs = inputs.to(torch.float32)
        targets = targets.to(torch.float32)

        optimizer.zero_grad()
        outputs = net(inputs)
        # loss = criterion(outputs, targets)
        loss = criterion(nn.functional.log_softmax(outputs, dim=1), targets)
        
        # lossのトリガー部分をλ倍にする
        # data_size_nowm = len(inputs)
        # lambda_loss = 5
        # loss = criterion(nn.functional.log_softmax(outputs[:data_size_nowm], dim=1), targets[:data_size_nowm]) \
        #     + lambda_loss * criterion(nn.functional.log_softmax(outputs[data_size_nowm:], dim=1), targets[data_size_nowm:])

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, target_data = torch.max(targets.data, 1)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(target_data).cpu().sum()

        progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    with open(logfile, 'a') as f:
        f.write('Epoch: %d\n' % epoch)
        f.write('Loss: %.3f | Acc: %.3f%% (%d/%d)\n'
                % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    
    # loss and acc
    return (train_loss / (batch_idx + 1)), (100. * correct / total)


# train function in a teacher-student fashion
def train_teacher(epoch, net, criterion, optimizer, use_cuda, logfile, loader, wmloader):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    iteration = -1

    # get the watermark images
    wminputs, wmtargets = [], []
    if wmloader:
        for wm_idx, (wminput, wmtarget) in enumerate(wmloader):
            if use_cuda:
                wminput, wmtarget = wminput.cuda(), wmtarget.cuda()
            wminputs.append(wminput)
            wmtargets.append(wmtarget)
        # the wm_idx to start from
        wm_idx = np.random.randint(len(wminputs))

    for batch_idx, (inputs, targets) in enumerate(loader):
        iteration += 1
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        if wmloader:
            # add wmimages and targets
            inputs = torch.cat([inputs, wminputs[(wm_idx + batch_idx) % len(wminputs)]], dim=0)
            targets = torch.cat([targets, wmtargets[(wm_idx + batch_idx) % len(wminputs)]], dim=0)

        inputs, targets = Variable(inputs), Variable(targets)

        optimizer.zero_grad()
        outputs = net(inputs)
        # loss = criterion(outputs, targets)
        loss = criterion(nn.functional.log_softmax(outputs, dim=1), targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    with open(logfile, 'a') as f:
        f.write('Epoch: %d\n' % epoch)
        f.write('Loss: %.3f | Acc: %.3f%% (%d/%d)\n'
                % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


# Test function
def test(net, criterion, logfile, loader, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        # loss = criterion(outputs, targets)
        loss = criterion(nn.functional.log_softmax(outputs, dim=1), targets)

        test_loss += loss.item()
        _, target_data = torch.max(targets.data, 1)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(target_data).cpu().sum()

        progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    with open(logfile, 'a') as f:
        f.write('Test results:\n')
        f.write('Loss: %.3f | Acc: %.3f%% (%d/%d)\n'
                % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        
    # loss and acc
    return (test_loss / (batch_idx + 1)), (100. * correct / total)

# Test function for trigger subsets
def trigger_subsets_test(net, criterion, logfile, loader, device, wm_path, labels_path, subset_size):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    wm_outputs = []
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        # loss = criterion(outputs, targets)
        loss = criterion(nn.functional.log_softmax(outputs, dim=1), targets)
        
        outputs_softmax = nn.functional.softmax(outputs, dim=1)
        for output in outputs_softmax:
            wm_outputs.append(output.cpu().detach().numpy())

        test_loss += loss.item()
        _, target_data = torch.max(targets.data, 1)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(target_data).cpu().sum()

        progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    wm_targets = np.loadtxt(os.path.join(wm_path, labels_path))
    subset_id = wm_targets[:, 0]
    wm_targets = wm_targets[:, 1:]
    wm_outputs = np.array(wm_outputs)
    wm_subset_indexed = []
    for id, output, target in zip(subset_id, wm_outputs, wm_targets):
        wm_subset_indexed.append([id, output, target])
    wm_subset_indexed.sort(key=lambda x: x[0])
    wm_subset_indexed = np.array(wm_subset_indexed, dtype=object)
    
    subset_outputs = []
    subset_targets = []
    for i in range(0, len(wm_subset_indexed), subset_size):
        # np.set_printoptions(precision=3)
        # print()
        # print(f"wm_output: {wm_subset_indexed[i:i+subset_size, 1]=}")
        # print(f"wm_target: {wm_subset_indexed[i:i+subset_size, 2]=}")
        # print(f"wm_output: {np.average(wm_subset_indexed[i:i+subset_size, 1], axis=0)=}")
        # print(f"wm_target: {np.average(wm_subset_indexed[i:i+subset_size, 2], axis=0)=}")
        subset_outputs.append(np.average(wm_subset_indexed[i:i+subset_size, 1], axis=0))
        subset_targets.append(np.average(wm_subset_indexed[i:i+subset_size, 2], axis=0))
        
    subset_correct = 0
    subset_num = len(wm_targets) / subset_size
    for subset_output, subset_target in zip(subset_outputs, subset_targets):
        if np.argmax(subset_output) == np.argmax(subset_target):
            subset_correct += 1
    print(f"WM subset acc: {100. * subset_correct / subset_num}% ({subset_correct}/{subset_num})")

    with open(logfile, 'a') as f:
        f.write('Test results:\n')
        f.write('Loss: %.3f | Acc: %.3f%% (%d/%d)\n'
                % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        
    # return the acc.
    return 100. * subset_correct / subset_num

# added
def distillation_loss(y, labels, teacher_scores, T, alpha):
    return nn.KLDivLoss()(torch.log_softmax(y / T, dim=1), torch.softmax(teacher_scores / T, dim=1)) * (T * T * 2.0 * alpha) + \
           nn.CrossEntropyLoss()(y, labels) * (1. - alpha)

# added
# Distillation
def distill(epoch, teacher_net, student_net, criterion, optimizer, logfile, loader, device, T=2.0, alpha=0.5, wmloader=False, tune_all=True):
    print('\nEpoch: %d' % epoch)
    teacher_net.eval()
    student_net.train()
    train_loss = 0
    correct = 0
    total = 0
    iteration = -1
    wm_correct = 0
    print_every = 5
    l_lambda = 1.2

    # update only the last layer
    # if not tune_all:
    #     if type(net) is torch.nn.DataParallel:
    #         net.module.freeze_hidden_layers()
    #     else:
    #         net.freeze_hidden_layers()

    # get the watermark images
    wminputs, wmtargets = [], []
    if wmloader:
        for wm_idx, (wminput, wmtarget) in enumerate(wmloader):
            wminput, wmtarget = wminput.to(device), wmtarget.to(device)
            wminputs.append(wminput)
            wmtargets.append(wmtarget)

        # the wm_idx to start from
        wm_idx = np.random.randint(len(wminputs))
    for batch_idx, (inputs, targets) in enumerate(loader):
        iteration += 1
        inputs, targets = inputs.to(device), targets.to(device)

        # add wmimages and targets
        if wmloader:
            inputs = torch.cat([inputs, wminputs[(wm_idx + batch_idx) % len(wminputs)]], dim=0)
            targets = torch.cat([targets, wmtargets[(wm_idx + batch_idx) % len(wminputs)]], dim=0)

        optimizer.zero_grad()
        teacher_outputs = teacher_net(inputs)
        student_outputs = student_net(inputs)
        loss = distillation_loss(student_outputs, targets, teacher_outputs, T, alpha)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, target_data = torch.max(targets.data, 1)
        _, predicted = torch.max(student_outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(target_data).cpu().sum()

        progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    with open(logfile, 'a') as f:
        f.write('Epoch: %d\n' % epoch)
        f.write('Loss: %.3f | Acc: %.3f%% (%d/%d)\n'
                % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    
    # loss and acc
    return (train_loss / (batch_idx + 1)), (100. * correct / total)
