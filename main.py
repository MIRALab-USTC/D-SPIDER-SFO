#!/usr/bin/env python

import copy
import os
from random import Random
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from math import ceil
from torch.autograd import Variable
from torch.multiprocessing import Process
from torchvision import datasets, transforms


class Partition(object):
    """ Dataset-like object, but each node only accesses a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Net(nn.Module):
    "Resnet-18"

    def __init__(self, ResidualBlock, num_classes=10):
        super(Net, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return F.log_softmax(out, dim=1)


def class_partition_cifar(large_batch_flag):
    rank = dist.get_rank()
    size = dist.get_world_size()
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = datasets.CIFAR10(root=args.data_path, train=True,
                               download=False, transform=transform)
    index_class_list = []
    for i in range(10):
        index_class_list.append([])
    for i in range(len(dataset)):
        index_class_list[dataset[i][1]].append(i)
    index_node_list = []
    for i in range(5):
        index_node_list.append([])
    for i in range(10):
        for j in range(len(index_class_list[i])):
            idx = int(i/2)
            if idx < 5:
                index_node_list[idx].append(index_class_list[i][j])
    partition_node = Partition(dataset, index_node_list[rank])
    if large_batch_flag:
        batch_size = int(args.large_batch / float(size))
    else:
        batch_size = int(args.small_batch / float(size))
    train_set = torch.utils.data.DataLoader(
        partition_node, batch_size=batch_size, shuffle=True)
    return train_set, batch_size


def partition_dataset_cifar(large_batch_flag, all_flag=False):
    rank = dist.get_rank()
    size = dist.get_world_size()
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = datasets.CIFAR10(root=args.data_path, train=True,
                               download=False, transform=transform)
    
    partition_sizes = [1.0 / size for _ in range(size)]
    if all_flag:
        batch_size = len(partition)
    elif large_batch_flag:
        batch_size = int(args.large_batch / float(size))
    else:
        batch_size = int(args.small_batch / float(size))
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(rank)
    train_set = torch.utils.data.DataLoader(
        partition, batch_size=batch_size, shuffle=True)
    return train_set, batch_size



def print_loss_file(file_name, loss_iteration, rank, size):
    if rank != 0:
        data = torch.tensor(loss_iteration)
        dist.barrier()
        req = dist.isend(data, dst=0)
        req.wait()
    else:
        loss_list_tensor = []
        loss_iter = []
        data = torch.tensor(loss_iteration)
        for i in range(size):
            data = copy.deepcopy(data)
            loss_list_tensor.append(data)
        dist.barrier()
        for i in range(size - 1):
            req = dist.irecv(loss_list_tensor[i + 1], src=i + 1)
            req.wait()
        for j in range(len(loss_list_tensor[0])):
            element = 0
            for i in range(size):
                element += loss_list_tensor[i][j].item()
            loss_iter.append(element / size)
    if rank == 0:
        file_object = open(file_name, 'w')
        for loss in loss_iter:
            file_object.write(str(loss))
            file_object.write('\t')
        file_object.close()


def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size


def broadcast_parameter(model, src=0):
    for param in model.parameters():
        data_local = copy.deepcopy(param.data)
        dist.broadcast(data_local, src=src)
        param.data = data_local
    return


def loss_compute(model, train_set, flag_all=False):
    rank = dist.get_rank()
    size = dist.get_world_size()
    device = torch.device("cuda:" + str(rank) if torch.cuda.is_available() else "cpu")
    if flag_all == True:
        train_set, bsz = partition_dataset_cifar(False, True)
    else:
        idata, (data, target) = train_set.__next__()
        data, target = Variable(data.cuda(rank)), Variable(target.cuda(rank))
        output = model(data)
        loss = F.nll_loss(output, target)
    return loss


def compute_gradient(model, optimizer, data_input, rank, size, bsz=32):
    (data, target) = data_input
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    for param in optimizer.param_groups[0]['params']:
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size
    return loss.item()


def compute_gradient_SPIDER(model, model_old, optimizer, optimizer_old,
                            optimizer_v_old, data_input, rank, size, bsz):
    for param, v_old in zip(optimizer.param_groups[0]['params'],
                            optimizer_v_old.param_groups[0]['params']):
        v_old.grad = copy.deepcopy(param.grad)
    (data, target) = data_input
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer_old.zero_grad()
    output_old = model_old(data)
    loss_old = F.nll_loss(output_old, target)
    loss_old.backward()
    for param, param_old, v_old in zip(optimizer.param_groups[0]['params'],
                                       optimizer_old.param_groups[0]['params'],
                                       optimizer_v_old.param_groups[0]['params']):
        a = v_old.grad.data
        b = param_old.grad.data
        c = param.grad.data
        param.grad.data = c - b + a

    for param, param_old in zip(model.parameters(), model_old.parameters()):
        param_old.data = copy.deepcopy(param.data)
    for param in optimizer.param_groups[0]['params']:
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size
    return loss.item()


def run_CSPIDER_my(rank, size, q=16, lr=0.003, epoches=5):
    torch.manual_seed(1234)
    device = torch.device("cuda:" + str(rank) if torch.cuda.is_available() else "cpu")
    train_set_small, bsz_small = partition_dataset_cifar(False)
    train_set_large, bsz_large = partition_dataset_cifar(True)
    if args.model_load:
        model = torch.load(args.CSPIDER_model_path)
        model.to(device)
    else:
        model = Net(ResidualBlock).to(device)
    model_old = copy.deepcopy(model)
    model_v_old = copy.deepcopy(model)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    optimizer_old = optim.SGD(model_old.parameters(), lr=lr)
    optimizer_v_old = optim.SGD(model_v_old.parameters(), lr=lr)
    num_batches = ceil(len(train_set_small.dataset) / float(bsz_small))
    iter_index = 0
    loss_iteration = []
    for epoch in range(epoches):
        epoch_loss = 0.0
        train_set_large = enumerate(train_set_large)
        train_set_small = enumerate(train_set_small)
        for i in range(num_batches):
            if iter_index % q == 0:
                idata, (data, target) = train_set_large.__next__()
                data, target = Variable(data.to(device)), Variable(target.to(device))
                for param, param_old in zip(model.parameters(), model_old.parameters()):
                    param_old.data = copy.deepcopy(param.data)
                if rank == 0:
                    broadcast_parameter(model, src=0)
                    dist.barrier()
                    loss = compute_gradient(model, optimizer, (data, target), rank, size, bsz_large)
                    dist.barrier()
                    optimizer.step()
                    loss_iteration.append(loss)
                else:
                    broadcast_parameter(model, src=0)
                    dist.barrier()
                    loss = compute_gradient(model, optimizer, (data, target), rank, size, bsz_large)
                    dist.barrier()
                    epoch_loss += loss
                    optimizer.step()
                    loss_iteration.append(loss)
            else:
                idata, (data, target) = train_set_small.__next__()
                data, target = Variable(data.to(device)), Variable(target.to(device))
                if rank == 0:
                    broadcast_parameter(model, src=0)
                    dist.barrier()
                    loss = compute_gradient_SPIDER(model, model_old, optimizer, optimizer_old,
                                                   optimizer_v_old, (data, target), rank, size, bsz_small)
                    dist.barrier()
                    optimizer.step()
                else:
                    broadcast_parameter(model, src=0)
                    dist.barrier()
                    loss = compute_gradient_SPIDER(model, model_old, optimizer, optimizer_old,
                                                   optimizer_v_old, (data, target), rank, size, bsz_small)
                    dist.barrier()
                    epoch_loss += loss
            iter_index += 1
        if rank != 0:
            print('Rank ',
                  dist.get_rank(), ', epoch ', epoch, ': ',
                  epoch_loss / num_batches)
        train_set_small, bsz_small = partition_dataset_cifar(False)
        train_set_large, bsz_large = partition_dataset_cifar(True)

    file_name = args.CSPIDER_file_name
    print_loss_file(file_name, loss_iteration, rank, size)
    if rank == 0 and args.model_save:
        PATH = args.CSPIDER_model_path_save
        torch.save(model, PATH)


def dist_sgd(model, rank):
    group = dist.new_group([0, 1, 3])
    for param in model.parameters():
        sending_right = copy.deepcopy(param.data)
        sending_left = copy.deepcopy(sending_right)
        recving_left_1 = copy.deepcopy(sending_right)
        recving_right_1 = copy.deepcopy(sending_right)
        size = dist.get_world_size()
        left = ((rank - 1) + size) % size
        right = (rank + 1) % size
        if rank % 2 == 0:
            req = dist.isend(sending_right, dst=right)
            req.wait()
            req = dist.irecv(recving_left_1, src=left)
            req.wait()
        else:
            req = dist.irecv(recving_left_1, src=left)
            req.wait()
            req = dist.isend(sending_right, dst=right)
            req.wait()
        dist.barrier()
        if rank % 2 == 0:
            req = dist.isend(sending_left, dst=left)
            req.wait()
            req = dist.irecv(recving_right_1, src=right)
            req.wait()
        else:
            req = dist.irecv(recving_right_1, src=right)
            req.wait()
            req = dist.isend(sending_left, dst=left)
            req.wait()
        param.data = (sending_left + recving_left_1 + recving_right_1) / 3


def local_average(model, rank, size, group_list):
    device = torch.device("cuda:" + str(dist.get_rank()) if torch.cuda.is_available() else "cpu")
    for param in model.parameters():
        sending_right = copy.deepcopy(param.data)
        recving_left_1 = copy.deepcopy(sending_right).to(device)
        recving_right_1 = copy.deepcopy(sending_right).to(device)
        rank = dist.get_rank()
        if rank == 0:
            dist.broadcast(sending_right, src=rank, group=group_list[rank])
            dist.broadcast(recving_right_1, src=(rank+1)%size, group=group_list[(rank+1)%size])
            dist.broadcast(recving_left_1, src=(rank-1)%size, group=group_list[(rank-1)%size])
        elif rank == size-1:
            dist.broadcast(recving_right_1, src=(rank+1)%size, group=group_list[(rank+1)%size])
            dist.broadcast(recving_left_1, src=(rank-1)%size, group=group_list[(rank-1)%size])
            dist.broadcast(sending_right, src=rank, group=group_list[rank])
        else:
            dist.broadcast(recving_left_1, src=(rank-1)%size, group=group_list[(rank-1)%size])
            dist.broadcast(sending_right, src=rank, group=group_list[rank])
            dist.broadcast(recving_right_1, src=(rank+1)%size, group=group_list[(rank+1)%size])
        param.data = sending_right / 2 + (recving_left_1 + recving_right_1) / 4


def run_DCnew_my(rank, size, q=16, lr=0.003, epoches=5):
    device = torch.device("cuda:" + str(rank) if torch.cuda.is_available() else "cpu")
    group_list = []
    for i in range(size):
        group_list.append(dist.new_group([i, (i + 1) % size, ((i - 1) + size) % size]))
    torch.manual_seed(1234)
    train_set_large, bsz_large = partition_dataset_cifar(True)
    train_set_small, bsz_small = partition_dataset_cifar(False)
    if args.model_load == True:
        model = torch.load(args.DCnew_model_path)
        model.to(device)
    else:
        model = Net(ResidualBlock).to(device)
    model_loss = copy.deepcopy(model)
    model_old = copy.deepcopy(model)
    model_v_old = copy.deepcopy(model)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    optimizer_old = optim.SGD(model_old.parameters(), lr=lr)
    optimizer_v_old = optim.SGD(model_v_old.parameters(), lr=lr)

    num_batches = ceil(len(train_set_small.dataset) / float(bsz_small))
    iter_index = 0
    loss_iteration = []

    for epoch in range(epoches):
        epoch_loss = 0.0
        train_set_large = enumerate(train_set_large)
        train_set_small = enumerate(train_set_small)
        for i in range(num_batches):
            if iter_index % q == 0:
                i_data, (data, target) = train_set_large.__next__()
                data, target = Variable(data.to(device)), Variable(target.to(device))
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                epoch_loss += loss.item()
                loss.backward()
                if iter_index != 0:
                    for param, param_old, param_v_old in zip(optimizer.param_groups[0]['params'],
                                                             optimizer_old.param_groups[0]['params'],
                                                             optimizer_v_old.param_groups[0]['params']):
                        a = param.data
                        b = param_old.data
                        c = param.grad.data
                        d = copy.deepcopy(param_v_old.grad.data)
                        param_v_old.grad = copy.deepcopy(param.grad)
                        param.grad.data = (b - a) / lr + c - d
                else:
                    for param, param_v_old in zip(optimizer.param_groups[0]['params'],
                                                             optimizer_v_old.param_groups[0]['params']):
                        param_v_old.grad = copy.deepcopy(param.grad)

                for param, param_old in zip(model.parameters(), model_old.parameters()):
                    param_old.data = copy.deepcopy(param.data)
                optimizer.step()
                local_average(model, rank, size, group_list)
                loss_iteration.append(loss.item())
            else:
                i_data, (data, target) = train_set_small.__next__()
                data, target = Variable(data.to(device)), Variable(target.to(device))
                optimizer.zero_grad()
                output = model(data)

                loss = F.nll_loss(output, target)
                epoch_loss += loss.item()
                loss.backward()
                optimizer_old.zero_grad()
                output_old = model_old(data)
                loss_old = F.nll_loss(output_old, target)
                loss_old.backward()
                for param, param_old, param_v_old in zip(optimizer.param_groups[0]['params'],
                                                         optimizer_old.param_groups[0]['params'],
                                                         optimizer_v_old.param_groups[0]['params']):
                    a = param.grad.data
                    b = param_old.grad.data
                    c = param_v_old.grad.data
                    param_v_old.grad.data = a - b + c
                for param, param_old in zip(optimizer.param_groups[0]['params'],
                                            optimizer_old.param_groups[0]['params']):
                    a = param.data
                    b = param_old.data
                    c = param.grad.data
                    d = param_old.grad.data
                    param.grad.data = (b - a) / lr + c - d
                for param, param_old in zip(model.parameters(), model_old.parameters()):
                    param_old.data = copy.deepcopy(param.data)
                optimizer.step()
                local_average(model, rank, size, group_list)
            iter_index += 1
        train_set_large, bsz_large = partition_dataset_cifar(True)
        train_set_small, bsz_small = partition_dataset_cifar(False)
        print('Rank ',
              dist.get_rank(), ', epoch ', epoch, ': ',
              epoch_loss / num_batches)
    file_name = args.DCnew_file_name
    print_loss_file(file_name, loss_iteration, rank, size)
    avg_model(model, model_loss)
    if rank == 0 and args.model_save:
        PATH = args.DCnew_model_path_save
        torch.save(model, PATH)


def avg_model(model, model_average):
    g_1 = model.parameters()
    g_2 = model_average.parameters()
    while True:
        try:
            param = g_1.__next__()
            param_avg = g_2.__next__()
        except StopIteration:
            break
        size = dist.get_world_size()
        rank = dist.get_rank()
        if size == 2:
            data_local = param.data
            data_0 = data_local if rank == 0 else torch.tensor(np.zeros(data_local.shape), dtype=data_local.dtype)
            data_1 = data_local if rank == 1 else torch.tensor(np.zeros(data_local.shape), dtype=data_local.dtype)
            dist.broadcast(data_0, src=0)
            dist.broadcast(data_1, src=1)
            average_data = (data_0 + data_1) / float(size)
        else:
            param_list = []
            for i in range(size):
                data = copy.deepcopy(param.data)
                param_list.append(data)
            for i in range(size):
                dist.broadcast(param_list[i], src=i)
            result = 0
            for i in range(size):
                result += param_list[i]
            average_data = result / size
        param_avg.data = average_data


def average_gradients_o(optimizer):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in optimizer.param_groups[0]['params']:
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        # allreduce(param.grad.data, param.grad.data)
        param.grad.data /= size


def allreduce(send, recv):
    rank = dist.get_rank()
    size = dist.get_world_size()
    device = torch.device("cuda:" + str(rank) if torch.cuda.is_available() else "cpu")
    send_buff = torch.zeros(send.size()).to(device)
    recv_buff = torch.zeros(send.size()).to(device)
    accum = torch.zeros(send.size()).to(device)
    accum[:] = send[:]

    left = ((rank - 1) + size) % size
    right = (rank + 1) % size

    for i in range(size - 1):
        if i % 2 == 0:
            # Send send_buff
            send_req = dist.isend(send_buff, right)
            dist.recv(recv_buff, left)
            accum[:] += recv[:]
        else:
            # Send recv_buff
            send_req = dist.isend(recv_buff, right)
            dist.recv(send_buff, left)
            accum[:] += send[:]
        send_req.wait()
    recv[:] = accum[:]

def run_CSGD_pytorch(rank, size, lr=0.003, epoches=5, q=16):
    torch.manual_seed(1234)
    device = torch.device("cuda:" + str(rank) if torch.cuda.is_available() else "cpu")
    train_set_large, bsz = partition_dataset_cifar()
    train_set, bsz_large = partition_dataset_cifar_large()
    # train_set, bsz = class_partition_cifar()
    if args.model_load:
        model = torch.load(args.CSGD_model_path)
        model.to(device)
    else:
        model = Net(ResidualBlock).to(device)
        # model = models.__dict__['resnet18']().to(device)
    # optimizer = my_SGD.my_SGD(model.parameters(), lr=lr)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_batches = ceil(len(train_set.dataset) / float(bsz))
    loss_iteration = []
    index_all = 0
    # criterion = nn.CrossEntropyLoss().to(device)
    for epoch in range(epoches):
        epoch_loss = 0.0
        index = 0
        index_all += index
        train_set_large, bsz_large = partition_dataset_cifar_large()
        train_set_large = enumerate(train_set_large)
        train_set, bsz = partition_dataset_cifar()
        # train_set, bsz = class_partition_cifar()
        for data, target in train_set:
            data, target = Variable(data.to(device)), Variable(target.to(device))
            # data, target = Variable(data.cuda()), Variable(target.cuda())
            # data = data.reshape(-1, 28 * 28)
            #            data, target = Variable(data.cuda(rank)), Variable(target.cuda(rank))
            optimizer.zero_grad()
            output = model(data)
            # loss = criterion(output, target)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.item()
            loss.backward()
            average_gradients_o(optimizer)
            optimizer.step()
            if index % q == 0:
                output_loss = loss_compute(model, train_set_large)
                loss_iteration.append(output_loss.item())
            index += 1
        print('Rank ',
              dist.get_rank(), ', epoch ', epoch, ': ',
              epoch_loss / num_batches)

    print(index_all)
    file_name = args.CSGD_file_name
    print_loss_file(file_name, loss_iteration, rank, size)
    if rank == 0:
        PATH = args.CSGD_model_path_save
        torch.save(model, PATH)

def run_DSGD_my(rank, size, lr, epoches, q=16):
    """ Distributed Synchronous SGD Example """
    torch.manual_seed(1234)
    device = torch.device("cuda:" + str(rank) if torch.cuda.is_available() else "cpu")
    group_list = []
    for i in range(size):
        group_list.append(dist.new_group([i, (i + 1) % size, ((i - 1) + size) % size]))
    train_set_large, bsz_large = partition_dataset_cifar_large()
    train_set, bsz = partition_dataset_cifar()
    # train_set, bsz = class_partition_cifar()
    if args.model_load:
        model = torch.load(args.DSGD_model_path)
        model.to(device)
    else:
        model = Net(ResidualBlock).to(device)
    model_loss = copy.deepcopy(model)
    # optimizer = my_SGD.my_SGD(model.parameters(), lr=lr)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    model.train()
    num_batches = ceil(len(train_set.dataset) / float(bsz))
    loss_iteration = []
    for epoch in range(epoches):
        epoch_loss = 0.0
        train_set_large, bsz_large = partition_dataset_cifar_large()
        train_set_large = enumerate(train_set_large)
        train_set, bsz = partition_dataset_cifar()
        # train_set, bsz = class_partition_cifar()
        index = 0
        for data, target in train_set:
            data, target = Variable(data.to(device)), Variable(target.to(device))
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.item()
            loss.backward()
            dist.barrier()
            dist_sgd_size8(model, rank, group_list)
            optimizer.step()
            dist.barrier()
            # calculate the loss in the average model
            if index % q == 0:
                avg_model(model, model_loss)
                output_loss = loss_compute(model_loss, train_set_large)
                loss_iteration.append(output_loss.item())
            index += 1
        print('Rank ',
              dist.get_rank(), ', epoch ', epoch, ': ',
              epoch_loss / num_batches)
    file_name = args.DSGD_file_name
    print_loss_file(file_name, loss_iteration, rank, size)
    avg_model(model, model_loss)
    if rank == 0:
        PATH = args.DSGD_model_path_save
        torch.save(model, PATH)

def run_D2_my(rank, size, lr, epoches, q=16):
    """ Distributed Synchronous SGD Example """
    torch.manual_seed(1234)
    device = torch.device("cuda:" + str(rank) if torch.cuda.is_available() else "cpu")
    group_list = []
    for i in range(size):
        group_list.append(dist.new_group([i, (i + 1) % size, ((i - 1) + size) % size]))
    train_set_large, bsz_large = partition_dataset_cifar_large()
    train_set, bsz = partition_dataset_cifar()
    # train_set, bsz = class_partition_cifar()
    if args.model_load:
        model = torch.load(args.D2_model_path)
        model.to(device)
    else:
        model = Net(ResidualBlock).to(device)
    model_old = copy.deepcopy(model)
    model_loss = copy.deepcopy(model)
    # optimizer = my_SGD.my_SGD(model.parameters(), lr=lr)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    optimizer_old = optim.SGD(model_old.parameters(), lr=lr)
    num_batches = ceil(len(train_set.dataset) / float(bsz))
    loss_iteration = []
    for epoch in range(epoches):
        epoch_loss = 0.0
        # train_set, bsz = partition_dataset_cifar_large()
#         train_set, bsz = class_partition_cifar()
        train_set_large, bsz_large = partition_dataset_cifar_large()
        train_set_large = enumerate(train_set_large)
        train_set, bsz = partition_dataset_cifar()
        index = 0
        for data, target in train_set:
            data, target = Variable(data.to(device)), Variable(target.to(device))
            # data, target = Variable(data), Variable(target)
            # data = data.reshape(-1, 28 * 28)
            #            data, target = Variable(data.cuda(rank)), Variable(target.cuda(rank))
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.item()
            loss.backward()
            if index == 0:
                for param, param_old in zip(optimizer.param_groups[0]['params'],
                                            optimizer_old.param_groups[0]['params']):
                    param_old.data = copy.deepcopy(param.data)
                    param_old.grad = copy.deepcopy(param.grad)
            if index != 0:
                for param, param_old in zip(optimizer.param_groups[0]['params'],
                                            optimizer_old.param_groups[0]['params']):
                    a = param.data
                    b = param_old.data
                    c = param.grad.data
                    d = copy.deepcopy(param_old.grad.data)
                    param_old.grad = copy.deepcopy(param.grad)
                    param.grad.data = (b - a) / lr + c - d
            for param, param_old in zip(optimizer.param_groups[0]['params'],
                                        optimizer_old.param_groups[0]['params']):
                param_old.data = copy.deepcopy(param.data)
            optimizer.step()
            dist.barrier()
            dist_sgd_size8(model, rank, group_list)
            dist.barrier()
            # calculate the loss in the average model
            if index % q == 0:
                avg_model(model, model_loss)
                output_loss = loss_compute(model_loss, train_set_large)
                loss_iteration.append(output_loss.item())
            index += 1
        print('Rank ',
              dist.get_rank(), ', epoch ', epoch, ': ',
              epoch_loss / num_batches)
    file_name = args.D2_file_name
    print_loss_file(file_name, loss_iteration, rank, size)
    avg_model(model, model_loss)
    if rank == 0:
        PATH = args.D2_model_path
        torch.save(model, PATH)


def run_experiment(rank, size, lr, epoches):
    # run_Cnew_my(rank, size, q=16, lr=lr, epoches=epoches)
    run_CSPIDER_my(rank, size, q=16, lr=0.005, epoches=10)
    # run_DSPIDER_my(rank, size, q=16, lr=lr, epoches=epoches)
    # run_CSGD_pytorch(rank, size, lr, epoches)
    # run_CSGD_my(rank, size, lr, epoches)
    # run_DSGD_my(rank, size, lr, epoches)
    # run_DSGD_unshuffle_my(rank, size, lr, epoches)
    # run_CNGD_pytorch(rank, size, lr, epoches)
    # SCSG.run_CSCSG_my(rank, size, q=16, lr=lr, epoches=epoches)
    # SCSG.run_DCSCSG_my(rank, size, q=16, lr=lr, epoches=epoches)
    # run_D2_my(rank, size, lr, epoches)
    run_DCnew_my(rank, size, q=16, lr=0.005, epoches=10)


def init_processes(rank, size, lr, epoches, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, lr, epoches)

epoch_before = 0
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='../data_cifar', help='Where the training and test data sets are')
parser.add_argument('--function_num', default=2, help='the number of functions that we need to run')
parser.add_argument('--function_name', action='append')
parser.add_argument('--model_load', default=False, help='Whether to load the existing model')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.00096, help='Learning rate')
parser.add_argument('-lb', '--large_batch', type=int, default=2048, help='Large batch size')
parser.add_argument('-sb', '--small_batch', type=int, default=128, help='Small batch size')
parser.add_argument('-e', '--epoches', type=int, default=20, help='Epoch size')
parser.add_argument('--size', type=int, default=8, help='the number of nodes')
parser.add_argument('--q', type=int, default=16, help='the number of nodes')
args = parser.parse_args()
parser.add_argument('--CSPIDER_file_name', default='./Results1113/loss_CSPIDER_' + str(args.size) + '-'
                                                   + str(args.learning_rate * 1000) + str(args.epoches) + '.txt')
parser.add_argument('--DCnew_file_name', default='./Results1113/loss_DCnew_my' + str(args.size) + '-'
                                                 + str(args.learning_rate * 1000) + str(args.epoches) + '.txt')
parser.add_argument('--CSGD_file_name', default='./Results1113/loss_CSGD_dd_pytorch' + str(args.size) + '-'
                                                + str(args.learning_rate * 1000) + str(args.epoches+epoch_before) + '.txt')
parser.add_argument('--DSGD_file_name', default='./Results1113/loss_DPSGD_dd_my' + str(args.size) + '-'
                                                + str(args.learning_rate * 1000) + str(args.epoches+epoch_before) + '.txt')
parser.add_argument('--D2_file_name', default='./Results1113/loss_D2_my' + str(args.size) + '-'
                                                + str(args.learning_rate * 1000) + str(args.epoches) + '.txt')
parser.add_argument('--model_save', default=False, help='whether to save the model')
parser.add_argument('--CSPIDER_model_path', default='./Results1113/model_CSPIDER' + str(args.size) + '-'
                                                    + str(args.learning_rate * 1000) + str(args.epoches) + '.pkl')
parser.add_argument("--DCnew_model_path", default='./Results1113/model_DCnew' + str(args.size) + '-'
                                                  + str(args.learning_rate * 1000) + str(args.epoches) + '.pkl')
parser.add_argument('--CSGD_model_path', default='./Results1113/model_CSGD' + str(args.size) + '-'
                                                 + str(args.learning_rate * 1000) + str(epoch_before) + '.pkl')
parser.add_argument('--DSGD_model_path', default='./Results1113/model_DPSGD' + str(args.size) + '-'
                                                 + str(args.learning_rate * 1000) + str(epoch_before) + '.pkl')
parser.add_argument('--D2_model_path', default='./Results1113/model_D2' + str(args.size) + '-'
                                                 + str(args.learning_rate * 1000) + str(args.epoches) + '.pkl')
parser.add_argument('--CSPIDER_model_path_save', default='./Results1113/model_CSPIDER_D2' + str(args.size) + '-'
                                                    + str(args.learning_rate * 1000) + str(args.epoches) + '.pkl')
parser.add_argument("--DCnew_model_path_save", default='./Results1113/model_DCnew_D2' + str(args.size) + '-'
                                                  + str(args.learning_rate * 1000) + str(args.epoches) + '.pkl')
parser.add_argument('--CSGD_model_path_save', default='./Results1113/model_CSPIDER_D2' + str(args.size) + '-'
                                                    + str(args.learning_rate * 1000) + str(args.epoches+epoch_before)+'.pkl')
parser.add_argument("--DSGD_model_path_save", default='./Results1113/model_DCnew_D2' + str(args.size) + '-'
                                                  + str(args.learning_rate * 1000) + str(args.epoches+epoch_before) + '.pkl')
args = parser.parse_args()

if __name__ == "__main__":
    size = args.size
    lr = args.learning_rate
    epoches = args.epoches
    processes = []
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, lr, epoches, run_experiment))
        p.start()
        pid = p.pid
        os.system("taskset -p -c %d %d" % ((2 * rank % os.cpu_count()), p.pid))
        # os.sched_setaffinity(pid, core_id[rank])
        processes.append(p)
    for p in processes:
        p.join()