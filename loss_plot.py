#!/usr/bin/env python
import os
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np
import sys
plt.style.use('ggplot')



def loss_plot(loss_1, loss_2=None, multi_fig=False):
    plt.switch_backend('agg')
    if multi_fig == False:
        fig, ax = plt.subplots()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        ax.set_xlim([1, len(loss_2)])
        plt.plot(loss_1, label='shuffled case')
        if loss_2:
            plt.plot(loss_2, label='unshuffled case')
        plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
        plt.savefig('loss_result_my_unshuffle.png')



def loss_list_expand(loss_list, flag):
    loss_result = []
    if flag == 1:
        loss_result.append(loss_list[0])
        for i in range(len(loss_list)-1):
            start = loss_list[i]
            end = loss_list[i+1]
            linespace = np.linspace(start, end, 32)
            for j in range(len(linespace)-1):
                loss_result.append(linespace[j+1])
    else:
        loss_result.append(loss_list[0])
        for i in range(len(loss_list)-1):
            start = loss_list[i]
            end = loss_list[i + 1]
            linespace = np.linspace(start, end, 17)
            for j in range(len(linespace) - 1):
                loss_result.append(linespace[j + 1])
    return loss_result

# def loss_list_plot(print_list, label_list, file_pic, x_list=None):
#     plt.switch_backend('agg')
#     fig, ax = plt.subplots()
#     plt.xlabel('Oracle calls (*12800)')
#     plt.ylabel('loss')
#     factor = 1
#     factor2 = 0
#     if x_list == None:
#         len_min = 5000000
#         print_list_term = []
#         for i in range(len(print_list)):
#             print_list_term.append([])
#         for i in range(len(print_list)):
#             for j in range(len(print_list[i])):
#                 if j % 100 == 0:
#                     print_list_term[i].append(print_list[i][j])
#         print_list = print_list_term
#         for loss in print_list:
#             if len_min > len(loss):
#                 len_min = len(loss)
#         print(len_min*factor+factor2)
#         ax.set_xlim([1, int(len_min*factor)+factor2])
#         loss_list = []
#         for i in range(len(print_list)):
#             loss_list.append(print_list[i][: int(len_min*factor)+factor2])
#         line_type = ['-','-','-','-','-']
#         for i in range(len(loss_list)):
#             if label_list[i] == 'D2':
#                 label_list[i] = 'D$^2$'
#             if i == 2:
#                 plt.plot(loss_list[i], label=label_list[i], linestyle=line_type[i], color='green')
#             else:
#                 plt.plot(loss_list[i], label=label_list[i], linestyle=line_type[i])
#     else:
#         # ax.set_xlim(x_list)
#         for i in range(len(loss_list)):
#             if label_list[i] == 'D2':
#                 label_list[i] = '$D^2$'
#             if i == 2:
#                 plt.plot(loss_list[i], label=label_list[i], linestyle=line_type[i], color='green')
#             else:
#                 plt.plot(loss_list[i], label=label_list[i], linestyle=line_type[i])
#     plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
#     plt.savefig(file_pic)
    
def loss_list_plot(print_list, label_list, file_pic, x_list=None):
    plt.switch_backend('agg')
    fig, ax = plt.subplots()
    plt.xlabel('Oracle calls (*25600)')
    plt.ylabel('loss')
    factor = 1
    factor2 = 0
    len_min = 5000000
    print_list_term = []
    for i in range(len(print_list)):
        print_list_term.append([])
    for i in range(len(print_list)):
        for j in range(len(print_list[i])):
            if j % 200 == 0:
                print_list_term[i].append(print_list[i][j])
    print_list = print_list_term
    for loss in print_list:
        if len_min > len(loss):
            len_min = len(loss)
    print(len_min*factor+factor2)
    ax.set_xlim([1, int(len_min*factor)+factor2])
    loss_list = []
    for i in range(len(print_list)):
        loss_list.append(print_list[i][: int(len_min*factor)+factor2])
    line_type = ['-','--','-','--','--']
#     dash_type = [(3,0),(3,3),(3,0),(3,3),(3,5)]
    # color_type = ['magenta','blue','green','yellow','cyan']
    for i in range(len(loss_list)):
#         if label_list[i] == 'C-SPIDER-SFO':
#             line, dash, color = '-', (3,0), 'crimson'
#         if label_list[i] == 'D-SPIDER-SFO':
#             line, dash, color = '--', (3,3), 'dodgerblue'
#         if label_list[i] == 'C-PSGD':
#             line, dash, color = '-.', (2,8), 'lightcoral'
#         if label_list[i] == 'D-PSGD':
#             line, dash, color = '--', (3,3), 'mediumseagreen'
        if label_list[i] == 'D2':
            # line, dash, color = '--', (3,5), 'violet'
            label_list[i] = 'D$^2$'
            plt.plot(loss_list[i], label=label_list[i], linestyle=line_type[i], dashes = (3,3), color = 'lightcoral', alpha = 0.9)
        # plt.plot(loss_list[i], label=label_list[i], linestyle=line, dashes = dash, color = color, alpha = 0.8)
        else:
            plt.plot(loss_list[i], label=label_list[i], linestyle=line_type[i])
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    plt.savefig(file_pic)

# def loss_list_plot(loss_list, label_list, file_pic, x_list):
#     plt.switch_backend('agg')
#     fig, ax = plt.subplots()
#     plt.xlabel('1/bandwidth')
#     plt.ylabel('time of one epoch/second')
#     for i in range(len(loss_list)):
#         plt.plot(x_list, loss_list[i], label=label_list[i])
#     plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
#     plt.savefig(file_pic)

def main(argv):
    num = int(argv[1])
    file_list = []
    label_list = []
    print_list = []
    for i in range(num):
        file_list.append(argv[i+2])
        label_list.append(argv[i+2+num])
        print_list.append([])
    for i in range(num):
        with open(file_list[i], 'r') as f1:
            data_list_1 = f1.read().strip().split('\t')
            for data in data_list_1:
                print_list[i].append(float(data))
    print_list[0] = loss_list_expand(print_list[0], 1)
    print_list[1] = loss_list_expand(print_list[1], 1)
    # print_list[2] = loss_list_expand(print_list[2], 0)
    # print_list[3] = loss_list_expand(print_list[3], 0)
    # print_list[4] = loss_list_expand(print_list[4], 0)
    for i in range(len(print_list)):
        print(len(print_list[i]))
    file_pic = argv[2*num + 2]
    loss_list_plot(print_list, label_list, file_pic)

# def main():
#     file_path = 'Untitled-10.txt'
#     latency_list = []
#     CSPIDER_list = []
#     DCSPIDER_list = []
#     with open(file_path, 'r') as f1:
#         lines = f1.readlines()
#         for line in lines:
#             [a, b, c] = line.strip().split('\t')
#             latency_list.append(float(a))
#             CSPIDER_list.append(float(b))
#             DCSPIDER_list.append(float(c))
#     print_list = []
#     print_list.append(CSPIDER_list)
#     print_list.append(DCSPIDER_list)
#     label_list = ['CSPIDER-SFO', 'DCSPIDER-SFO']
#     file_pic = 'latency.pdf'
#     loss_list_plot(print_list, label_list, file_pic, latency_list)


if __name__ == '__main__':
    # 设计 两个参数列表，一个是文件列表，一个是标签列表
    # 调用 python loss_plot.py 3 a.txt b.txt c.txt a b c pic.png
    main(sys.argv)
