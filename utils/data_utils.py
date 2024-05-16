"""
@FileName：data_utils.py
@Description：数据工具类，用于数据读取构建等
@Author：HaoBoli
"""
from torch.utils.data import TensorDataset,DataLoader
from utils import *
import torch

import pickle
import torch
import numpy as np
from sklearn.metrics import jaccard_score, roc_auc_score, precision_score, f1_score, average_precision_score
import random

def read_data(name):
    """
    读取pkl文件
    Args:
        name: pkl文件名，读取 ./data/name.pkl
    Returns:
        rq: 读取的对象
    """
    with open("../data/%s.pkl"%(name),'rb') as file:
        print("正在读取:", "./data/%s.pkl"%(name))
        rq = pickle.loads(file.read())
        print("读取完成")
        return rq

def write_data(name, data):
    """
    写入pkl文件
    Args:
        name: 新的文件名，文件存储在./data/name.pkl中
        data: 要写入的数据
    """
    pkl_path = "../../data/%s.pkl"%(name)
    # 将对象写入Pickle文件
    with open(pkl_path, 'wb') as file:
        print("正在写入:",pkl_path)
        pickle.dump(data, file)
        print("写入完成")

def get_data(device, args):
    """
    准备数据：图，训练数据以及测试数据
    Args:
        device: cpu/gpu
        args: 超参数

    Returns:
        data: 图数据
        data_args: 数据参数，如图大小，个数等
        train_data_loader: 训练数据
        test_data_loader: 测试数据
    """
    data = read_data('graph_tcm')
    train_dis, train_tcm, train_syn = read_data('train_data')
    test_dis, test_tcm, test_syn = read_data('test_data')
    data = data.to(device)
    train_dis = torch.from_numpy(train_dis).type(torch.FloatTensor).to(device)
    train_tcm = torch.from_numpy(train_tcm).type(torch.FloatTensor).to(device)
    train_syn = torch.from_numpy(train_syn).type(torch.FloatTensor).to(device)
    test_dis = torch.from_numpy(test_dis).type(torch.FloatTensor).to(device)
    test_tcm = torch.from_numpy(test_tcm).type(torch.FloatTensor).to(device)
    test_syn = torch.from_numpy(test_syn).type(torch.FloatTensor).to(device)

    train_dataset = TensorDataset(train_dis, train_syn, train_tcm)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset = TensorDataset(test_dis, test_syn, test_tcm)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size)

    data_args = dict()
    data_args['tcm_num'] = data['tcm'].num_nodes
    data_args['dis_num'] = data['dis'].num_nodes
    data_args['syn_num'] = data['syn'].num_nodes
    data_args['tcm_dim'] = data['tcm'].x.shape[1]
    data_args['dis_dim'] = data['dis'].x.shape[1]
    data_args['syn_dim'] = data['syn'].x.shape[1]


    return data, data_args, train_data_loader, test_data_loader