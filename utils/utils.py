"""
@FileName：crawler_utils.py
@Description：用于处理中医医院数据工具类
@Author：HaoBoli
"""

import pickle
import torch
import numpy as np
from sklearn.metrics import jaccard_score, roc_auc_score, precision_score, f1_score, average_precision_score
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



def multi_label_metric(y_gt, y_prob):
    def roc_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(roc_auc_score(y_gt[b], y_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_at_k(y_gt, y_prob, k=3):
        precision = 0
        sort_index = np.argsort(y_prob, axis=-1)[:, ::-1][:, :k]
        for i in range(len(y_gt)):
            TP = 0
            for j in range(len(sort_index[i])):
                if y_gt[i, sort_index[i, j]] == 1:
                    TP += 1
            precision += TP / k
        return precision / len(y_gt)

    def recall_at_k(y_gt, y_prob, k=3):
        precision = 0
        sort_index = np.argsort(y_prob, axis=-1)[:, ::-1][:, :k]
        for i in range(len(y_gt)):
            TP = 0
            for j in range(len(sort_index[i])):
                if y_gt[i, sort_index[i, j]] == 1:
                    TP += 1
            precision += TP / np.sum(np.array(y_gt[i]))
        return precision / len(y_gt)

    def dcg_at_k(scores, k):
        """
        计算DCG@K（Discounted Cumulative Gain at K）
        """
        scores = scores[:k]
        gains = (2 ** scores - 1)
        discounts = np.log2(np.arange(len(scores)) + 2)
        return np.sum(gains / discounts)

    def idcg_at_k(true_scores, k):
        """
        计算IDCG@K（Ideal Discounted Cumulative Gain at K）
        """
        ideal_scores = np.ones_like(true_scores)  # 将真实分数映射为1
        return dcg_at_k(ideal_scores, k)

    def ndcg_at_k(true_scores, predicted_scores, k):
        """
        计算NDCG@K（Normalized Discounted Cumulative Gain at K）
        """
        dcg_value = dcg_at_k(predicted_scores[0], k)
        idcg_value = idcg_at_k(true_scores[0], k)

        # 避免分母为零
        if idcg_value == 0:
            return 0.0
        return dcg_value / idcg_value

    # precision
    p_5 = precision_at_k(y_gt, y_prob, k=5)
    p_10 = precision_at_k(y_gt, y_prob, k=10)
    p_20 = precision_at_k(y_gt, y_prob, k=20)
    r_5 = recall_at_k(y_gt, y_prob, k=5)
    r_10 = recall_at_k(y_gt, y_prob, k=10)
    r_20 = recall_at_k(y_gt, y_prob, k=20)
    n_5 = ndcg_at_k(y_gt, y_prob, k=5)
    n_10 = ndcg_at_k(y_gt, y_prob, k=10)
    n_20 = ndcg_at_k(y_gt, y_prob, k=20)

    return p_5,p_10,p_20, r_5,r_10,r_20,n_5,n_10,n_20

def print_local_history(epoch, local_history, max_epoch=10000):
    s = 'Epoch-{}/{}:  '.format(str(epoch), str(max_epoch))
    for k, v in local_history.items():
        s = s + '{}={:.4f}  '.format(k, v)
    print(s)
    return s


def out_num(num):
    '''
    对输入的array求平均，乘100后，保留两位小数返回
    Args:
        num: 待求值nparray

    Returns: 处理后的值

    '''
    return round(float(np.mean(num)) * 100, 2)



def deduplication_data(herbs_list):
    """
    对二维列表去重，列表中每一行为一组数据，去掉行内重复的值以及重复的行
    Args:
        herbs_list: 二维列表

    Returns:
        去重后的二维列表
    """
    # 将每个处方转换为集合，进行去重操作
    prescriptions =[list(set(prescription)) for prescription in herbs_list]
    # 使用集合来去重，保持原始顺序
    unique_prescriptions = []
    seen_prescriptions = set()

    for prescription in prescriptions:
        prescription_tuple = tuple(prescription)  # 将列表转换为元组以用于集合
        if prescription_tuple not in seen_prescriptions:
            seen_prescriptions.add(prescription_tuple)
            unique_prescriptions.append(prescription)
    herbs_list_uni = unique_prescriptions

    return herbs_list_uni

def same_num(list1, list2):
    """
    求两个集合相同元素个数
    Args:
        list1: 第一个一维列表
        list2: 第二个一维列表

    Returns:
        相同元素个数
    """
    # 使用集合求交集
    set1 = set(list1)
    set2 = set(list2)
    intersection = set1.intersection(set2)

    # 获取交集的元素个数
    return len(intersection)/len(list1)