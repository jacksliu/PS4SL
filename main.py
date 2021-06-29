#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @File    : main.py
# @Date    : 2021/5/7
# @Author  : Xin Liu
# @Software: PyCharm
# @Python Version: python 3.8


import argparse
import torch
import numpy as np
from sklearn.model_selection import ShuffleSplit
from data_loader import load_label, load_feature_list, concat_data, sample_unlabel
from train import train

device = 'cuda' if torch.cuda.is_available() else 'cpu'
parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200, help='the help of epochs')
parser.add_argument('--dim', type=int, default=128, help='dimension of hidden layer')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--layer', type=int, default=1, help='the numbers of hidden layer')
parser.add_argument('--weight_decay', type=float, default=0.05, help='weight_decay')
parser.add_argument('--p_prob', type=float, default=0.5, help='dropout_probability')
parser.add_argument('--device', type=str, default=device, help='device')
parser.add_argument('--repeats', type=int, default=1, help='device')
args = parser.parse_args()


if __name__ == '__main__':
    # Load feature
    tissue = "A549"
    gene_list = np.load('./L1000/shrna_gene_list_' + tissue + '.npy')
    feature_list, feature_dict = load_feature_list(tissue, gene_list)

    # Load label
    symbolA_list, symbolB_list, labels = load_label(tissue, feature_dict)
    symbol_set = set(symbolA_list).union(set(symbolB_list))
    print('number of unique genes with SL labels', len(symbol_set))

    # Generate unknown sample, the num is equal to the num of the labeled data
    # symbolA_unknow, symbolB_unknow = [], []
    symbolA_unknow, symbolB_unknow = sample_unlabel(symbolA_list, symbolB_list, labels, feature_dict)

    # Concat feature and label
    sl_feature, sl_label = concat_data(symbolA_list, symbolB_list, symbolA_unknow, symbolB_unknow, labels, feature_dict)

    # Split train/test
    kf = ShuffleSplit(n_splits=1, test_size=0.2, random_state=89)
    sl_index = list(range(len(sl_label)))

    # Evaluation
    train_auc_kkf_list = []
    train_f1_kkf_list = []
    train_aupr_kkf_list = []

    eval_auc_kkf_list = []
    eval_f1_kkf_list = []
    eval_aupr_kkf_list = []

    # 观察是否过拟合
    test_auc_kkf_list = []
    test_f1_kkf_list = []
    test_aupr_kkf_list = []

    loss_kkf_list = []
    k = 0
    for train_index, test_index in kf.split(sl_index):
        data = []
        train_data = {'feature': sl_feature[train_index], 'label': sl_label[train_index]}
        test_data = {'feature': sl_feature[test_index], 'label': sl_label[test_index]}
        data.append(train_data)
        data.append(test_data)
        # train one split
        loss_kf_mean, train_auc_kf_mean, train_f1_kf_mean, train_aupr_kf_mean, eval_auc_kf_mean, eval_f1_kf_mean, eval_aupr_kf_mean, test_auc_kf_mean, test_f1_kf_mean, test_aupr_kf_mean = train(args, data, k)

        train_auc_kkf_list.append(train_auc_kf_mean)
        train_f1_kkf_list.append(train_f1_kf_mean)
        train_aupr_kkf_list.append(train_aupr_kf_mean)
        eval_auc_kkf_list.append(eval_auc_kf_mean)
        eval_f1_kkf_list.append(eval_f1_kf_mean)
        eval_aupr_kkf_list.append(eval_aupr_kf_mean)
        test_auc_kkf_list.append(test_auc_kf_mean)
        test_f1_kkf_list.append(test_f1_kf_mean)
        test_aupr_kkf_list.append(test_aupr_kf_mean)
        loss_kkf_list.append(loss_kf_mean)
        k = k + 1


    train_auc_kkf_mean = np.mean(train_auc_kkf_list)
    train_f1_kkf_mean = np.mean(train_f1_kkf_list)
    train_aupr_kkf_mean = np.mean(train_aupr_kkf_list)
    eval_auc_kkf_mean = np.mean(eval_auc_kkf_list)
    eval_f1_kkf_mean = np.mean(eval_f1_kkf_list)
    eval_aupr_kkf_mean = np.mean(eval_aupr_kkf_list)

    test_auc_kkf_mean = np.mean(test_auc_kkf_list)
    test_f1_kkf_mean = np.mean(test_f1_kkf_list)
    test_aupr_kkf_mean = np.mean(test_aupr_kkf_list)
    loss_kkf_mean = np.mean(loss_kkf_list)

    print('The mean of AUC, AUPR and F1 values on the training data are: %.4f, %.4f, %.4f' % (train_auc_kkf_mean, train_aupr_kkf_mean, train_f1_kkf_mean))
    print('The mean of AUC, AUPR and F1 values on the validation data are: %.4f, %.4f, %.4f' % (eval_auc_kkf_mean, eval_aupr_kkf_mean, eval_f1_kkf_mean))
    print('The mean of AUC, AUPR and F1 values on the validation data are: %.4f, %.4f, %.4f' % (test_auc_kkf_mean, test_aupr_kkf_mean, test_f1_kkf_mean))
    print('The mean of training loss is: %.4f' % np.std(loss_kkf_mean))
