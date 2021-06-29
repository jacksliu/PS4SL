#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @File    : model.py
# @Date    : 2021/5/7
# @Author  : Xin Liu
# @Software: PyCharm
# @Python Version: python 3.8


import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score


class NeuralNet(nn.Module):
    ''' A simple fully-connected deep neural network '''

    def __init__(self, dim, dnn_layers, p_prob, t1, t2, af):
        super(NeuralNet, self).__init__()
        self.dim = dim
        self.dnn_layers = dnn_layers
        self.T1 = t1
        self.T2 = t2
        self.af = af
        self.model = nn.Sequential()

        self.model.add_module('begin', nn.Linear(1956, self.dim))
        for i in range(self.dnn_layers):
            self.model.add_module((str(-i)), nn.Dropout(p_prob))
            self.model.add_module(str(i + i), nn.Linear(self.dim, self.dim))
            self.model.add_module(str(i + i + 1), nn.ReLU())
        self.model.add_module('last', nn.Linear(self.dim, 1))

        self.criterion = nn.BCEWithLogitsLoss(reduce=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        ''' Given input of size (batch_size x input_dim), compute output of the network '''
        return self.model(x).squeeze(1)

    def cal_loss(self, pred, target, epoch):
        ''' Calculate loss '''
        with torch.no_grad():
            unlabel_index = []
            label_index = []
            for i in range(len(target)):
                if target[i] == -1:
                    unlabel_index.append(i)
                else:
                    label_index.append(i)

            prop = self.sigmoid(pred)
            prop[prop.gt(0.5)] = 1
            prop[prop.le(0.5)] = 0
            pseudo_label = prop

            unlabeled_target = pseudo_label[unlabel_index]
            labeled_target = target[label_index]

            idx = target.eq(-1).float()
            idx[idx.eq(0)] = 1

        unlabeled_loss = torch.sum(self.criterion(pred[unlabel_index], unlabeled_target)) / (len(unlabel_index) + 1e-10)
        labeled_loss = torch.sum(self.criterion(pred[label_index], labeled_target)) / len(label_index)
        loss = labeled_loss + self.unlabeled_weight(epoch) * unlabeled_loss
        return loss

    def cal_eval(self, target, pred):
        """ Evaluation """
        auc_list = []
        aupr_list = []

        auc = roc_auc_score(target, pred)
        auc_list.append(auc)

        aupr = average_precision_score(target, pred)
        aupr_list.append(aupr)

        scores = pred
        scores_output = scores.copy()

        scores[scores >= 0.5] = 1
        scores[scores < 0.5] = 0

        f1 = f1_score(target, scores)
        acc = accuracy_score(target, scores)

        scores_binary_output = scores

        return scores_output, scores_binary_output, auc, f1, aupr, acc

    def unlabeled_weight(self, epoch):
        alpha = 0.0
        if epoch > self.T1:
            alpha = (epoch - self.T1) / (self.T2 - self.T1) * self.af
            if epoch > self.T2:
                alpha = self.af
        return alpha