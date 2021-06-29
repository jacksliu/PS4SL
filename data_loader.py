#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @File    : data_loader.py
# @Date    : 2021/5/7
# @Author  : Xin Liu
# @Software: PyCharm
# @Python Version: python 3.8


import torch
from torch.utils.data import Dataset
import numpy as np
import pickle


def load_feature_list(tissue, gene_list):
    if tissue == 'HEK':
        tissue = 'HEK293T'
    with open('./L1000/shRNA_cgs', 'rb') as f:
        shrna_dict = pickle.load(f, encoding='bytes')
    feature_dict = {}
    for symbol, t in shrna_dict.keys():
        t_str = t.decode('utf-8', 'ignore')
        if t_str == tissue:
            feature_dict[symbol.decode('utf-8', 'ignore')] = shrna_dict[(symbol, t)]
    feature_list = [feature_dict[gene.decode('utf-8', 'ignore')] for gene in gene_list]
    return np.array(feature_list), feature_dict


def load_label(tissue, feature_dict):
    symbolA_list, symbolB_list, label_list = [], [], []
    selected = set()
    pos_gene = set()
    f = open('./GEMINI/gemini_' + tissue + '_labels.tsv')
    for line in f.readlines():
        symbolA, geneA, symbolB, geneB, label = line.strip().split('\t')
        if symbolA not in feature_dict or symbolB not in feature_dict:
            continue
        if label == '1':
            pos_gene.add(geneA)
            pos_gene.add(geneB)
        else:
            pass
        if symbolA + ' ' + symbolB not in selected:
            symbolA_list.append(symbolA)
            symbolB_list.append(symbolB)
            label_list.append(int(label))
            selected.add(symbolA + ' ' + symbolB)
        assert symbolB + ' ' + symbolA not in selected
    f.close()
    print('number of samples', len(symbolA_list), len(symbolB_list), len(label_list))
    print('positive', np.sum(label_list), 'negative', len(label_list) - np.sum(label_list))
    return np.array(symbolA_list), np.array(symbolB_list), np.array(label_list)


def concat_data(symbolA_list, symbolB_list, symbolA_unkow, symbolB_unkow, labels, feature_dict):
    sl_feature = []
    sl_labels = []
    for i in range(len(labels)):
        gene_a = symbolA_list[i]
        gene_b = symbolB_list[i]
        gene_af = feature_dict[gene_a]
        gene_bf = feature_dict[gene_b]
        gene_abf = np.concatenate((gene_af, gene_bf))
        gene_baf = np.concatenate((gene_bf, gene_af))
        sl_feature.append(gene_abf)
        sl_feature.append(gene_baf)
        if labels[i] == 0:
            sl_labels.append(0)
            sl_labels.append(0)
        else:
            sl_labels.append(1)
            sl_labels.append(1)
    for i in range(len(symbolB_unkow)):
        gene_ab = np.concatenate((feature_dict[symbolA_unkow[i]], feature_dict[symbolB_unkow[i]]))
        sl_feature.append(gene_ab)
        sl_labels.append(-1)

    sl_labels = np.array(sl_labels)
    sl_feature = np.array(sl_feature)

    # # Balance the positive sample and negative sample

    # sl_pos_index = np.where(sl_labels == 1)[0].tolist()
    # len_pos = len(sl_pos_index)
    # sl_neg_index = np.where(sl_labels == 0)[0].tolist()
    # sl_neg_sample_index = np.random.choice(sl_neg_index, len_pos, replace=False).tolist()
    # sl_sample = np.concatenate((sl_pos_index, sl_neg_sample_index), axis=0).tolist()
    #
    # sl_feature = sl_feature[sl_sample]
    # sl_labels = sl_labels[sl_sample]
    #
    # np.random.shuffle(sl_feature)
    # np.random.shuffle(sl_labels)

    return sl_feature, sl_labels


def sample_unlabel(symbolA_list, symbolB_list, labels, feature_dict):
    # construct feature gene symbol to index
    keys = []
    for i in feature_dict.keys():
        keys.append(i)
    keys_index = list(range(len(feature_dict)))
    symbol_index_dict = dict(zip(keys, keys_index))
    index_symbol_dict = dict(zip(keys_index, keys))

    # construct label gene symbol to index
    first_label_pairs_index = []
    second_label_pairs_index = []
    for i in range(len(symbolA_list)):
        first_label_pairs_index.append(symbol_index_dict[symbolA_list[i]])
        second_label_pairs_index.append(symbol_index_dict[symbolB_list[i]])
        first_label_pairs_index.append(symbol_index_dict[symbolB_list[i]])
        second_label_pairs_index.append(symbol_index_dict[symbolA_list[i]])

    first_all_pairs_index = []
    second_all_pairs_index = []

    first_unknow_pairs_index = np.load('./Unknow_pairs/unknowA_HT29.npy')
    second_unknow_pairs_index = np.load('./Unknow_pairs/unknowB_HT29.npy')

    # first_unknow_pairs_index = []
    # second_unknow_pairs_index = []
    # for i in range(len(feature_dict)):
    #     for j in range(i+1, len(feature_dict)):
    #         first_all_pairs_index.append(i)
    #         second_all_pairs_index.append(j)
    #         first_all_pairs_index.append(j)
    #         second_all_pairs_index.append(i)
    #         if i in first_label_pairs_index and j in second_label_pairs_index:
    #             continue
    #         elif j in first_label_pairs_index and i in first_label_pairs_index:
    #             continue
    #         else:
    #             first_unknow_pairs_index.append(i)
    #             second_unknow_pairs_index.append(j)
    #             first_unknow_pairs_index.append(j)
    #             second_unknow_pairs_index.append(i)
    # np.save("./Unknow_pairs/unknowA_HT29.npy", first_unknow_pairs_index)
    # np.save("./Unknow_pairs/unknowB_HT29.npy", second_unknow_pairs_index)
    len_unknow = len(first_unknow_pairs_index)
    index_unknow = list(range(len_unknow))
    np.random.seed(0)
    sample_unknow_index = np.random.choice(index_unknow, len(first_label_pairs_index), replace=False)
    symbolA_unknow = []
    symbolB_unknow = []

    for i in range(len(sample_unknow_index)):
        symbolA_unknow.append(index_symbol_dict[first_unknow_pairs_index[sample_unknow_index[i]]])
        symbolB_unknow.append(index_symbol_dict[second_unknow_pairs_index[sample_unknow_index[i]]])
    return symbolA_unknow, symbolB_unknow


class SLDataset(Dataset):
    ''' Dataset for loading and preprocessing the SL dataset '''
    def __init__(self, mode, sl_feature, sl_labels):
        # Convert data into PyTorch tensors
        self.data = torch.FloatTensor(sl_feature)
        self.target = torch.FloatTensor(sl_labels)

        self.dim = self.data.shape[1]
        # print('Finished split the {} set of SL Dataset ({} samples found, each dim = {})'
        #       .format(mode, len(self.data), self.dim))

    def __getitem__(self, index):
        # Returns one sample at a time
        return self.data[index], self.target[index]

    def __len__(self):
        # Returns the size of the dataset
        return len(self.data)