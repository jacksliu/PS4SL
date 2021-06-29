#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @File    : train.py
# @Date    : 2021/5/7
# @Author  : Xin Liu
# @Software: PyCharm
# @Python Version: python 3.8


import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
from model import NeuralNet
from data_loader import SLDataset
from sklearn.model_selection import ShuffleSplit


def train(args, data, k):
    train_data = data[0]
    test_data = data[1]
    # Split train dataset into train and validation dataset
    train_feature = train_data['feature']
    train_label = train_data['label']
    train_index = list(range(len(train_label)))
    kf = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

    train_auc_kf_list = []
    train_f1_kf_list = []
    train_aupr_kf_list = []

    eval_auc_kf_list = []
    eval_f1_kf_list = []
    eval_aupr_kf_list = []

    test_auc_kf_list = []
    test_f1_kf_list = []
    test_aupr_kf_list = []

    loss_kf_list = []
    kk = 1
    for train_kf, dev_kf in kf.split(train_index):
        train_kf_feature = train_feature[train_kf]
        train_kf_label = train_label[train_kf]
        dev_kf_feature = train_feature[dev_kf]
        dev_kf_label = train_label[dev_kf]

        train_kf_data = {'feature': train_kf_feature, 'label': train_kf_label}
        dev_kf_data = {'feature': dev_kf_feature, 'label': dev_kf_label}

        train_dataset = SLDataset('train', train_kf_feature, train_kf_label)
        train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, pin_memory=True)

        model = NeuralNet(args.dim, args.layer, args.p_prob, t1=20, t2=100, af=1).to(args.device)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay, amsgrad=False)
        best_eval_auc_flag = 0
        for epoch in range(args.n_epochs):
            model.train()
            loss_list = []
            for batch in train_dataloader:
                x, y = batch
                optimizer.zero_grad()
                x, y = x.to(args.device), y.to(args.device)
                pred = model(x)
                mse_loss = model.cal_loss(pred, y, epoch)
                mse_loss.backward()
                optimizer.step()
                loss_list.append(mse_loss.item())
            loss_mean = np.mean(loss_list)

            # ctr_eval on train, dev and test dataset
            train_score, train_score_binary, train_auc, train_f1, train_aupr, train_acc = ctr_eval(args, model, train_kf_data)
            dev_score, dev_score_binary, dev_auc, dev_f1, dev_aupr, dev_acc = ctr_eval(args, model, dev_kf_data)
            test_score, test_score_binary, test_auc, test_f1, test_aupr, test_acc = ctr_eval(args, model, test_data)

            # save the models with the highest eval_auc
            if (dev_auc > best_eval_auc_flag):
                best_eval_auc_flag = dev_auc
                best_k = k
                best_kk = kk

                best_train_auc = train_auc
                best_train_f1 = test_f1
                best_train_aupr = train_aupr

                best_eval_auc = dev_auc
                best_eval_f1 = dev_f1
                best_eval_aupr = dev_aupr

                best_test_auc = test_auc
                best_test_f1 = test_f1
                best_test_aupr = test_aupr
                best_loss = loss_mean

                best_test_score = test_score
                best_test_score_binary = test_score_binary

                torch.save(model.state_dict(), './best_models/best_model_' + str(best_k) + '_' + str(best_kk) + '.pth')

                print("-" * 50)
                print('Best Saving Epoch %d' % epoch + ':')
                print('The AUC, AUPR and F1 values on the training data are: %.4f, %.4f, %.4f' % (train_auc, train_aupr, train_f1))
                print('The AUC, AUPR and F1 values on the validation data are: %.4f, %.4f, %.4f' % (dev_auc, dev_aupr, dev_f1))
                print('The AUC, AUPR and F1 values on the testing data are: %.4f, %.4f, %.4f' % (test_auc, test_aupr, test_f1))
                print('The training loss is: %.4f' % loss_mean)

        train_auc_kf_list.append(best_train_auc)
        train_f1_kf_list.append(best_train_f1)
        train_aupr_kf_list.append(best_train_aupr)

        eval_auc_kf_list.append(best_eval_auc)
        eval_f1_kf_list.append(best_eval_f1)
        eval_aupr_kf_list.append(best_eval_aupr)

        test_auc_kf_list.append(best_test_auc)
        test_f1_kf_list.append(best_test_f1)
        test_aupr_kf_list.append(best_test_aupr)

        loss_kf_list.append(best_loss)

        kk = kk + 1

    train_auc_kf_mean = np.mean(train_auc_kf_list)
    train_f1_kf_mean = np.mean(train_f1_kf_list)
    train_aupr_kf_mean = np.mean(train_aupr_kf_list)

    eval_auc_kf_mean = np.mean(eval_auc_kf_list)
    eval_f1_kf_mean = np.mean(eval_f1_kf_list)
    eval_aupr_kf_mean = np.mean(eval_aupr_kf_list)

    test_auc_kf_mean = np.mean(test_auc_kf_list)
    test_f1_kf_mean = np.mean(test_f1_kf_list)
    test_aupr_kf_mean = np.mean(test_aupr_kf_list)

    loss_kf_mean = np.mean(loss_kf_list)
    print("-" * 100)
    print('%.d kk_fold final results' % kk)
    print('The std of AUC, AUPR and F1 values on the training data are: %.4f, %.4f, %.4f' % (np.std(train_auc_kf_list), np.std(train_aupr_kf_list), np.std(train_f1_kf_list)))
    print('The std of AUC, AUPR and F1 values on the validation data are: %.4f, %.4f, %.4f' % (np.std(eval_auc_kf_list), np.std(eval_aupr_kf_list), np.std(eval_f1_kf_list)))
    print('The std of AUC, AUPR and F1 values on the testing data are: %.4f, %.4f, %.4f' % (np.std(test_auc_kf_list), np.std(test_aupr_kf_list), np.std(test_f1_kf_list)))
    print('The std of training loss is: %.4f' % np.std(loss_kf_list))

    return loss_kf_mean, train_auc_kf_mean, train_f1_kf_mean, train_aupr_kf_mean, eval_auc_kf_mean, eval_f1_kf_mean, eval_aupr_kf_mean, test_auc_kf_mean, test_f1_kf_mean, test_aupr_kf_mean


def ctr_eval(args, model, data):
    model.eval()
    eval_feature = data['feature']
    eval_label = data['label']
    eval_index = np.where(eval_label != -1)

    # filter the unknow label
    eval_label = eval_label[eval_index]
    eval_feature = eval_feature[eval_index]

    eval_dataset = SLDataset('eval', eval_feature, eval_label)
    eval_dataloader = DataLoader(eval_dataset, args.batch_size, shuffle=True, pin_memory=True)

    preds = []
    labels = []

    score_list = []
    score_binary_list = []
    auc_list = []
    aupr_list = []
    f1_list = []

    for batch in eval_dataloader:  # iterate through the dataloader
        x, y = batch
        x = x.to(args.device)
        with torch.no_grad():
            pred = model(x)

            # scores_output, scores_binary_output, auc, f1, aupr, acc = model.cal_eval(y.numpy(), pred.detach().cpu().reshape(-1).numpy())
            # score_list.append(scores_output)
            # score_binary_list.append(scores_binary_output)
            # auc_list.append(auc)
            # aupr_list.append(aupr)
            # f1_list.append(f1)

            preds.append(pred.detach().cpu())
            labels.append(y)

    preds = torch.cat(preds, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    auc = roc_auc_score(labels, preds.reshape(-1))
    aupr = average_precision_score(labels, preds.reshape(-1))

    scores = preds.reshape(-1)
    scores_output = scores.copy()

    scores[scores >= 0.5] = 1
    scores[scores < 0.5] = 0

    f1 = f1_score(labels, scores)
    acc = accuracy_score(labels, scores)

    scores_binary_output = scores

    return scores, scores_binary_output, auc, f1, aupr, acc

