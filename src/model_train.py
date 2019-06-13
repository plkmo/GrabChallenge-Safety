# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:25:44 2019

@author: WT
"""
import os
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from models import DenseNetV2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

def load_pickle(filename):
    completeName = os.path.join("./data/",\
                                filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

def save_as_pickle(filename, data):
    completeName = os.path.join("./data/",\
                                filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)

class series_data(Dataset):
    '''
    ['Accuracy', 'Bearing', 'Speed', 'acceleration',\
                                             'gyro']
    cols = ['bookingID', 'Accuracy', 'Bearing', 'Speed', "acceleration_x","acceleration_y",\
            "acceleration_z", "gyro_x","gyro_y","gyro_z", 'label']
    '''
    def __init__(self, df_series, features=['Accuracy', 'Bearing', 'Speed', "acceleration",\
                                            "gyro_x","gyro_y","gyro_z"], bin_length=30):
        self.series = torch.tensor(np.array(df_series[features]), dtype=torch.float32,\
                                   requires_grad=False)
        self.labels = torch.tensor(np.array(df_series["label"].astype(int)), dtype=torch.long,\
                                   requires_grad=False)
        self.bin_length = bin_length
        
    def __len__(self):
        return int(len(self.series)/self.bin_length)
    
    def __getitem__(self, idx):
        return self.series[idx*self.bin_length:(idx*self.bin_length + self.bin_length),:],\
                self.labels[idx*self.bin_length]

### Loads model and optimizer states
def load(net, optimizer, model_no, load_best=True):
    base_path = "./data/"
    if load_best == False:
        checkpoint = torch.load(os.path.join(base_path,"test_checkpoint_%d.pth.tar" % model_no))
    else:
        checkpoint = torch.load(os.path.join(base_path,"test_model_best_%d.pth.tar" % model_no))
    start_epoch = checkpoint['epoch']
    best_pred = checkpoint['best_acc']
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return start_epoch, best_pred

def evaluate(output, labels_e):
    labels = torch.softmax(output, dim=1).max(1)[1]
    return sum(labels_e == labels).item()/len(labels)

def evaluate_results(net, data_loader, cuda):
    net.eval(); acc = 0; auc = 0
    preds = []; t = []
    for i, (seq, pred) in enumerate(data_loader):
        if cuda:
            seq = seq.cuda(); pred = pred.cuda()
        output = net(seq)
        acc += evaluate(output, pred)
        preds.extend(list(pred.detach().cpu()))
        t.extend(list(torch.softmax(output,dim=1).detach().cpu()[:,1]))
    auc = roc_auc_score(preds, t)
    return acc/(i + 1), auc

if __name__ == "__main__":
    model_no = 0
    seq_len = 1200
    #df_series = load_pickle("df_series_train_%d.pkl" % model_no)
    df_series = load_pickle("df_series.pkl")
    df_test = load_pickle("df_series_test_%d.pkl" % model_no)
    batch_size = 25
    trainset = series_data(df_series, bin_length=seq_len)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    testset = series_data(df_test, bin_length=seq_len)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    
    cuda = torch.cuda.is_available()
    #net = lstm(input_size=len(df_series.columns)-2, batch_size=batch_size, lstm_hidden_size=5,\
    #           num_layers=2, cuda1=cuda, bin_length=seq_len)
    net = DenseNetV2(features_size=len(df_series.columns)-2, c_in=1, c_out=16, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30,40,50,70,80], gamma=0.5)
    if cuda:
        net.cuda()
        
    try:
        start_epoch, best_auc = load(net, optimizer, model_no, load_best=False)
    except:
        start_epoch = 0; best_auc = 0
    stop_epoch = 50; end_epoch = 50
    
    try:
        losses_per_epoch = load_pickle("test_losses_per_epoch_%d.pkl" % model_no)
        accuracy_per_epoch = load_pickle("test_accuracy_per_epoch_%d.pkl" % model_no)
        auc_per_epoch = load_pickle("test_auc_per_epoch_%d.pkl" % model_no)
        train_auc_per_epoch = load_pickle("train_auc_per_epoch_%d.pkl" % model_no)
    except:
        losses_per_epoch = []; accuracy_per_epoch = []; auc_per_epoch = []; train_auc_per_epoch = [];
        
    for e in range(start_epoch, end_epoch):
        scheduler.step()
        net.train()
        losses_per_batch = []; total_loss = 0.0
        for i, (seq, pred) in enumerate(train_loader):
            if cuda:
                seq = seq.cuda(); pred = pred.cuda()
            optimizer.zero_grad()
            output = net(seq)
            loss = criterion(output, pred)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if i % 100 == 99: # print every 100 mini-batches of size = batch_size
                losses_per_batch.append(total_loss/100)
                print('[Epoch: %d, %5d/ %d points] total loss per batch: %.7f' %
                      (e, (i + 1)*batch_size, len(trainset), total_loss/100))
                total_loss = 0.0
        losses_per_epoch.append(sum(losses_per_batch)/len(losses_per_batch))
        train_acc, train_auc = evaluate_results(net, train_loader, cuda)
        acc, auc = evaluate_results(net, test_loader, cuda)
        accuracy_per_epoch.append(acc)
        auc_per_epoch.append(auc)
        train_auc_per_epoch.append(train_auc)
        print("Losses at Epoch %d: %.7f" % (e, losses_per_epoch[-1]))
        print("Accuracy at Epoch %d: %.7f" % (e, accuracy_per_epoch[-1]))
        print("ROC AUC at Epoch %d: %.7f" % (e, auc_per_epoch[-1]))
        print("Train ACC, ROC: %.5f, %.5f" %(train_acc, train_auc))
        if auc_per_epoch[-1] > best_auc:
            best_auc = auc_per_epoch[-1]
            torch.save({
                    'epoch': e + 1,\
                    'state_dict': net.state_dict(),\
                    'best_acc': best_auc,\
                    'optimizer' : optimizer.state_dict(),\
                }, os.path.join("./data/" ,\
                    "test_model_best_%d.pth.tar" % model_no))
        if (e % 5) == 0:
            save_as_pickle("test_losses_per_epoch_%d.pkl" % model_no, losses_per_epoch)
            save_as_pickle("test_accuracy_per_epoch_%d.pkl" % model_no, accuracy_per_epoch)
            save_as_pickle("test_auc_per_epoch_%d.pkl" % model_no, auc_per_epoch)
            save_as_pickle("train_auc_per_epoch_%d.pkl" % model_no, train_auc_per_epoch)
            torch.save({
                    'epoch': e + 1,\
                    'state_dict': net.state_dict(),\
                    'best_acc': best_auc,\
                    'optimizer' : optimizer.state_dict(),\
                }, os.path.join("./data/",\
                    "test_checkpoint_%d.pth.tar" % model_no))
    
    fig = plt.figure(figsize=(13,13))
    ax = fig.add_subplot(111)
    ax.scatter([i for i in range(len(losses_per_epoch))], losses_per_epoch)
    ax.set_xlabel("Epoch", fontsize=15)
    ax.set_ylabel("Loss", fontsize=15)
    ax.set_title("Loss vs Epoch", fontsize=20)
    plt.savefig(os.path.join("./data/",\
                             "test_loss_vs_epoch_%d.png" % model_no))
    
    fig = plt.figure(figsize=(13,13))
    ax = fig.add_subplot(111)
    ax.scatter([i for i in range(len(accuracy_per_epoch))], accuracy_per_epoch)
    ax.set_xlabel("Epoch", fontsize=15)
    ax.set_ylabel("Accuracy", fontsize=15)
    ax.set_title("Accuracy vs Epoch", fontsize=20)
    plt.savefig(os.path.join("./data/",\
                             "test_Accuracy_vs_epoch_%d.png" % model_no))
    
    fig = plt.figure(figsize=(13,13))
    ax = fig.add_subplot(111)
    ax.scatter([i for i in range(len(auc_per_epoch))], auc_per_epoch)
    ax.set_xlabel("Epoch", fontsize=15)
    ax.set_ylabel("Test ROC AUC", fontsize=15)
    ax.set_title("Test ROC AUC vs Epoch", fontsize=20)
    plt.savefig(os.path.join("./data/",\
                             "test_auc_vs_epoch_%d.png" % model_no))
    
    fig = plt.figure(figsize=(13,13))
    ax = fig.add_subplot(111)
    ax.scatter([i for i in range(len(train_auc_per_epoch))], train_auc_per_epoch)
    ax.set_xlabel("Epoch", fontsize=15)
    ax.set_ylabel("Train ROC AUC", fontsize=15)
    ax.set_title("Train ROC AUC vs Epoch", fontsize=20)
    plt.savefig(os.path.join("./data/",\
                             "train_auc_vs_epoch_%d.png" % model_no))