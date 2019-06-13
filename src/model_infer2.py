# -*- coding: utf-8 -*-
"""
Created on Thu May 30 10:15:28 2019

@author: WT
"""

import sys
import os
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import RobustScaler
import torch
from torch.utils.data import Dataset, DataLoader
from models import DenseNetV2
from engineer_conv import convert_timeseries

def load_pickle(filename):
    completeName = os.path.join("./submission_data/",\
                                filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

def save_as_pickle(filename, data):
    completeName = os.path.join("./submission_data/",\
                                filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)
        
class series_infer(Dataset):
    def __init__(self, df_series, features=['Accuracy', 'Bearing', 'Speed', "acceleration",\
                                            "gyro_x","gyro_y","gyro_z"], bin_length=30):
        self.series = torch.tensor(np.array(df_series[features]), dtype=torch.float32,\
                                   requires_grad=False)
        self.labels = torch.tensor(df_series["label"].astype(int), requires_grad=False)
        self.booking_id = torch.tensor(df_series["bookingID"], dtype=torch.int64, \
                                       requires_grad=False).long()
        self.bin_length = bin_length
        
    def __len__(self):
        return int(len(self.series)/self.bin_length)
    
    def __getitem__(self, idx):
        return self.series[idx*self.bin_length:(idx*self.bin_length + self.bin_length),:],\
                self.labels[idx*self.bin_length], self.booking_id[idx*self.bin_length]

def eval_inference(df_labels):
    data = "./data/"
    labels = pd.read_csv(os.path.join(data + "labels/", \
                                  "part-00000-e9445087-aa0a-433b-a7f6-7f4c19d78ad6-c000.csv"))
    dl = pd.merge(df_labels, labels, how="left", on="bookingID")
    acc = 0
    for i in range(len(dl)):
        b_id, x, y = dl.iloc[i];
        if x == y:
            acc += 1
    return acc/len(dl)

if __name__ == "__main__":
    ### loads test file
    test_file = sys.argv[1] ### test csv filename
    df = pd.read_csv(os.path.join("./submission_data/", test_file))
    
    ### Data processing and Engineering ######################
    print("Engineering Features...")
    df.drop_duplicates(inplace=True)
    df["label"] = 0 ### add dummy labels
    
    df_series = convert_timeseries(df, seq_len=1200, interval=3)
    
    ### Standardize data
    print("Scaling data...")
    scaler = load_pickle("scaler.pkl")
    X = scaler.transform(df_series[['Accuracy', 'Bearing', 'Speed', "acceleration",\
                                        "gyro_x","gyro_y","gyro_z"]])
    X = pd.DataFrame(data=X, columns=['Accuracy', 'Bearing', 'Speed', "acceleration",\
                                      "gyro_x","gyro_y","gyro_z"])
    X["bookingID"] = df_series["bookingID"]
    X["label"] = df_series["label"]
    X = X[["bookingID", 'Accuracy', 'Bearing', 'Speed', "acceleration",\
           "gyro_x","gyro_y","gyro_z", "label"]]
    save_as_pickle("X_infer.pkl", X)    
    print("Done!")
    
    ### Inference ############################################
    ### loads test data and model
    print("Loading data and model...")
    batch_size = 25
    trainset = series_infer(X, bin_length=1200)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    cuda = torch.cuda.is_available()
    net = DenseNetV2(features_size=len(df_series.columns)-2, c_in=1, c_out=16, batch_size=batch_size)
    if cuda:
        net.cuda()
    checkpoint = torch.load(os.path.join("./submission_data/","test_model_best_%d.pth.tar" % 0))
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    
    b_id = []; b_label = []
    print("Inferring...")
    total_len = len(train_loader)
    for i, (seq, _, booking_id) in enumerate(train_loader):
        if (i % 10) == 0:
            print("%.3f %% completed" % (100*i/total_len))
        if cuda:
            seq = seq.cuda()
        output = net(seq)
        b_id.extend(list(booking_id.numpy()))
        if cuda:
            b_label.extend(list(torch.softmax(output, dim=1).max(1)[1].detach().cpu().numpy()))
        else:
            b_label.extend(list(torch.softmax(output, dim=1).max(1)[1].detach().numpy()))
    
    print("Completed")
    print("Saving as \"test_labels.csv\"...")
    df_labels = pd.DataFrame(columns=["bookingID","label"])
    df_labels["bookingID"] = b_id
    df_labels["label"] = b_label
    df_labels.to_csv(os.path.join("./submission_data/", "test_labels.csv"), index=False)
    try:
        print(eval_inference(df_labels))
    except:
        pass
    print("Saved and done!")