# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 23:17:47 2019

@author: WT
"""

import os
import pandas as pd
import pickle

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

if __name__ == "__main__":
    data = "./data/"
    df_series = load_pickle("df_series.pkl")
    ### remove bookingIDs with dubious labels
    df_labels = pd.read_csv(os.path.join(data + "labels/", \
                                      "part-00000-e9445087-aa0a-433b-a7f6-7f4c19d78ad6-c000.csv"))
    df_id = df_labels.groupby("bookingID")
    labels = df_id.apply(lambda x: x["label"].mean())
    df_labels.set_index("bookingID", drop=False, inplace=True)
    df_labels.drop(labels=labels[(labels != 1) & (labels !=0)].index, axis=0, inplace=True)

    ### train test split
    X = df_labels["bookingID"]
    y = df_labels["label"]
    ### stratify split data into independent parts
    y1_length = len(y[y==1]); y0_length = len(y[y==0])
    df_series.set_index("bookingID", drop=False, inplace=True)
    splits = 5
    for i in range(splits):
        test_idxs = []
        test_idxs.extend(list(y[y==0][i*y0_length//splits:((i + 1)*y0_length//splits)].index)) # 0 labels
        test_idxs.extend(list(y[y==1][i*y1_length//splits:((i + 1)*y1_length//splits)].index)) # 1 labels
        train_idxs = [a for a in y.index if a not in test_idxs]
        print("\nNo. %d split: " %(i + 1))
        print("Test_idxs: %d" % len(test_idxs))
        print("Train_idxs: %d" % len(train_idxs))
        df_series_train = df_series.loc[train_idxs]
        df_series_test = df_series.loc[test_idxs]
        save_as_pickle("df_series_train_%d.pkl" % i, df_series_train)
        save_as_pickle("df_series_test_%d.pkl" % i, df_series_test)
        print("Saved %d rows of training data." % len(df_series_train))
        print("Saved %d rows of test data." % len(df_series_test))