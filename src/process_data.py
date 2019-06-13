# -*- coding: utf-8 -*-
"""
Created on Tue May 28 14:31:05 2019

@author: WT
"""

import pandas as pd
import os
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
    ### combine features into one dataframe
    for idx, file in enumerate(os.listdir(data + "features/")):
        if file != ".DS_Store":
            print(file)
            filename = os.path.join(data + "features/", file)
            if idx == 1:
                df = pd.read_csv(filename)
            else:
                df = pd.concat((df, pd.read_csv(filename)), ignore_index=True)
    ### appends labels (GOT INCONSISTENT bookingID-label PAIRS!!! EG. bookingID 13)
    labels = pd.read_csv(os.path.join(data + "labels/", \
                                      "part-00000-e9445087-aa0a-433b-a7f6-7f4c19d78ad6-c000.csv"))
    df = pd.merge(df, labels, left_on="bookingID", right_on="bookingID", how="left")
    df.drop_duplicates(inplace=True)
    save_as_pickle("df.pkl", df)