# -*- coding: utf-8 -*-
"""
Created on Tue May 28 15:31:35 2019

@author: WT
"""

import pandas as pd
import os
import pickle
from sklearn.preprocessing import QuantileTransformer, RobustScaler
import time

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
    df = load_pickle("df.pkl")

    ### remove bookingIDs with dubious labels
    df_labels = pd.read_csv(os.path.join(data + "labels/", \
                                      "part-00000-e9445087-aa0a-433b-a7f6-7f4c19d78ad6-c000.csv"))
    df_id = df_labels.groupby("bookingID")
    labels = df_id.apply(lambda x: x["label"].mean())
    df.set_index("bookingID", drop=False, inplace=True)
    df.drop(labels=labels[(labels != 1) & (labels !=0)].index, axis=0, inplace=True)
    save_as_pickle("df_engineered.pkl", df)
    
    ### convert to time-series by bookingID, ignore time > 3600 s
    print("Converting to time_series...")
    #cols = ['bookingID', 'Accuracy', 'Bearing', 'Speed', 'acceleration', 'gyro', 'label']
    cols = ['bookingID', 'Accuracy', 'Bearing', 'Speed', "acceleration_x","acceleration_y",\
            "acceleration_z", "gyro_x","gyro_y","gyro_z", 'label']
    cols_p = ['bookingID', 'Accuracy', 'Bearing', 'Speed', 'gyro_x', 'gyro_y',\
              'gyro_z', 'label', 'acceleration']
    seq_len = 1200
    interval = 3
    df_series = pd.DataFrame(index=[interval*i for i in range(int(seq_len/100))], columns=cols_p)
    a = pd.to_datetime([interval*i for i in range(seq_len)], unit="s")
    total_len = len(df["bookingID"].unique()); time_counter = time.time()
    for idx, booking_id in enumerate(df["bookingID"].unique()):
        if (idx % 1000) == 0:
            print("%.3f %% completed, took %d seconds." % ((100*idx/total_len), time.time() - time_counter))
            time_counter = time.time()
        df_s = df[df["bookingID"] == booking_id]
        df_s = df_s[df_s["second"] <= 3600]
        df_s.sort_values(by="second", ascending=True, inplace=True)
        df_s["second"] = pd.to_datetime(df_s["second"], unit="s")
        df_s.set_index("second", inplace=True)
        df_dum = pd.DataFrame(data=df_s.resample("%dS" % interval).apply(lambda x: x.mean()), index=a, columns=cols)
        df_dum["bookingID"] = booking_id
        df_dum["label"] = int(df_s["label"][0])
        for c in ["acceleration_x", "acceleration_y", "acceleration_z"]: # centre acceleration data to zero
            df_dum[c] = df_dum[c] - df_dum[c].mean()
        df_dum["acceleration"] = (df_dum["acceleration_x"]**2 + df_dum["acceleration_y"]**2 +\
                            df_dum["acceleration_z"]**2)**(1/2)
        df_dum.drop(["acceleration_x","acceleration_y","acceleration_z"], axis=1, inplace=True)
        df_dum.fillna(value=0, inplace=True)
        df_series = df_series.append(df_dum, ignore_index=True)
    df_series.dropna(inplace=True)
    df_series = df_series[['bookingID', 'Accuracy', 'Bearing', 'Speed', 'acceleration', 'gyro_x', 'gyro_y',\
                           'gyro_z', 'label']]
    df_series.reset_index(inplace=True); df_series.drop(["index"],axis=1,inplace=True)
    save_as_pickle("df_series_pre.pkl", df_series)
    
    ### Standardize data
    print("Scaling data...")
    #scaler = QuantileTransformer(n_quantiles=8500, subsample=int(5e6))
    scaler = RobustScaler(quantile_range=(3.0, 97.0))
    X = scaler.fit_transform(df_series[['Accuracy', 'Bearing', 'Speed', "acceleration",\
                                        "gyro_x","gyro_y","gyro_z"]])
    X = pd.DataFrame(data=X, columns=['Accuracy', 'Bearing', 'Speed', "acceleration",\
                                      "gyro_x","gyro_y","gyro_z"])
    X["bookingID"] = df_series["bookingID"]
    X["label"] = df_series["label"]
    X = X[["bookingID", 'Accuracy', 'Bearing', 'Speed', "acceleration",\
           "gyro_x","gyro_y","gyro_z", "label"]]
    
    save_as_pickle("scaler.pkl", scaler)
    save_as_pickle("df_series.pkl", X)
    print("Done!")