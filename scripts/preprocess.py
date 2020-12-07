from collections import defaultdict
import json
import gzip
import pandas as pd
import numpy as np
import itertools
from utils import *
from sklearn import preprocessing



def create_time_series_data(df, targetDim):
    df = df.reset_index(drop=True)
    data = np.zeros((len(df), 300, 3))
    targData = np.zeros((len(df), 300, 1))
    for i in range(len(df)):
        data[i, :, 0] = df["time_elapsed"][i]
        data[i, :, 1] = df["distance"][i]
        data[i, :, 2] = df["altitude"][i]
        targData[i, :, 0] = df["heart_rate"][i]
    return data, targData

def process_catData(df, feature):

    df = df.reset_index(drop=True)
    le = preprocessing.LabelEncoder()
    data = np.zeros((len(df), 1, 1))
    le.fit(df[feature])
    data[:, 0, 0] = le.transform(df[feature])

    return len(np.unique(df["userId"])), data
if __name__ == "__main__":
    set_path("saman")
    df = pd.read_json('./data/female_bike.json')

    data = []
    #with open('./data/processed_endomondoHR_proper_interpolate.json', 'r') as train_file:
    #    for l in train_file:
    #        data.append(json.loads(l.strip()))
    #df = pd.DataFrame.from_dict(data)
    print(df.columns)
    targetDim = 3
    formatted_data, targData = create_time_series_data(df, targetDim)
    print("Input data shape:", formatted_data.shape)
    print("Output data shape:", targData.shape)

    nUser, formatted_gender = process_catData(df, 'gender')
    print("Number of Users:", nUser)
    print("Transformed Gender:", formatted_gender)

    nUser, formatted_sport = process_catData(df, 'sport')
    print("Number of Users:", nUser)
    print("Transformed Sport:", np.unique(formatted_sport ))
