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
        targData[i, :, 0] = df["tar_heart_rate"][i]
    return data, targData

def process_catData(df, feature):

    df = df.reset_index(drop=True)
    le = preprocessing.LabelEncoder()
    data = np.zeros((len(df), 300, 1))
    le.fit(df[feature])
    transfrom_data = le.transform(df[feature])
    for i in range(len(data)):
        data[i, :, 0] = [transfrom_data[i]]*300
    return len(np.unique(df["userId"])), data

def scaleData(data, feature, zMultiple=1):
    flat_data = list(itertools.chain.from_iterable(df[feature].values.flatten()))
    mean, std = np.mean(flat_data), np.std(flat_data)
    diff = [d - mean for d in data[feature]]
    zScore = [d / std for d in diff]
    scaled_array = np.array([x * zMultiple for x in zScore])
    return scaled_array.reshape(len(data), 300)

def clean_time(data):
    for i, row in data.iterrows():
        time = np.array(row.timestamp)
        time -= time[0]
        data.loc[i, 'timestamp'] = list(time)
    return data
if __name__ == "__main__":
    set_path("sayeh")
    df = pd.read_json('./data/female_bike.json')
    print(df.columns)

    data = []
    #with open('./data/processed_endomondoHR_proper_interpolate.json', 'r') as train_file:
    #    for l in train_file:
    #        data.append(json.loads(l.strip()))
    #df = pd.DataFrame.from_dict(data)

    targetDim = 3
    formatted_data, targData = create_time_series_data(df, targetDim)
    print("Input data shape:", formatted_data.shape)
    print("Output data shape:", targData.shape)

    nUser, formatted_gender = process_catData(df, 'gender')
    print("Number of Users:", nUser)
    print("Transformed Gender:", formatted_gender)

    nUser, formatted_sport = process_catData(df, 'sport')
    print("Number of Users:", nUser)
    print("Transformed Sport:", formatted_sport.shape)

    scaled_speed = scaleData(df, 'tar_derived_speed', zMultiple=1)
    print(scaled_speed.shape)
