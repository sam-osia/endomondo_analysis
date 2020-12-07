from collections import defaultdict
import json
import gzip
import pandas as pd
import numpy as np
import itertools
from utils import *



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

if __name__ == "__main__":
    set_path("saman")
    df = pd.read_json('./data/female_bike.json')
    print(df.head())
    targetDim = 3
    formatted_data, targData = create_time_series_data(df, targetDim)
    print("Input data shape:", formatted_data.shape)
    print("Output data shape:", targData.shape)
