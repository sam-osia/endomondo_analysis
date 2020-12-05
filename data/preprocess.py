from collections import defaultdict
import json
import gzip
import pandas as pd
import numpy as np
import itertools


def create_time_series_data(df, targetDim):
    df = df.reset_index(drop=True)
    nRows = len(df)
    print(nRows)
    s = (nRows,targetDim*300)
    data = np.zeros(s)
    targData = np.zeros((nRows, 300))
    for i in range(nRows):
        data[i, :] = list(itertools.chain(df["time_elapsed"][i], df["distance"][i], df["altitude"][i]))
        targData[i,:] = df["heart_rate"][i]
    return data, targData

if __name__ == "__main__":
    df = pd.read_json('/Users/sayehbayat/Downloads/female_bike.json')
    targetDim = 3
    formatted_data, targData = create_time_series_data(df, targetDim*300)
    print(formatted_data.shape)
    print(targData.shape)