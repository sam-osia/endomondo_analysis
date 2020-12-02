import io
import numpy as np
import pandas as pd
import os
import pickle
import json


def pkl_load(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(file_path)
        print(type(data))
        print(data.shape)


def io_load(file_path):
    with io.open(file_path, 'r', encoding='windows-1252') as f:
        for line in f:
            print(line)
            break

def pd_load(file_path):
    df = pd.read_pickle(file_path)
    print(df.shape)
    print(df.head())


def json_load():
    # Since the dataset downloaded is in json format, apply the following function to open the file, then transform it into a csv file.
    reviews = []
    with open('./data/processed_endomondoHR_proper_interpolate.json', 'r') as train_file:
        for l in train_file:
            reviews.append(json.loads(l.strip()))


os.chdir('..')
file_path = './data/endomondoHR_proper_metaData.pkl'
json_load()
