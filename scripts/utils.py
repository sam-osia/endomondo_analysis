import io
import numpy as np
import pandas as pd
import os
import pickle


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

os.chdir('..')
file_path = './data/endomondoHR_proper_metaData.pkl'
# pkl_load(file_path)
io_load(file_path)
# pd_load(file_path)
