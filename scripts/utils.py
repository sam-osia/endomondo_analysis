import io
import numpy as np
import pandas as pd
import os
import pickle
import json
import sys


def set_path(user):
    if user == 'sayeh':
        os.chdir(os.path.dirname(sys.argv[0]))
    elif user == 'kasra':
        os.chdir(os.chdir(os.path.dirname(sys.argv[0])))

    os.chdir('..')

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


def create_chunk(rows=5000):
    # Since the dataset downloaded is in json format, apply the following function to open the file, then transform it into a csv file.
    reviews = []
    with open('./data/processed_endomondoHR_proper_interpolate.json', 'r') as train_file:
        for i, l in enumerate(train_file):
            reviews.append(json.loads(l.strip()))
            if i > rows:
                break

    df = pd.DataFrame.from_dict(reviews)
    df.to_json('./data/data_chunk.json')

    return df

if __name__ == '__main__':
    os.chdir('..')
    create_chunk()
