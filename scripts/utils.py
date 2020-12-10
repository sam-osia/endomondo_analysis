import io
import numpy as np
import pandas as pd
import os
import pickle
import json
import sys
from pprint import pprint
from pathlib import Path
import time
import matplotlib.pyplot as plt


def rescale(heart_rates_not_scaled, sigma, meow):
    '''
        This function rescales the predicted heart rates in to BPM based on the population meow and sigma
    '''
    heart_rates_scaled = []
    for i in range(len(heart_rates_not_scaled)):
        hr = np.array(heart_rates_not_scaled[i])
        
        hr = hr*sigma + meow
        
        heart_rates_scaled.append(list(hr))
        
        
    return heart_rates_scaled


def set_path(user):
    if user == 'sayeh':
        os.chdir('/Users/sayehbayat/Documents/GIT/endomondo_analysis/')
    elif user == 'kasra':
        os.chdir(os.chdir(os.path.dirname(sys.argv[0])))
    elif user == 'saman':
        os.chdir('..')


def override(f):
    return f


def get_log_dir(parent_dir, model_name):
    run_id = time.strftime(f'{model_name}_%Y_%m_%d-%H_%M_%S')
    return os.path.join(parent_dir, run_id)


def get_save_dir(parent_dir, model_name):
    run_id = time.strftime(f'{model_name}_%Y_%m_%d-%H_%M_%S.h5')
    return os.path.join(parent_dir, run_id)

def pkl_load(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(file_path)
        print(type(data))
        print(data.shape)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def io_load(file_path):
    with io.open(file_path, 'r', encoding='windows-1252') as f:
        for line in f:
            print(line)
            break


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def pd_load(file_path):
    df = pd.read_pickle(file_path)
    print(df.shape)
    print(df.head())


def create_chunk(rows=5000, save=True):
    # Since the dataset downloaded is in json format, apply the following function to open the file, then transform it into a csv file.
    reviews = []
    with open('./data/processed_endomondoHR_proper_interpolate.json', 'r') as train_file:
        for i, l in enumerate(train_file):
            reviews.append(json.loads(l.strip()))
            if rows is not None:
                if i > rows:
                    break

    df = pd.DataFrame.from_dict(reviews)

    if save:
        df.to_json('./data/data_chunk.json')

    return df

if __name__ == '__main__':
    set_path('sayeh')
    #print(os.getcwd())
    create_chunk()
