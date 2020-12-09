import itertools
import random
import json

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, Lambda, Activation, Embedding, Input, Dropout, LSTM, concatenate

from sklearn import preprocessing

from preprocess import *
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import os

from utils import *


class BaseModel:
    def __init__(self, model_name='base_model', run_id=None, df_paths=None, generate_hyperparams=False):
        assert run_id is not None
        self.model_name = model_name
        self.run_id = run_id
        self.df_paths = df_paths

        self.model_dir = f'./models/{self.model_name}'
        self.hyperparams_dir = os.path.join(self.model_dir, 'hyperparameters')
        self.weights_dir = os.path.join(self.model_dir, 'model_weights')

        self.hyperparams_range = {}

        mkdir(self.model_dir)
        mkdir(self.hyperparams_dir)
        mkdir(self.weights_dir)

    def generate_hyperparams(self):
        keys, vals = zip(*self.hyperparams_range.items())
        trials = [dict(zip(keys, v)) for v in itertools.product(*vals)]

        trial_params = random.sample(trials, 100)

        params_dir = os.path.join(self.model_dir, 'hyperparameters')
        mkdir(params_dir)

        for i, trial_param in enumerate(trial_params):
            with open(os.path.join(params_dir, f'run{i}.json'), 'w') as f:
                json.dump(trial_param, f)
        return

    def create_model(self, **kwargs):
        return Sequential()

    def run_pipeline(self):
        self.parse_hyperparams()
        self.inputs, self.labels = self.preprocess()
        self.hyperparams = self.parse_hyperparams()

        model = self.create_model(**self.hyperparams)

        logdir = get_log_dir(f'{self.model_dir}/tb_logs', self.model_name)
        savedir = get_save_dir(f'{self.model_dir}/model_weights', self.model_name)

        early_stopping_cb = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        tensorboard_cb = keras.callbacks.TensorBoard(logdir)

        model.fit(self.inputs, self.labels,
                  epochs=1000,
                  callbacks=[early_stopping_cb, tensorboard_cb],
                  validation_split=0.2)

        model.save(savedir)

    def preprocess(self):
        inputs = []
        outputs = []
        return inputs, outputs


    def load_data(self):
        df = None

        if isinstance(self.df_paths, str):
            df = pd.read_json(self.df_paths)

        else:
            for df_path in self.df_paths:
                df_temp = pd.read_json(df_path)
                if 'female' not in df_path:
                    df_temp = df_temp.sample(frac=0.5)
                if df is None:
                    df = df_temp
                else:
                    df = pd.concat([df, df_temp])

        df = shuffle(df)
        return df

    def fit_model(self):
        pass

    def parse_hyperparams(self):
        hyperparams = json.loads(os.path.join(self.hyperparams_dir, f'run{self.run_id}.json'))
        return hyperparams



if __name__ == '__main__':
    set_path('saman')
    data_names = ['male_run',
                  'female_run',
                  'male_bike',
                  'female_bike']

    df = None
    for data_name in data_names:
        print(f'./data/{data_name}.json')
        df_temp = pd.read_json(f'./data/{data_name}.json')
        if 'male' in data_name:
            df_temp = df_temp.sample(frac=0.5)
        if df is None:
            df = df_temp
        else:
            df = pd.concat([df, df_temp])
        print(len(df_temp))
        print(len(df))
        print()

    df = shuffle(df)
    model = keras.models.load_model(
        './models/more_complicated_architecture/model_weights/more_complicated_architecture_2020_12_07-18_42_50.h5')

    # more_complicated_model = MoreComplicatedModel(-1)

    _, sport_feature = process_catData(df, 'sport')
    _, gender_feature = process_catData(df, 'gender')
    temporal_features, labels = create_time_series_data(df, 3)

    inputs = [sport_feature, gender_feature, temporal_features]

    for i in range(9):
        inputs = [sport_feature[i].reshape(1, 300, 1),
                  gender_feature[i].reshape(1, 300, 1),
                  temporal_features[i].reshape(1, 300, 3)]

        sport = sport_feature[i][0]
        gender = gender_feature[i][0]

        pred = model.predict(inputs).reshape(-1)
        actual = labels[i]
        plt.subplot(3, 3, i+1)
        plt.plot(actual, color='r', label='actual')
        plt.plot(pred, color='b', label='pred')
        plt.title(f'Gender: {gender}, Sport: {sport}')


    plt.show()

    # more_complicated_model.run_model(inputs, labels)
