import itertools
import random
import json

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, Lambda, Activation, Embedding, Input, Dropout, LSTM, concatenate

from sklearn import preprocessing
from sklearn.utils import shuffle

from preprocess import create_time_series_data
import numpy as np
import pandas as pd
import os

from utils import *


class SimpleModel:
    def __init__(self, run_id=None):
        assert run_id is not None
        self.model_name = 'simple_architecture'
        self.model_dir = f'./models/{self.model_name}'
        self.run_id = run_id
        mkdir(self.model_dir)

    def generate_hyperparams(self):
        hyperparams = {
            'lstm_dim': np.arange(30, 101, 10).tolist(),
            'dense_context_dim': np.arange(30, 101, 10).tolist(),
            'dropout_rate': np.arange(0, 0.51, 0.25).tolist()
        }

        keys, vals = zip(*hyperparams.items())
        trials = [dict(zip(keys, v)) for v in itertools.product(*vals)]

        trial_params = random.sample(trials, 100)

        params_dir = os.path.join(self.model_dir, 'hyperparameters')
        mkdir(params_dir)

        for i, trial_param in enumerate(trial_params):
            with open(os.path.join(params_dir, f'run{i}.json'), 'w') as f:
                json.dump(trial_param, f)

        return

    def create_model(self):
        temporal_features = ['distance', 'altitude', 'time_elapsed']     # input_dim in their code
        target_features = ['heart_rate']

        n_temporal = len(temporal_features)
        n_target = len(target_features)
        n_steps = 300

        input_layer = Input(shape=(n_steps, 3), name='input_time')

        lstm = LSTM(self.lstm_neurons, return_sequences=True, name='lstm')(input_layer)
        dense = Dense(1)(lstm)

        model = keras.Model(inputs=[input_layer], outputs=[dense])

        model.compile(optimizer='adam',
                      loss='mse')

        print(model.summary())

        return model

    def run_model(self, x, y):
        self.parse_hyperparams()
        model = self.create_model()

        logdir = get_log_dir(f'{self.model_dir}/tb_logs', 'simple_model')
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
        tensorboard_cb = keras.callbacks.TensorBoard(logdir)

        model.fit(x, y,
                  epochs=1000,
                  callbacks=[early_stopping_cb, tensorboard_cb],
                  validation_split=0.2)

    def parse_hyperparams(self):
        self.lstm_neurons = 100
        pass


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
        if 'female' not in data_name:
            df_temp = df_temp.sample(frac=0.5)
        if df is None:
            df = df_temp
        else:
            df = pd.concat([df, df_temp])
        print(len(df_temp))
        print(len(df))
        print()

    df = shuffle(df)

    simple_model = SimpleModel(-1)
    print('here')
    x, y = create_time_series_data(df, 3)
    simple_model.run_model(x, y)
