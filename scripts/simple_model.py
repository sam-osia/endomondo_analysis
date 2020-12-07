import itertools
import random
import json

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, Lambda, Activation, Embedding, Input, Dropout, LSTM, concatenate

from sklearn import preprocessing

import numpy as np
import pandas as pd
import os

from utils import *


class ContextLSTM:
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
        print(model.summary())

    def run_model(self):
        self.parse_hyperparams()
        model = self.create_model()
        pass

    def parse_hyperparams(self):
        self.lstm_neurons = 100
        pass



if __name__ == '__main__':
    set_path('saman')
    simple_model = ContextLSTM(-1)
    simple_model.run_model()
    # simple_model.generate_hyperparams()