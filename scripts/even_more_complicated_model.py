import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, Lambda, Activation, Embedding, Input, Dropout, LSTM, concatenate

from sklearn import preprocessing
from sklearn.utils import shuffle

import numpy as np
import pandas as pd
import os

from base_model import BaseModel
from preprocess import *
from utils import *


class EvenMoreComplicatedModel(BaseModel):
    def __init__(self, run_id=None, df_paths=None, generate_hyperparams=False, testing=False):
        super(EvenMoreComplicatedModel, self).__init__('even_more_complicated', run_id, df_paths, generate_hyperparams)

        self.hyperparams_range = {
            'lstm_dim': np.arange(30, 101, 10).tolist(),
            'dense_context_dim': np.arange(30, 101, 10).tolist(),
            'dropout_rate': np.arange(0, 0.51, 0.25).tolist()
        }

        self.testing = testing
        if generate_hyperparams:
            self.generate_hyperparams()

    def create_model(self, categorical_features, embedding_dim, lstm_neurons, dense_neurons, **kwargs):
        num_steps = 300

        # initialize array that the model expects as an input
        user_inputs = []

        predict_layers = []
        # categorical features layers:
        for category in categorical_features:
            input = Input(shape=(num_steps, 1), name=f'{category}_input')
            user_inputs.append(input)
            embedding = Embedding(input_dim=2, output_dim=embedding_dim, name=f'{category}_embedding')(input)
            embedding = Lambda(lambda y: tf.squeeze(y, 2))(embedding)
            predict_layers.append(embedding)

        input_temporal = Input(shape=(num_steps, 2), name='temporal_input')
        user_inputs.append(input_temporal)
        lstm_temporal = LSTM(lstm_neurons, return_sequences=True, name='lstm_temporal')(input_temporal)

        input_temporal_prev = Input(shape=(num_steps, 3), name='temporal_input_prev')
        user_inputs.append(input_temporal_prev)
        lstm_temporal_prev = LSTM(lstm_neurons, return_sequences=True, name='lstm_temporal_prev')(input_temporal_prev)

        context_concat = concatenate([lstm_temporal, lstm_temporal_prev])

        dense_temporal = Dense(dense_neurons, activation='relu', name='temporal_dense')(context_concat)
        predict_layers.append(dense_temporal)

        predict_vector = concatenate(predict_layers)

        lstm_out1 = LSTM(lstm_neurons, return_sequences=True, name='lstm_out1')(predict_vector)
        lstm_result = LSTM(lstm_neurons, return_sequences=True, name='lstm_result')(lstm_out1)
        output = Dense(1, name='output')(lstm_result)

        model = keras.Model(inputs=user_inputs, outputs=[output])
        print(model.summary)

        model.compile(optimizer='adam',
                      loss='mse',
                      metrics=['mae'])

        print(model.summary())

        return model

    @override
    def preprocess(self):
        df = super(EvenMoreComplicatedModel, self).load_data()
        input_speed, input_alt, input_gender, input_sport, input_user, input_time_last, prevData, targData = \
            curr_preprocess(df)

        input_temporal = np.dstack([input_speed, input_alt])

        inputs = [input_sport, input_gender, input_temporal, prevData]
        labels = targData

        return inputs, labels

    @override
    def parse_hyperparams(self):
        if self.testing:
            hyperparams = {
                'categorical_features': ['sport', 'gender'],
                'temporal_features': ['speed', 'altitude'],
                'embedding_dim': 5,
                'lstm_neurons': 100,
                'dense_neurons': 100
            }
        else:
            hyperparams = super(EvenMoreComplicatedModel, self).parse_hyperparams()

        return hyperparams




if __name__ == '__main__':
    set_path('saman')
    data_paths = ['./data/male_run.json',
                  './data/female_run.json',
                  './data/male_bike.json',
                  './data/female_bike.json']

    data_paths = './data/female_bike.json'

    model = EvenMoreComplicatedModel(run_id=-1,
                                 df_paths=data_paths,
                                 generate_hyperparams=False,
                                 testing=True)
    model.run_pipeline()


