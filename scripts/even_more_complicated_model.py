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


class MoreComplicatedModel(BaseModel):
    def __init__(self, run_id=None, df_paths=None, generate_hyperparams=False, testing=False):
        super(MoreComplicatedModel, self).__init__('more_complicated', run_id, df_paths, generate_hyperparams)

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

        input_temporal = Input(shape=(num_steps, 3), name='temporal_input')
        user_inputs.append(input_temporal)
        lstm_temporal = LSTM(lstm_neurons, return_sequences=True, name='lstm')(input_temporal)
        dense_temporal = Dense(dense_neurons, activation='relu', name='temporal_dense')(lstm_temporal)
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

        return model

    @override
    def preprocess(self):
        df = super(MoreComplicatedModel, self).load_data()
        seqs, input_gender, input_sport, input_time_last, prevData, targData = curr_preprocess(df)
        inputs = [input_sport, input_gender, seqs]
        labels = targData

        return inputs, labels

    @override
    def parse_hyperparams(self):
        if self.testing:
            hyperparams = {
                'categorical_features': ['sport', 'gender'],
                'temporal_features': ['distance', 'altitude', 'time_elapsed'],
                'embedding_dim': 5,
                'lstm_neurons': 100,
                'dense_neurons': 100
            }
        else:
            hyperparams = super(MoreComplicatedModel, self).parse_hyperparams()

        return hyperparams




if __name__ == '__main__':
    set_path('saman')
    data_paths = ['./data/male_run.json',
                  './data/female_run.json',
                  './data/male_bike.json',
                  './data/female_bike.json']

    model = MoreComplicatedModel(run_id=-1,
                                 df_paths=data_paths,
                                 generate_hyperparams=False,
                                 testing=True)
    model.run_pipeline()


