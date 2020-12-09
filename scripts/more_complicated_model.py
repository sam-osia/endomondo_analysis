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

        input_temporal = Input(shape=(num_steps, 2), name='temporal_input')
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
        input_speed, input_alt, input_gender, input_sport, input_user, input_time_last, prevData, targData = \
            curr_preprocess(df)
        input_temporal = np.dstack([input_speed, input_alt])
        inputs = [input_sport, input_gender, input_temporal]
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

    data_names = ['male_run',
                  'female_run',
                  'male_bike',
                  'female_bike']

    # df = None
    # for data_name in data_names:
    #     print(f'./data/{data_name}.json')
    #     df_temp = pd.read_json(f'./data/{data_name}.json')
    #     if 'male' in data_name:
    #         df_temp = df_temp.sample(frac=0.5)
    #     if df is None:
    #         df = df_temp
    #     else:
    #         df = pd.concat([df, df_temp])
    #     print(len(df_temp))
    #     print(len(df))
    #     print()
    #
    # model = keras.models.load_model(
    #     './models/more_complicated/model_weights/more_complicated_2020_12_08-16_29_51.h5')
    #
    # input_speed, input_alt, input_gender, input_sport, input_user, input_time_last, prevData, targData \
    #     = curr_preprocess(df)
    #
    # np.save('./data/our_preprocess.npy',
    #         [input_speed, input_alt, input_gender, input_sport, input_user, input_time_last, prevData, targData])
    #
    # input_temporal = np.dstack([input_speed, input_alt])
    #
    # idx_list = np.random.randint(1, 20000, 12)
    #
    # for i, idx in enumerate(idx_list):
    #     inputs = [input_sport[idx].reshape(1, 300, 1),
    #               input_gender[idx].reshape(1, 300, 1),
    #               input_temporal[idx].reshape(1, 300, 2)]
    #
    #     sport = input_sport[idx][0]
    #     gender = input_gender[idx][0]
    #
    #     pred = model.predict(inputs).reshape(-1)
    #     actual = targData[idx]
    #     plt.subplot(3, 4, i+1)
    #     plt.plot(actual, color='r', label='actual')
    #     plt.plot(pred, color='b', label='pred')
    #     plt.title(f'Gender: {gender}, Sport: {sport}')
    #
    # plt.legend()
    # plt.show()
