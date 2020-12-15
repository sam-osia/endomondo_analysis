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
from prediction_analysis import find_n_plot
from preprocess import *
from utils import *


class MakesSenseVisuallyModel(BaseModel):
    def __init__(self, run_id=None, df_paths=None, generate_hyperparams=False, testing=False, hyperparams=None,
                 load_existing=False, preprocessed_name=None, model_tag=None):
        super(MakesSenseVisuallyModel, self).__init__('makes_sense_visually', run_id, df_paths, generate_hyperparams,
                                                      load_existing, preprocessed_name, model_tag)

        self.hyperparams_range = {
            'lstm_dim': np.arange(30, 101, 10).tolist(),
            'dense_context_dim': np.arange(30, 101, 10).tolist(),
            'dropout_rate': np.arange(0, 0.51, 0.25).tolist()
        }

        self.testing = testing
        self.hyperparams = hyperparams
        if generate_hyperparams:
            self.generate_hyperparams()

    @override
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
        # lstm_temporal = LSTM(lstm_neurons, return_sequences=True, name='lstm_temporal')(input_temporal)

        input_temporal_prev = Input(shape=(num_steps, 4), name='temporal_input_prev')
        predict_layers.append(input_temporal_prev)
        user_inputs.append(input_temporal_prev)
        # lstm_temporal_prev = LSTM(lstm_neurons, return_sequences=True, name='lstm_temporal_prev')(input_temporal_prev)

        # context_concat = concatenate([lstm_temporal, lstm_temporal_prev])

        # dense_temporal = Dense(dense_neurons, activation='relu', name='temporal_dense')(context_concat)
        # predict_layers.append(dense_temporal)

        predict_layers.append(input_temporal)
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
    def preprocess(self, categorical_features, **kwargs):
        if not self.load_existing:
            df = super(MakesSenseVisuallyModel, self).load_data()
            [input_speed, input_alt, input_gender, input_sport, input_user, input_time_last, prevData, targData] = \
                curr_preprocess(df)
        else:
            [input_speed, input_alt, input_gender, input_sport, input_user, input_time_last, prevData, targData] = \
                curr_preprocess(None, load_exist=True, dataset_name=self.preprocessed_path)

        input_temporal = np.dstack([input_speed, input_alt])
        # input_prev = prevData
        input_prev = np.dstack([prevData, input_time_last])
        inputs = []

        if 'sport' in categorical_features:
            inputs.append(input_sport)
        if 'gender' in categorical_features:
            inputs.append(input_gender)
        inputs.append(input_temporal)
        inputs.append(input_prev)

        labels = targData

        return inputs, labels

    @override
    def parse_hyperparams(self):
        if self.testing:
            if self.hyperparams is None:
                self.hyperparams = {
                    'categorical_features': ['sport', 'gender'],
                    'temporal_features': ['speed', 'altitude'],
                    'temporal_prev_features': ['speed', 'altitude', 'heartRate', 'sinceLast'],
                    'embedding_dim': 5,
                    'lstm_neurons': 100,
                    'dense_neurons': 100
                }
        else:
            self.hyperparams = super(MakesSenseVisuallyModel, self).parse_hyperparams()

        print(self.hyperparams)
        return self.hyperparams


if __name__ == '__main__':
    set_path('saman')
    data_paths = ['./data/male_run.json',
                  './data/female_run.json',
                  './data/male_bike.json',
                  './data/female_bike.json']


    all_hyperparams = {
        'categorical_features': ['sport', 'gender'],
        'temporal_features': ['speed', 'altitude'],
        'temporal_prev_features': ['speed', 'altitude', 'heartRate', 'sinceLast'],
        'embedding_dim': 20,
        'lstm_neurons': 200,
        'dense_neurons': 200
    }

    genderless_hyperparams = {
        'categorical_features': ['sport'],
        'temporal_features': ['speed', 'altitude'],
        'temporal_prev_features': ['speed', 'altitude', 'heartRate', 'sinceLast'],
        'embedding_dim': 5,
        'lstm_neurons': 100,
        'dense_neurons': 100
    }

    sportless_hyperparams = {
        'categorical_features': ['gender'],
        'temporal_features': ['speed', 'altitude'],
        'temporal_prev_features': ['speed', 'altitude', 'heartRate', 'sinceLast'],
        'embedding_dim': 20,
        'lstm_neurons': 200,
        'dense_neurons': 200
    }

    preprocessed_data = {'all': all_hyperparams,
                         'male': genderless_hyperparams,
                         'female': genderless_hyperparams,
                         'run': sportless_hyperparams,
                         'bike': sportless_hyperparams}

    preprocessed_name = 'all'
    hyperparams = preprocessed_data[preprocessed_name]

    model = MakesSenseVisuallyModel(run_id=-1,
                                    df_paths=data_paths,
                                    generate_hyperparams=False,
                                    testing=True,
                                    load_existing=True,
                                    preprocessed_name=preprocessed_name,
                                    model_tag=preprocessed_name,
                                    hyperparams=hyperparams)

    model.run_pipeline()

    # data_names = ['male_run',
    #               'female_run',
    #               'male_bike',
    #               'female_bike']
    #
    # data_names = ['female_bike']
    #
    # df = None
    # for data_name in data_names:
    #     print(f'./data/{data_name}.json')
    #     df_temp = pd.read_json(f'./data/{data_name}.json')
    #     if 'female' not in data_name:
    #         df_temp = df_temp.sample(frac=0.5)
    #     if df is None:
    #         df = df_temp
    #     else:
    #         df = pd.concat([df, df_temp])
    #
    # model = keras.models.load_model(
    #     './models/makes_sense_visually/model_weights/makes_sense_visually_2020_12_09-22_55_02.h5')
    #
    # [input_speed, input_alt, input_gender, input_sport, input_user, input_time_last, prevData, targData] \
    #     = curr_preprocess(df)
    #
    # input_prev = np.dstack([prevData, input_time_last])
    #
    # idx_list, errors = find_n_plot(input_speed, input_alt, input_gender, input_sport, input_user,
    #                                input_time_last, input_prev, targData, model, 12, True, True)

