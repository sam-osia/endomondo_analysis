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
        self.model_name = 'fitrec_paper_architecture'
        self.model_dir = f'./models/{self.model_name}'
        self.run_id = run_id
        mkdir(self.model_dir)

    def generate_hyperparams(self):
        hyperparams = {
            'embedding_dim': np.arange(3, 10).tolist(),
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
        categorical_features = ['userId', 'sport', 'gender']
        temporal_features = ['distance', 'altitude', 'time_elapsed']     # input_dim in their code
        target_features = ['heart_rate']

        n_categorical = len(categorical_features)
        n_temporal = len(temporal_features)
        n_target = len(target_features)

        num_users = 51
        num_steps = 300
        embedding_dim = 5
        lstm_dim = 50
        dense_context_dim = 50    # dimension for dense layer which accepts
        dropout_rate = 0.2

        # initialize array that the model expects as an input
        user_inputs = []

        categorical_embedding = []
        # categorical features layers:
        for category in categorical_features:
            input = Input(shape=(num_steps, 1), name=f'{category}_input')
            user_inputs.append(input)

            embedding = Embedding(input_dim=num_users, output_dim=embedding_dim, name=f'{category}_embedding')(input)
            embedding = Lambda(lambda y: tf.squeeze(y, 2))(embedding)

            # predict_vector = concatenate([predict_vector, embedding])
            categorical_embedding.append(embedding)

        # temporal features layers:

        # feature 1 - wtf is this?
        temporal_input_1 = Input(shape=(num_steps, n_temporal + 1), name='temporal_input_1')    # add 1 for since_last
        user_inputs.append(temporal_input_1)
        temporal_lstm_1 = LSTM(lstm_dim, return_sequences=True, name='temporal_lstm_1')(temporal_input_1)

        # feature 2 - wtf is this?
        temporal_input_2 = Input(shape=(num_steps, n_target), name='temporal_input_2')
        user_inputs.append(temporal_input_2)
        temporal_lstm_2 = LSTM(lstm_dim, return_sequences=True, name='temporal_lstm_2')(temporal_input_2)

        # connect the output of the temporal features and pass it through a dense to learn stuff
        context_vector = concatenate([temporal_lstm_1, temporal_lstm_2])
        context_vector = Dense(dense_context_dim, activation='relu', name='context_projection')(context_vector)

        # connect the categorical and temporal embeddings
        predict_vector = concatenate([categorical_embedding[0],
                                      categorical_embedding[1],
                                      categorical_embedding[2],
                                      context_vector])

        # pass predict vector through two LSTMs to learn stuff
        layer_1 = LSTM(lstm_dim, return_sequences=True, name='layer_1')(predict_vector)
        dropout_1 = Dropout(dropout_rate, name='dropout_1')(layer_1)
        layer_2 = LSTM(lstm_dim, return_sequences=True, name='layer_2')(dropout_1)
        dropout_2 = Dropout(dropout_rate, name='dropout_2')(layer_2)
        output = Dense(n_target, activation='relu', name='output')(dropout_2)

        model = keras.Model(inputs=user_inputs, outputs=[output])
        print(model.summary())

        return model


    def run_model(self):
        pass

    def parse_hyperparams(self):
        pass



if __name__ == '__main__':
    set_path('saman')
    context_lstm = ContextLSTM(-1)
    context_lstm.create_model()


'''
set_path('saman')
model = ContextLSTM.create_model()

exit()

df = pd.read_json('./data/data_chunk.json')
le = preprocessing.LabelEncoder()
le.fit(df['userId'])
df['userCat'] = le.transform(df['userId'])
print(df['userCat'])

input = np.array(df['userCat']).astype('int64')
shape_var = model(tf.convert_to_tensor(input))
print(shape_var.shape)
'''