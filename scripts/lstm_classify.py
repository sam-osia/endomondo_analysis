import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, Lambda, Activation, Embedding, Input, Dropout, LSTM, concatenate

from sklearn import preprocessing

import numpy as np
import pandas as pd
import os

from utils import *


def create_model():
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
    output_dim = 300

    # initialize array that the model expects as an input
    user_inputs = []

    # main input layer:
    main_input = Input(shape=(num_steps, n_temporal), name='main_input')
    # link predict vector to input. We setting up a concat layer to connect other inputs later
    predict_vector = main_input


    model = Sequential()
    model.add(Input(shape=(10, 3)))
    model.add(LSTM(100))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(5, activation='softmax'))

    input = Input()
    lstm = LSTM(100)(input)
    dense_1 = Dense(500, activation='relu')(lstm)
    output = Dense(5, activation='softmax')(dense_1)
    model = keras.Model(inputs=[input], outputs=[output])


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
    temporal_input_1 = Input(shape=(num_steps, n_temporal + 1), name='temporal_input_1')
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

set_path('saman')
model = create_model()

exit()

df = pd.read_json('./data/data_chunk.json')
le = preprocessing.LabelEncoder()
le.fit(df['userId'])
df['userCat'] = le.transform(df['userId'])
print(df['userCat'])


input = np.array(df['userCat']).astype('int64')
shape_var = model(tf.convert_to_tensor(input))
print(shape_var.shape)
