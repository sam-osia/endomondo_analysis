import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, Model, load_model, save_model
from tensorflow.keras.layers import Dense, Lambda, Activation
from tensorflow.keras.layers import Embedding, Input, Dense, Reshape, Flatten, MaxPooling1D, Dropout, LSTM

from sklearn import preprocessing

import numpy as np
import pandas as pd
import os

from utils import *


def create_model():
    categorical_feature = ['userId', 'sport', 'gender']
    temporal_feature = ['distance, altitude', 'time_elapsed']     # input_dim in their code
    target_feature = ['heart_rate']

    n_categorical = len(categorical_feature)
    n_time_series = len(temporal_feature)

    num_users = 51
    num_steps = 300
    embedding_dim = 5

    # main input layer:
    main_input = Input(shape=(num_steps, n_time_series), name='main_input')

    # categorical features layers:
    for category in categorical_feature:
        input = Input(shape=(num_steps, 1), name=f'{category}_input')
        embedding = Embedding(input_dim=num_users, output_dim=embedding_dim, name=f'{category}_embedding')(input)
        embedding = Lambda(lambda y: tf.squeeze(y, 2))(embedding)

    # temporal features layers:
    # all temporal variables + target variable from previous workout





set_path('saman')

df = pd.read_json('./data/data_chunk.json')
le = preprocessing.LabelEncoder()
le.fit(df['userId'])
df['userCat'] = le.transform(df['userId'])
print(df['userCat'])


model = create_model()
input = np.array(df['userCat']).astype('int64')
shape_var = model(tf.convert_to_tensor(input))
print(shape_var.shape)
