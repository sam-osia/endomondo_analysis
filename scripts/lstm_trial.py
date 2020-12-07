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

#%%
def category_layer():
    num_users = 51
    num_steps = 300
    embedding_dim = 5

    model = Sequential()
    model.add(Input(shape=(num_steps, 1)))
    model.add(Embedding(input_dim=num_users, output_dim=embedding_dim))
    model.add(Lambda(lambda y: tf.squeeze(y, 2)))

    print(model.summary())
    return model


set_path('sayeh')
df = pd.read_json('./data/data_chunk.json')
le = preprocessing.LabelEncoder()
le.fit(df['userId'])
df['userCat'] = le.transform(df['userId'])
print(df['userCat'])


model = category_layer()
print(df['userCat'].shape)

input = np.array(df['userCat']).astype('int64')
shape_var = model(tf.convert_to_tensor(input))
print(shape_var.shape)
