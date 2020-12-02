import numpy as np
import pandas as pd
import os


os.chdir('..')

df = np.load('./data/processed_interpolate.npy', allow_pickle=True)
print(df)