import gdown
import os
from pathlib import Path


os.chdir('..')

Path('./data1/').mkdir(parents=True, exist_ok=True)

processed_interpolate_npy = 'https://drive.google.com/u/0/uc?id=1L0BqpXtYrLyrG7A9JP7w0ACvTuRTXhxT'
output = './data/processed_interpolate.npy'
gdown.download(processed_interpolate_npy, output, quiet=False)