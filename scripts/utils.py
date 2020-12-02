import gdown
import os


def pull_data():
    processed_interpolate_npy = 'https://drive.google.com/u/0/uc?id=1L0BqpXtYrLyrG7A9JP7w0ACvTuRTXhxT'
    output = './data/processed_interpolate.npy'
    gdown.download(processed_interpolate_npy, output, quiet=False)


if __name__ == '__main__':
    os.chdir('..')
    pull_data()