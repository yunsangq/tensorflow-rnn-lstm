import reader
import tensorflow as tf

data_path = './'


def train():
    raw_data = reader.raw_data(data_path)
    train_data, _ = raw_data


if __name__ == '__main__':
    train()