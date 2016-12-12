import os
import collections
import numpy as np
import nltk
import tensorflow as tf


def _read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return nltk.word_tokenize(f.read().decode("utf-8").replace(",,,,,,,,,,,,,,,,,,,,,,,,,", " "))


def _build_vocab(filename, vocab_cut):
    data = _read_words(filename)

    counter = collections.Counter(data)
    # count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    count_pairs = counter.most_common(vocab_cut)

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id, words


def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def raw_data(vocab_cut, data_path=None):
    train_path = os.path.join(data_path, "reddit_comments_15000.csv")

    word_to_id, words = _build_vocab(train_path, vocab_cut)
    train_data = _file_to_word_ids(train_path, word_to_id)
    vocab_size = len(word_to_id)
    return np.array(train_data), word_to_id, words, vocab_size


def create_batches(raw_data, num_batches, batch_size, seq_length):
    raw_data = raw_data[:num_batches*batch_size*seq_length]
    xdata = raw_data
    ydata = np.copy(raw_data)

    ydata[:-1] = xdata[1:]
    ydata[-1] = xdata[0]
    x_batches = np.split(xdata.reshape(batch_size, -1), num_batches, 1)
    y_batches = np.split(ydata.reshape(batch_size, -1), num_batches, 1)
    return x_batches, y_batches


