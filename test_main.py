import test_reader
import tensorflow as tf


learning_rate = 0.01
epochs = 10
batch_size = 50
n_hidden = 256
num_layers = 2
seq_max_len = 40
vocabulary_size = 8000

reader = test_reader.SequenceData(seq_max_len, vocabulary_size)
x_data, y_data, seqlen_data = reader.get_raw_data()

cell_fn = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
cell = tf.nn.rnn_cell.MultiRNNCell([cell_fn]*num_layers)

outputs, last_states = tf.nn.dynamic_rnn(
    cell=cell,
    dtype=tf.int32,
    sequence_length=seqlen_data,
    inputs=x_data
)

result = tf.contrib.learn.run_n(
    {"outputs": y_data, "last_states": last_states},
    n=1,
    feed_dict=None
)