import test_reader
import tensorflow as tf


learning_rate = 0.01
epochs = 10
batch_size = 50
n_hidden = 256
num_layers = 2

reader = test_reader.SequenceData()
x_data, y_data, seqlen_data = reader.get_raw_data()

x_batches = np.split(xdata.reshape(batch_size, -1), num_batches, 1)
y_batches = np.split(ydata.reshape(batch_size, -1), num_batches, 1)


cell_fn = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
cell = tf.nn.rnn_cell.MultiRNNCell([cell_fn]*num_layers)

input_data = tf.placeholder(tf.int32, [batch_size, None])
targets = tf.placeholder(tf.int32, [batch_size, None])