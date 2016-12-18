import reader
import tensorflow as tf
import argparse
from model import Model
import time
import os
from six.moves import cPickle
import json


def graph_save(filename, train_loss):
    data = {
        "train_loss": train_loss
    }
    f = open(filename, "w")
    json.dump(data, f)
    f.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./')
    parser.add_argument('--save_dir', type=str, default='save')
    parser.add_argument('--rnn_size', type=int, default=256)
    parser.add_argument('--keep_prob', type=int, default=0.5)
    parser.add_argument('--vocab_cut', type=int, default=40000)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--model', type=str, default='lstm')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--seq_length', type=int, default=25)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=10000)
    args = parser.parse_args()
    graph_train_loss = []

    raw_data = reader.raw_data(args.vocab_cut, args.data_dir)
    train_data, vocab, words, vocab_size = raw_data
    args.vocab_size = vocab_size

    num_batches = int(train_data.size / (args.batch_size * args.seq_length))
    x_batches, y_batches = reader.create_batches(train_data, num_batches=num_batches,
                                                 batch_size=args.batch_size, seq_length=args.seq_length)

    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(args, f)
    with open(os.path.join(args.save_dir, 'words_vocab.pkl'), 'wb') as f:
        cPickle.dump((words, vocab), f)

    model = Model(args)

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())

        for e in xrange(args.num_epochs):
            batch_pointer = 0
            state = sess.run(model.initial_state)
            tmp_train_loss = 0
            for b in xrange(num_batches):
                start = time.time()
                x, y = x_batches[batch_pointer], y_batches[batch_pointer]
                batch_pointer += 1
                feed = {model.input_data: x, model.targets: y, model.initial_state: state}
                train_loss, state, _ = sess.run([model.cost, model.final_state, model.train_op], feed)
                end = time.time()
                tmp_train_loss = train_loss
                print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                      .format(e * num_batches + b,
                              args.num_epochs * num_batches,
                              e, train_loss, end - start))
                if (e * num_batches + b) % args.save_every == 0 \
                        or (e==args.num_epochs-1 and b == num_batches-1): # save for the last result
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step = e * num_batches + b)
                    print("model saved to {}".format(checkpoint_path))
            graph_train_loss.append(str(tmp_train_loss))
        graph_save(args.model+".json", graph_train_loss)

if __name__ == '__main__':
    main()