import tensorflow as tf
import argparse
import os
from model import Model
from six.moves import cPickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='save')
    parser.add_argument('-n', type=int, default=25)
    parser.add_argument('--prime', type=str, default=' ')
    parser.add_argument('--sample', type=int, default=1)

    args = parser.parse_args()
    sample(args)


def sample(args):
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.save_dir, 'words_vocab.pkl'), 'rb') as f:
        words, vocab = cPickle.load(f)
    model = Model(saved_args, True)
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print(model.sample(sess, words, vocab, args.n, args.prime, args.sample))

if __name__ == '__main__':
    main()