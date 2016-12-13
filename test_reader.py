import nltk
import csv
import itertools
import os
from six.moves import cPickle
import numpy as np


class SequenceData(object):
    def __init__(self, max_seqlen=40, vocabulary_size=8000):
        self.data = []
        self.labels = []
        self.seqlen = None

        print "Reading CSV file..."
        with open('reddit_comments_15000.csv', 'rb') as f:
            reader = csv.reader(f, skipinitialspace=True)
            reader.next()
            sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
            sentences = ["%s" % x for x in sentences]
        print "Parsed %d sentences." % (len(sentences))
        tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
        word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
        print "Found %d unique words tokens." % len(word_freq.items())
        vocab = word_freq.most_common(vocabulary_size)
        index_to_word = [x[0] for x in vocab]
        word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])
        self.save_vocab(vocab, word_to_index)
        print "Using vocabulary size %d." % vocabulary_size
        print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])

        dataset = []
        for sent in tokenized_sentences:
            if len(sent) < max_seqlen:
                dataset.append([word_to_index[w] for w in sent if w in word_to_index])

        self.seqlen = [len(sent) for sent in dataset]
        self.data = [np.pad(x, (0, max_seqlen - len(x)), 'constant') for x in dataset]
        self.labels = [np.pad(x[1:], (0, max_seqlen - len(x[1:])), 'constant') for x in dataset]

    def save_vocab(self, vocab, word_to_index):
        with open(os.path.join('test_save', 'words_vocab.pkl'), 'wb') as f:
            cPickle.dump((vocab, word_to_index), f)

    def get_raw_data(self):
        return self.data, self.labels, self.seqlen