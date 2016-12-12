import nltk
import csv
import itertools
import os
from six.moves import cPickle
import numpy as np


class SequenceData(object):
    def __init__(self, vocabulary_size=40000):
        self.data = None
        self.labels = None
        self.seqlen = None

        unknown_token = "UNKNOWN_TOKEN"
        sentence_start_token = "SENTENCE_START"
        sentence_end_token = "SENTENCE_END"

        print "Reading CSV file..."
        with open('reddit_comments_15000.csv', 'rb') as f:
            reader = csv.reader(f, skipinitialspace=True)
            reader.next()
            sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
            sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
        print "Parsed %d sentences." % (len(sentences))
        tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
        word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
        print "Found %d unique words tokens." % len(word_freq.items())
        vocab = word_freq.most_common(vocabulary_size-1)
        index_to_word = [x[0] for x in vocab]
        index_to_word.append(unknown_token)
        word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])
        self.save_vocab(vocab, word_to_index)
        print "Using vocabulary size %d." % vocabulary_size
        print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])

        # Replace all words not in our vocabulary with the unknown token
        for i, sent in enumerate(tokenized_sentences):
            tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

        self.data = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
        self.labels = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
        self.seqlen = np.asarray([len(sent) for sent[1:] in tokenized_sentences])

    def save_vocab(self, vocab, word_to_index):
        with open(os.path.join('test_save', 'words_vocab.pkl'), 'wb') as f:
            cPickle.dump((vocab, word_to_index), f)

    def get_raw_data(self):
        return self.data, self.labels, self.seqlen