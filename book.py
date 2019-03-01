import pickle
import torch
from collections import Counter
from random import sample

from the_logger import nlp_logger
from stats import *
from constants import *
from padder import *

from stanfordnlp import Document


class Book:

    def __init__(self, name):
        self.name = name

        self.corpus = None
        self.doc = None
        self.words_per_sentence = None
        self.freqs = None
        self.upos_freqs = None
        self.pos_sentences = []
        self.training_indexes = []

    def load_corpus(self, corpus_filename):
        with open(corpus_filename, "r", encoding="utf-8") as f:
            self.corpus = f.read()
        nlp_logger.warning("{} corpus loaded".format(self.name))

    def parse(self, pipeline, crop_value):
        self.doc = pipeline(self.crop(crop_value))
        nlp_logger.warning("{} doc processed with {} sentences".format(self.name, len(self.doc.sentences)))
        torch.cuda.empty_cache()
        self.compute_describe_freqs()

    def to_file(self, filename):
        # serializes book for further processing
        with open(filename, 'wb') as outfile:
            pickle.dump(self.__dict__, outfile)

        nlp_logger.warning("{} doc serialized to disk".format(self.name))

    def crop(self, crop_value, separator="."):
        # takes the first crop_value sentences of corpus. Useful for testing
        # crop = -1 returns the whole corpus
        if crop_value < 0:
            return self.corpus

        start = self.corpus.find(separator)
        while start >= 0 and crop_value > 1:
            start = self.corpus.find(separator, start + len("."))
            crop_value -= 1

        return self.corpus[:start]

    def from_file(self, filename):
        # loads the serialized book from file
        with open(filename, 'rb') as infile:
            self.__dict__ = pickle.load(infile)

        nlp_logger.warning("{} doc loaded from disk".format(self.name))
        self.describe()

    def describe(self):

        self.freqs = Counter(self.words_per_sentence)

        nlp_logger.warning("Lexical diversity is {:,.3}".format(self.lexical_diversity()))

        describe(self.words_per_sentence, "{} - # of words".format(self.name))

    def compute_describe_freqs(self):
        self.words_per_sentence = [len(s.words) for s in self.doc.sentences]
        self.upos_freqs = Counter([w.upos for s in self.doc.sentences for w in s.words])
        self.feat_count = Counter([w.feats for s in self.doc.sentences for w in s.words])

    def lexical_diversity(self):
        return len(set(self.corpus)) / len(self.corpus)

    def encode_as_pos(self):
        self.pos_sentences = []
        for sentence in self.doc.sentences:
            self.pos_sentences.append([w.upos for w in sentence.words])

    def readable_upos_freqs(self):
        # Replaces upos key (such as ADJ) with full descriptive (Adjective)
        return {upos_dict[k]: v for k, v in self.upos_freqs.items()}

    @staticmethod
    def pos_to_array(pos):
        # a part of speech such as ADV is converted into [0,0,1,0,0,0,0,0,0,0,0,0]

        pos_as_int = list(upos_dict.keys()).index(pos)
        a = [0]*len(upos_dict)
        a[pos_as_int] = 1
        return a

    @staticmethod
    def sentence_to_np_array(sentence):
        # here sentence is a list of POS
        np_array = np.zeros((len(sentence), len(upos_dict)))

        for i in range(len(sentence)):
            np_array[i] = Book.pos_to_array(sentence[i])

        return np_array

    def get_base_training_set(self):
        # training_set is made of several sentences_per_sample and used in sequential net
        # training_set_4_multi is just one sentence per sample and will be processed in multi_sentence_net

        base_training_set = []

        for i in range(len(self.pos_sentences)):
            base_training_set.append(Book.sentence_to_np_array(self.pos_sentences[i]))

        # for i in self.training_indexes:
        #    new_paragraph = Book.sentence_to_np_array(self.pos_sentences[i])
        #    new_text = Book.sentence_to_text(self.doc.sentences[i])
        #    for j in range(1, sentences_per_sample):
        #        new_sentence = Book.sentence_to_np_array(self.pos_sentences[i + j])
        #        new_paragraph = np.vstack((new_paragraph, new_sentence))
        #        new_text += Book.sentence_to_text(self.doc.sentences[i+j])

        #    training_set.append(new_paragraph)
        #    paragraphs.append(new_text)

        return base_training_set

    @staticmethod
    def get_seq_training_set(writer0_set, writer1_set, num_sentences=3):
        seq_set_0 = Book.one_writer_set(writer0_set, num_sentences)
        seq_set_1 = Book.one_writer_set(writer1_set, num_sentences)

        # Undersample to keep both samples at equal length
        if len(seq_set_0) > len(seq_set_1):
            seq_set_0 = random.sample(seq_set_0, len(seq_set_1))
        else:
            seq_set_1 = random.sample(seq_set_1, len(seq_set_0))

        samples_per_writer = len(seq_set_0)

        # merge the 2 sets
        seq_set = seq_set_0
        seq_set.extend(seq_set_1)

        # Label the samples
        labels = [0]*samples_per_writer
        labels.extend([1]*samples_per_writer)

        # And shuffle them
        shuffled = list(zip(seq_set, labels))
        random.shuffle(shuffled)
        seq_set, labels = zip(*shuffled)

        return seq_set, labels

    @staticmethod
    def one_writer_set(writer_set, sentences_per_sample):
        concat_sentence_set = []
        for i in range(len(writer_set) - sentences_per_sample + 1):
            new_paragraph = writer_set[i]
            for j in range(1, sentences_per_sample):
                new_sentence = writer_set[i + j]
                new_paragraph = np.vstack((new_paragraph, new_sentence))

            concat_sentence_set.append(new_paragraph)

        return concat_sentence_set

    @staticmethod
    def get_multi_training_set(writer0_set, writer1_set, num_sentences=3):
        samples_per_writer = min(len(writer0_set), len(writer1_set)) - num_sentences + 1
        max_sentence_length_0 = max([len(s) for s in writer0_set])
        max_sentence_length = max(max_sentence_length_0, max([len(s) for s in writer1_set]))

        multi_set_0 = Book.one_writer_multi_set(writer0_set, num_sentences, samples_per_writer, max_sentence_length)
        multi_set_1 = Book.one_writer_multi_set(writer1_set, num_sentences,samples_per_writer, max_sentence_length)

        # create multiset by vstacking both lists of sentences element-wise
        multi_set = []
        for i in range(num_sentences):
            multi_set.append(np.vstack((multi_set_0[i], multi_set_1[i])))

        # Label the samples
        labels = [0] * samples_per_writer
        labels.extend([1] * samples_per_writer)

        # Shuffle
        shuffled_indices = [i for i in range(len(labels))]
        random.shuffle(shuffled_indices)

        for num_sentence in range(num_sentences):
            multi_set[num_sentence] = multi_set[num_sentence][shuffled_indices, ]

        labels = [labels[i] for i in shuffled_indices]

        return multi_set, labels

    @staticmethod
    def one_writer_multi_set(writer_set, sentences_per_sample, samples_per_writer, max_sentences_length):
        # produces a list of np arrays of length sentences_per_sample
        # corresponding to the sample input for one of the writers
        dim_0 = len(writer_set) - sentences_per_sample + 1
        dim_1 = max_sentences_length
        dim_2 = len(upos_dict)

        multi_set = [np.zeros((dim_0, dim_1, dim_2)) for i in range(sentences_per_sample)]

        for i in range(dim_0):
            for num_sentence in range(sentences_per_sample):
                padded = pad_sentence(writer_set[i + num_sentence], dim_1, dim_2)
                multi_set[num_sentence][i] = padded

        # Undersample to keep both samples at equal length
        if multi_set[0].shape[0] > samples_per_writer and samples_per_writer >0:  # neg samples_per_writer = all
            indices = random.sample(range(multi_set[0].shape[0]), samples_per_writer)
            for num_sentence in range(sentences_per_sample):
                multi_set[num_sentence] = multi_set[num_sentence][indices, ]  # ext

        # drop paragraphs where at least 1 sentence is too long. May happen in validation.
        num_samples = multi_set[0].shape[0]
        kept_indices = []
        for i in range(num_samples):
            nlp_logger.debug("Row {}".format(i))
            keep = True
            for num_sentence in range(sentences_per_sample):
                keep = keep and multi_set[num_sentence][i].shape[0] <= max_sentences_length
            if keep:
                kept_indices.append(i)

        for num_sentence in range(sentences_per_sample):
            multi_set[num_sentence] = multi_set[num_sentence][kept_indices,]  # ext

        return multi_set

    @staticmethod
    def sentence_to_text(sentence):
        return ' '.join([w.text for w in sentence.words])



