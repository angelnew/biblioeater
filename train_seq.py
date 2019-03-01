import pickle
import random

from biblio_eater import BiblioEater
from constants import *
from the_logger import nlp_logger
from padder import pad


# Load training set from disk first
with open(TRAINING_SET_FILE, "rb") as infile:
    training_set = pickle.load(infile)
# pad with zero rows up to max sentence length
max_length = max([s.shape[0] for s in training_set])
training_set = pad(training_set, max_length)

with open(LABELS_FILE, "rb") as infile:
    writer_labels = pickle.load(infile)

# Sequential network

nlp_logger.warning("Shape of training set ({}, {})".format(training_set[0].shape[0], training_set[0].shape[1]))

# Prepare net
biblio_eater = BiblioEater()

# all sentences are padded to same length, althoug Keras has a padding option that we are not using
biblio_eater.design_sequential_net(training_set[0].shape[0], training_set[0].shape[1])
biblio_eater.train_sequential_net(training_set, writer_labels)


