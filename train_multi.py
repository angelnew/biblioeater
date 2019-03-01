import pickle
import random

from biblio_eater import BiblioEater
from constants import *
from the_logger import nlp_logger
from padder import pad


# Load training set from disk first
with open(TRAINING_SET_FILE_4_MULTI, "rb") as infile:
    training_set = pickle.load(infile)

with open(LABELS_FILE_4_MULTI, "rb") as infile:
    writer_labels = pickle.load(infile)

nlp_logger.warning("Shape of training set ({}, {}, {})".format(training_set[0].shape[0], training_set[0].shape[1], training_set[0].shape[2]))

# Prepare net
biblio_eater = BiblioEater()
# all sentences are padded to same length


biblio_eater.design_multi_sentence_net(training_set[0].shape[1], training_set[0].shape[2])
biblio_eater.train_multi_sentence_net(training_set, writer_labels)
