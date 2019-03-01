import os
import random
import pickle
import numpy as np

from constants import *
from the_logger import nlp_logger

from book import Book

pym = Book("Arthur Gordon Pym")
tom = Book("Tom Sawyer")

# read the parsed texts
pym.from_file(PYM_FILE)
tom.from_file(TOM_FILE)

# replace word with POS in sentences
pym.encode_as_pos()
tom.encode_as_pos()

# Now we have to prepare the data for training
pym_base_set = pym.get_base_training_set()
tom_base_set = tom.get_base_training_set()

# Prepare training set for sequential network
(sequential_set, writer_labels) = Book.get_seq_training_set(pym_base_set, tom_base_set, num_sentences=3)

nlp_logger.warning("writing {} sentences to training set file".format(len(sequential_set)))

with open(TRAINING_SET_FILE, "wb") as outfile:
    pickle.dump(sequential_set, outfile)

with open(LABELS_FILE, "wb") as outfile:
    pickle.dump(writer_labels, outfile)


# Prepare training set for multi input network
(multi_set, multi_writer_labels) = Book.get_multi_training_set(pym_base_set, tom_base_set, num_sentences=3)

nlp_logger.warning("writing {} sentences to multi sentence training set file".format(multi_set[0].shape[0]))

with open(TRAINING_SET_FILE_4_MULTI, "wb") as outfile:
    pickle.dump(multi_set, outfile)

with open(LABELS_FILE_4_MULTI, "wb") as outfile:
    pickle.dump(multi_writer_labels, outfile)






