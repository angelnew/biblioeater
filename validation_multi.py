import os

from constants import *
from the_logger import nlp_logger
from padder import pad
from book import Book

import pickle
import numpy as np

# Load the books
pym = Book("Arthur Gordon Pym")
tom = Book("Tom Sawyer")
eureka = Book("Eureka")
huck = Book("Huckleberry Finn")

pym.from_file(PYM_FILE)
tom.from_file(TOM_FILE)
eureka.from_file(EUREKA_FILE)
huck.from_file(HUCK_FILE)

# replace word with POS in sentences
pym.encode_as_pos()
tom.encode_as_pos()
eureka.encode_as_pos()
huck.encode_as_pos()

# read the model
with open(MULTI_MODEL_FILE, "rb") as infile:
    sequential_model = pickle.load(infile)
    
# Now we have to prepare Pym for validation
pym_set = pym.get_base_training_set()
pym_validation = Book.one_writer_multi_set(pym_set, sentences_per_sample=3,
                                                samples_per_writer=-1, max_sentences_length=208)

# apply the model to pym
pym_predictions = sequential_model.predict(pym_validation)
poe_accuracy = sum([probs[0] < 0.5 for probs in pym_predictions]) / len(pym_predictions)

nlp_logger.warning("Accuracy for Poe/pym: {:.4f}".format(poe_accuracy))

# Now we have to prepare Tom for validation
tom_set = tom.get_base_training_set()
tom_validation = Book.one_writer_multi_set(tom_set, sentences_per_sample=3,
                                                samples_per_writer=-1, max_sentences_length=208)

# apply the model to tom
tom_predictions = sequential_model.predict(tom_validation)
tom_accuracy = sum([probs[0] > 0.5 for probs in tom_predictions]) / len(tom_predictions)

nlp_logger.warning("Accuracy for Twain/tom: {:.4f}".format(tom_accuracy))

# Now we have to prepare Erureka for validation
eureka_set = eureka.get_base_training_set()
eureka_validation = Book.one_writer_multi_set(eureka_set, sentences_per_sample=3,
                                                samples_per_writer=-1, max_sentences_length=208)

# apply the model to eureka
eureka_predictions = sequential_model.predict(eureka_validation)
poe_accuracy = sum([probs[0] < 0.5 for probs in eureka_predictions]) / len(eureka_predictions)

nlp_logger.warning("Accuracy for Poe/Eureka: {:.4f}".format(poe_accuracy))

# Now we have to prepare huck for validation
huck_set = huck.get_base_training_set()
huck_validation = Book.one_writer_multi_set(huck_set, sentences_per_sample=3,
                                                samples_per_writer=-1, max_sentences_length=208)

# apply the model to huck
huck_predictions = sequential_model.predict(huck_validation)
huck_accuracy = sum([probs[0] > 0.5 for probs in huck_predictions]) / len(huck_predictions)

nlp_logger.warning("Accuracy for Twain/huck: {:.4f}".format(huck_accuracy))


