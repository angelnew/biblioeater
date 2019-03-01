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

# Now we have to prepare the data for training
pym_set = pym.get_base_training_set()
pym_validation = Book.one_writer_set(pym_set, sentences_per_sample=3)

tom_set = tom.get_base_training_set()
tom_validation = Book.one_writer_set(tom_set, sentences_per_sample=3)

eureka_set = eureka.get_base_training_set()
eureka_validation = Book.one_writer_set(eureka_set, sentences_per_sample=3)

huck_set = huck.get_base_training_set()
huck_validation = Book.one_writer_set(huck_set, sentences_per_sample=3)

# read the model
with open(MODEL_FILE, "rb") as infile:
    sequential_model = pickle.load(infile)

# for simplicity we ignore parapgraphs that are longer than the longest one found in training
max_expected_length = sequential_model.input_shape[1]

# apply the model to pym
pym_validation = pad([u for u in pym_validation if u.shape[0] <= max_expected_length], max_expected_length)
pym_validation = np.asarray(pym_validation)
pym_predictions = sequential_model.predict(pym_validation)
pym_accuracy = sum([probs[0] > 0.5 for probs in pym_predictions]) / len(pym_predictions)
nlp_logger.warning("Accuracy for Poe/pym: {:.4f}".format(pym_accuracy))

# apply the model to tom
tom_validation = pad([u for u in tom_validation if u.shape[0] <= max_expected_length], max_expected_length)
tom_validation = np.asarray(tom_validation)
tom_predictions = sequential_model.predict(tom_validation)
tom_accuracy = sum([probs[1] > 0.5 for probs in tom_predictions]) / len(tom_predictions)
nlp_logger.warning("Accuracy for Twain/tom: {:.4f}".format(tom_accuracy))

# apply the model to Eureka
eureka_validation = pad([u for u in eureka_validation if u.shape[0] <= max_expected_length], max_expected_length)
eureka_validation = np.asarray(eureka_validation)
eureka_predictions = sequential_model.predict(eureka_validation)
eureka_accuracy = sum([probs[0] > 0.5 for probs in eureka_predictions]) / len(eureka_predictions)
nlp_logger.warning("Accuracy for Poe/eureka: {:.4f}".format(eureka_accuracy))

# apply the model to Huck
huck_validation = pad([u for u in huck_validation if u.shape[0] <= max_expected_length], max_expected_length)
huck_validation = np.asarray(huck_validation)
huck_predictions = sequential_model.predict(huck_validation)
huck_accuracy = sum([probs[1] > 0.5 for probs in huck_predictions]) / len(huck_predictions)
nlp_logger.warning("Accuracy for Twain/huck: {:.4f}".format(huck_accuracy))




