import stanfordnlp
import os

from constants import *
from the_logger import nlp_logger

from book import Book

nlp_logger.warning("----")
nlp_logger.warning("Process starts")

pym = Book("Arthur Gordon Pym")
tom = Book("Tom Sawyer")
eureka = Book("Eureka")
huck = Book("Huckleberry Finn")

en_nlp = stanfordnlp.Pipeline()
nlp_logger.warning("Pipeline read")

crop_value = -1

# Read and parse texts used for training
# Parsing takes time, even with GPU!

pym.load_corpus(os.path.join(CORPORA_FOLDER, "pym.txt"))
pym.parse(en_nlp, crop_value)
pym.to_file(PYM_FILE)

tom.load_corpus(os.path.join(CORPORA_FOLDER, "tom.txt"))
tom.parse(en_nlp, crop_value)
tom.to_file(TOM_FILE)

# Read and parse texts used for validation
eureka.load_corpus(os.path.join(CORPORA_FOLDER, "eureka.txt"))
eureka.parse(en_nlp, crop_value)
eureka.to_file(EUREKA_FILE)

huck.load_corpus(os.path.join(CORPORA_FOLDER, "huck.txt"))
huck.parse(en_nlp, crop_value)
huck.to_file(HUCK_FILE)

nlp_logger.warning("Process completed")

