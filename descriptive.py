import os

from constants import *
from the_logger import nlp_logger

from frequency_plotter import FrequencyPlotter
from book import Book

pym = Book("Arthur Gordon Pym")
tom = Book("Tom Sawyer")

pym.from_file(PYM_FILE)
tom.from_file(TOM_FILE)

pym.compute_describe_freqs()
pym.describe()
tom.compute_describe_freqs()
tom.describe()

# Compare sentence length in pym and tom
my_plot = FrequencyPlotter(plot_title="Sentence length in words")
my_plot.continuous_hist(pym.words_per_sentence, tom.words_per_sentence)
my_plot.persist(os.path.join(OUT_FOLDER, LENGTH_CHART_FILE))

# Compare relative frequencies of parts of speech in pym and tom
my_plot = FrequencyPlotter("Frequencies of parts of speech")
my_plot.categorical_hist(pym.readable_upos_freqs(), tom.readable_upos_freqs())
my_plot.persist(os.path.join(OUT_FOLDER, UPOS_CHART_FILE))
