import numpy as np

from the_logger import nlp_logger


def pad(data_set, max_length):
    nlp_logger.info("{} is the max length of paragraph".format(max_length))

    padded = []
    num_cols = data_set[0].shape[1]  # all sentences have same number of cols
    for i in range(len(data_set)):
        padded.append(pad_sentence(data_set[i], max_length, num_cols))

    return padded


def pad_sentence(sentence, max_length, num_cols):
    words_in_sentence = sentence.shape[0]
    if words_in_sentence < max_length:
        to_pad = np.zeros((max_length - words_in_sentence, num_cols))
        # pad columns to the right
        padded = np.append(sentence, to_pad, axis=0)
    elif words_in_sentence == max_length:
        padded = sentence
    else:
        padded = sentence[:max_length]

    return padded



