import os
import pickle

import numpy as np
from keras.layers import Conv1D
from keras.layers import Dense, GlobalMaxPooling1D, MaxPooling1D, Dropout, Flatten, Input, concatenate
from keras.models import Sequential, Model, load_model
from keras.utils import plot_model

from random import randint

from constants import *
from the_logger import nlp_logger


class BiblioEater:

    DROPOUT_PROB = 0.2
    DROPOUT_PROB_OUT = 0.3
    NUM_WRITERS = 2
    NUM_FILTERS_1 = 8
    NUM_FILTERS_2 = 16
    NUM_FILTERS_3 = 32
    HIDDEN_DIMS = 16
    RECEPTIVE_FIELD = 4
    STRIDES = 1
    KERNEL_SIZE_2 = 3
    KERNEL_SIZE_3 = 2
    POOL_SIZE = 2
    POOL_SIZE_2 = 2
    POOL_SIZE_3 = 2
    NUM_EPOCHS = 12
    BATCH_SIZE = 8

    def __init__(self):
        self.model = None
        self.max_tokens_per_paragraph = 0
        self.pos_vector_length = 0
    
    def design_sequential_net(self, max_tokens_per_paragraph, pos_vector_length):

        self.max_tokens_per_paragraph = max_tokens_per_paragraph
        self.pos_vector_length = pos_vector_length

        input_shape = (max_tokens_per_paragraph, pos_vector_length)
        
        self.model = Sequential()

        # Block 1
        self.model.add(Conv1D(filters=self.NUM_FILTERS_1, kernel_size=self.RECEPTIVE_FIELD, strides=self.STRIDES,
                            input_shape=input_shape, activation='relu'))
        self.model.add(Dropout(self.DROPOUT_PROB))
        self.model.add(MaxPooling1D(pool_size=self.POOL_SIZE))
        
        # Block 2
        self.model.add(Conv1D(filters=self.NUM_FILTERS_2, kernel_size=self.KERNEL_SIZE_2, activation='relu'))
        self.model.add(MaxPooling1D(pool_size=self.POOL_SIZE_2))

        # Block 3
        self.model.add(Conv1D(filters=self.NUM_FILTERS_3, kernel_size=self.KERNEL_SIZE_3, activation='relu'))
        self.model.add(MaxPooling1D(pool_size=self.POOL_SIZE_3))
        
        # Final block
        self.model.add(Flatten())
        self.model.add(Dropout(self.DROPOUT_PROB_OUT))
        self.model.add(Dense(self.HIDDEN_DIMS, activation="relu"))
        self.model.add(Dense(self.NUM_WRITERS, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

        print(self.model.summary())
        # plot_model(self.model, to_file=os.path.join(OUT_FOLDER, "model.png"))

    def train_sequential_net(self, pos_training_set, writer_labels):

        # We train with the whole set. We will validate with other books
        model_steps = round(len(pos_training_set) / self.BATCH_SIZE)

        self.model.fit_generator(
            self.generate_training_batch(pos_training_set, writer_labels, self.BATCH_SIZE),
            nb_epoch=self.NUM_EPOCHS,
            steps_per_epoch=model_steps, verbose=2
        )

        # serialize
        with open(MODEL_FILE, "wb") as outfile:
            pickle.dump(self.model, outfile)

        nlp_logger.info("Sequential model written to file")

    def design_multi_sentence_net(self, max_tokens_per_sentence, pos_vector_length):
        # The Functional API is required for non sequential networks
        sentence_input_1 = Input(shape=(max_tokens_per_sentence, pos_vector_length,), name='sentence_input_1')
        sentence_input_2 = Input(shape=(max_tokens_per_sentence, pos_vector_length,), name='sentence_input_2')
        sentence_input_3 = Input(shape=(max_tokens_per_sentence, pos_vector_length,), name='sentence_input_3')

        shared_conv = Conv1D(filters=self.NUM_FILTERS_1, kernel_size=self.RECEPTIVE_FIELD, strides=2, activation='relu')
        shared_max_pooling = MaxPooling1D(pool_size=self.POOL_SIZE)
        x1 = shared_conv(sentence_input_1)
        x1 = shared_max_pooling(x1)

        x2 = shared_conv(sentence_input_2)
        x2 = shared_max_pooling(x2)

        x3 = shared_conv(sentence_input_3)
        x3 = shared_max_pooling(x3)

        # Now we concatenate the 3 outputs as input to the next layer
        x = concatenate([shared_max_pooling.get_output_at(0), shared_max_pooling.get_output_at(1),
                         shared_max_pooling.get_output_at(2)], axis=-1)

        # Block 2
        x = Conv1D(filters=self.NUM_FILTERS_3, kernel_size=self.KERNEL_SIZE_2, activation='relu')(x)
        x = Dropout(self.DROPOUT_PROB)(x)
        x = MaxPooling1D(pool_size=self.POOL_SIZE_2)(x)

        # Final block
        x = Flatten()(x)
        x = Dense(self.HIDDEN_DIMS, activation="relu")(x)
        main_output = Dense(1, activation='sigmoid', name='main_output')(x)

        self.model = Model(inputs=[sentence_input_1, sentence_input_2, sentence_input_3], outputs=[main_output])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        print(self.model.summary())

    def train_multi_sentence_net(self, pos_training_set, writer_labels):

        self.model.fit(pos_training_set,
                       writer_labels,
                       batch_size=self.BATCH_SIZE,
                       epochs=self.NUM_EPOCHS, verbose=2)

        # serialize
        with open(MULTI_MODEL_FILE, "wb") as outfile:
            pickle.dump(self.model, outfile)

    def generate_training_batch(self, training_set, labels, batch_size):

        # generator function avoid memory problems with big training sets

        batch_features = np.zeros((batch_size, self.max_tokens_per_paragraph, self.pos_vector_length))
        batch_labels = np.zeros((batch_size, self.NUM_WRITERS))

        while True:
            for i in range(batch_size):
                # choose random index in features
                index = randint(0, len(training_set)-1)

                batch_features[i] = training_set[index]
                if labels[index] == 1:
                    batch_labels[i] = [0, 1]
                else:
                    batch_labels[i] = [1, 0]

            yield batch_features, batch_labels


if __name__ == "__main__":
    biblio_eater = BiblioEater()
    biblio_eater.design_multi_sentence_net(290, 17)
