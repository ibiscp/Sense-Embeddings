from sklearn.model_selection import ParameterGrid
import tensorflow.keras as K
from gensim.models.callbacks import CallbackAny2Vec
from tqdm import tqdm
import random
import numpy as np

class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''

    def __init__(self, total_epochs):
        self.epoch = 0
        self.pbar = tqdm(total=total_epochs)

    def on_epoch_end(self, model):
        self.epoch += 1
        self.pbar.update(1)

class gridSearch:

    def __init__(self, build_fn, param_grid, vocab_size, sentence_size):
        self.param_grid = param_grid
        self.best_score = 0
        self.best_params = None
        self.results = []
        self.vocab_size = vocab_size
        self.sentence_size = sentence_size

    def fit(self, X, y, X_test, y_test):

        for g in ParameterGrid(self.param_grid):
            model = self.build_fn(vocab_size=self.vocab_size, sentence_size=self.sentence_size, mergeMode=g['mergeMode'], lstmLayers=g['lstmLayers'], embedding_size=g['embedding_size'])

            print('\nUsing parameters:', g)
            callback_str = '_'.join(['%s-%s' % (key, str(value)) for (key, value) in g.items()])
            cbk = K.callbacks.TensorBoard("../resources/logging/" + callback_str)
            model.fit_generator(self.generator(X, y, batch_size=g['batchSize']), steps_per_epoch=100, validation_data=(X_test, y_test), epochs=g['epochs'], callbacks=[cbk])

            print('Evaluating')
            loss, acc = model.evaluate(X_test, y_test, verbose=1)
            print('Loss: %f - Accuracy: %f' % (loss, acc))

            self.results.append({'loss':loss, 'acc':acc, 'params':g})

            # Write to results
            with open('../resources/results.txt', "a+") as f:
                f.write("Loss: %f - Accuracy: %f - Parameters: %r\n" % (loss, acc, g))

            if acc > self.best_score:
                self.best_score = acc
                self.best_params = g

                # Save model
                print("Saving model")
                model.save("../resources/model.h5")

                # Substitute
                with open("../resources/results.txt") as f:
                    lines = f.readlines()

                lines[0] = "Best: " + self.best_score + " using " + self.best_params

                with open("../resources/results.txt", "w+") as f:
                    f.writelines(lines)

    def summary(self):
        # Summarize results
        print('\nSummary')
        print("Best: %f using %s" % (self.best_score, self.best_params))
        for res in self.results:
            print("Loss: %f - Accuracy: %f - Parameters: %r" % (res['loss'], res['acc'], res['params']))
