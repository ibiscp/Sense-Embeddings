from sklearn.model_selection import ParameterGrid
from gensim.models.callbacks import CallbackAny2Vec
import multiprocessing
from gensim.models import Word2Vec
from score import score
import tensorflow.keras as K

class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''

    def __init__(self, total_epochs):
        self.epoch = 0
        self.total = total_epochs

    def on_epoch_end(self, model):
        self.epoch += 1
        print('\tEpoch', str(self.epoch) + '/' + str(self.total))

class gridSearch:

    def __init__(self, sentences, param_grid):
        self.param_grid = param_grid
        self.best_score = -10
        self.best_params = None
        self.results = []
        self.iter = 0
        self.sentences = sentences
        self.cores = multiprocessing.cpu_count()

    def fit(self):

        for g in ParameterGrid(self.param_grid):
            self.iter += 1
            print('\nTraining:', str(self.iter) + '/' + str(len(ParameterGrid(self.param_grid))), '- Parameters:', g)

            # Epoch logger
            epoch_logger = EpochLogger(g['epochs'])

            # Model
            model = Word2Vec(self.sentences,
                             min_count=g['min_count'],
                             window=g['window'],
                             size=g['embedding_size'],
                             sample=6e-5,
                             alpha=0.03,
                             min_alpha=0.0007,
                             negative=g['negative'],
                             workers=self.cores,
                             iter=g['epochs'],
                             callbacks=[epoch_logger])

            corr = score(model=model)
            print('\tScore: %f' % (corr))

            self.results.append({'corr':corr, 'params':g})

            # Write to results
            with open('../resources/results.txt', "a+") as f:
                f.write("Correlation: %f - Parameters: %r\n" % (corr, g))

            if corr > self.best_score:
                self.best_score = corr
                self.best_params = g

                # Save model
                print("\tSaving model")
                model.wv.save_word2vec_format("../resources/embeddings_total.vec", binary=False)

                # Substitute
                with open("../resources/results.txt") as f:
                    lines = f.readlines()

                lines[0] = "Best -> Correlation: " + str(self.best_score) + " using " + str(self.best_params) + "\n"

                with open("../resources/results.txt", "w+") as f:
                    f.writelines(lines)

    def summary(self):
        # Summarize results
        print('\nSummary')
        print("Best -> Correlation: %f using %s" % (self.best_score, self.best_params))
        for res in self.results:
            print("Correlation: %f - Parameters: %r" % (res['corr'], res['params']))
