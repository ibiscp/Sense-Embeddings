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
        print('\tEpoch', self.epoch, '/', self.total)

class gridSearch:

    def __init__(self, sentences, param_grid):
        self.param_grid = param_grid
        self.best_score = -10
        self.best_loss = 0
        self.best_params = None
        self.results = []
        self.sentences = sentences
        self.cores = multiprocessing.cpu_count()

    def fit(self):

        for g in ParameterGrid(self.param_grid):

            print('\nUsing parameters:', g)
            # callback_str = '_'.join(['%s-%s' % (key, str(value)) for (key, value) in g.items()])
            # cbk = K.callbacks.TensorBoard("../resources/logging/" + callback_str)

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
                             compute_loss=True,
                             callbacks=[epoch_logger])

            loss = model.get_latest_training_loss()

            corr = score(model=model)
            print('\tScore: %f Loss: %f' % (corr, loss))

            self.results.append({'corr':corr, 'loss':loss, 'params':g})

            # Write to results
            with open('../resources/results.txt', "a+") as f:
                f.write("Correlation: %f Loss: %f - Parameters: %r\n" % (corr, loss, g))

            if corr > self.best_score:
                self.best_score = corr
                self.best_params = g
                self.best_loss = loss

                # Save model
                print("\tSaving model")
                #model.wv.save_word2vec_format("../resources/model.bin")
                model.wv.save_word2vec_format("../resources/embeddings.vec", binary=False)

                # Substitute
                with open("../resources/results.txt") as f:
                    lines = f.readlines()

                lines[0] = "Best -> Correlation: " + str(self.best_score) + "Loss: " + str(self.best_loss) + " using " + str(self.best_params)

                with open("../resources/results.txt", "w+") as f:
                    f.writelines(lines)

    def summary(self):
        # Summarize results
        print('\nSummary')
        print("Best -> Correlation: %f Loss: %f using %s" % (self.best_score, self.best_loss, self.best_params))
        for res in self.results:
            print("Correlation: %f Loss: %f - Parameters: %r" % (res['corr'], res['loss'], res['params']))
