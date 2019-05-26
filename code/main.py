# from time import time
from preprocess import load_data
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import Word2Vec
import multiprocessing
from argparse import ArgumentParser
# import logging
from tqdm import tqdm

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("resources_path", nargs='?', default='../resources/', help="The path of the resources needed to load your model")
    parser.add_argument("sentence_size", nargs='?', const=626, type=int, default=626, help="The size of the maximum sentence")

    return parser.parse_args()

class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''

    def __init__(self, total_epochs):
        self.epoch = 0
        # self.time = None
        self.pbar = tqdm(total=total_epochs)

    # def on_epoch_begin(self, model):
    #     self.time = time()

    def on_epoch_end(self, model):
        #print("Epoch {} - {}".format(self.epoch, round((time() - self.time) / 60, 2)))
        self.epoch += 1
        self.pbar.update(1)

if __name__ == '__main__':
    args = parse_args()

    # Logging
    # logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)

    # Number of cores
    cores = multiprocessing.cpu_count()

    # Epoch logger
    epoch_logger = EpochLogger(5)

    # Load sentences
    sentences = load_data()

    # Model
    model = Word2Vec(sentences,
                     min_count=5,
                     window=5,
                     size=100,
                     sample=6e-5,
                     alpha=0.03,
                     min_alpha=0.0007,
                     negative=20,
                     workers=cores - 1,
                     iter=5,
                     callbacks=[epoch_logger])

    # # Build vocabulary
    # t = time()
    # model.build_vocab(sentences, progress_per=100)
    # print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

    # Train
    # model.train(sentences, total_examples=model.corpus_count, epochs=100, report_delay=1)
    # model.train()


    # # Define the grid search parameters
    # epochs = [5, 10, 20]
    # negative = [0, 5, 10]
    # window = [3, 5]
    # embedding_size = [100, 200, 300]
    # param_grid = dict(batchSize=batchSize, epochs=epochs, embedding_size=embedding_size)
    #
    # # Train
    # grid = gridSearch(build_fn=model, param_grid=param_grid, vocab_size=vocabulary_size, sentence_size=sentenceSize)
    # grid.fit(train_x, train_y, dev_x, dev_y)

    # Print grid search summary
    # grid.summary()