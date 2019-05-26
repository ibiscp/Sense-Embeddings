# from time import time
from preprocess import load_data
from argparse import ArgumentParser
from gridSearch import *

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("resources_path", nargs='?', default='../resources/', help="The path of the resources needed to load your model")
    parser.add_argument("sentence_size", nargs='?', const=626, type=int, default=626, help="The size of the maximum sentence")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # Load sentences
    sentences = load_data(dictionary_name='dictionary')

    # Define the grid search parameters
    epochs = [5, 10, 20]
    negative = [0, 5, 10]
    window = [3, 5]
    embedding_size = [100, 200, 300]
    min_count = [5, 10]
    param_grid = dict(epochs=epochs, negative=negative, window=window, embedding_size=embedding_size, min_count=min_count)

    # Train
    grid = gridSearch(sentences=sentences, param_grid=param_grid)
    grid.fit()

    # Print grid search summary
    grid.summary()