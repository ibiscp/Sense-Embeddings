from preprocess import load_data
from argparse import ArgumentParser
from gridSearch import *

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("dictionary_name", nargs='?', default='precision', help="Name of the dictionary file to use")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # Load sentences
    sentences = load_data(dictionary_name=args.dictionary_name)

    # Define the grid search parameters
    epochs = [30]
    negative = [10]
    window = [5]
    embedding_size = [500]
    min_count = [5]
    param_grid = dict(epochs=epochs, negative=negative, window=window, embedding_size=embedding_size, min_count=min_count)

    # Train
    grid = gridSearch(sentences=sentences, param_grid=param_grid)
    grid.fit()

    # Print grid search summary
    grid.summary()