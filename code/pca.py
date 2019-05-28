from argparse import ArgumentParser
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("resource_folder", nargs='?', default='../resources/', help="Resource folder path")
    parser.add_argument("filtered_vec_name", nargs='?', default='embeddings.vec', help="Name of the filtered embedding file")
    parser.add_argument("top_number", nargs='?', default=40, help="Number of top sampling words")

    return parser.parse_args()

def plot_pca(resource_folder, filtered_vec_name, topnumber):

    # Load model
    model = KeyedVectors.load_word2vec_format(resource_folder + filtered_vec_name, binary=False)
    vocab = model.wv.vocab

    # Get top n
    top = {k: vocab[k] for k in list(vocab)[:topnumber]}

    # Plot PCA
    X = model[top]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    plt.scatter(result[:, 0], result[:, 1])

    # Add top annotations
    words = list(top)
    for i, word in enumerate(words):
        word = word.split('_')[0]
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))

    # Plot
    plt.show()

if __name__ == '__main__':
    args = parse_args()

    # Filter vec file
    plot_pca(resource_folder=args.resource_folder, filtered_vec_name=args.filtered_vec_name, topnumber=args.top_number)