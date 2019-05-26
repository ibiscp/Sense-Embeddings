from argparse import ArgumentParser
from gensim.models import KeyedVectors
from scipy.stats import spearmanr

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("prediction_file", nargs='?',
                        default='../resources/dataset/predicted/cityu_test_gold_prediction.txt',
                        help="The path to the prediction file (in BIES format)")
    parser.add_argument("gold_file", nargs='?', default='../resources/dataset/predicted/cityu_test_gold.txt',
                        help="The path to the gold file (in BIES format)")

    return parser.parse_args()


def score(gold_path='../resources/', gold_file='dataset/combined.tab'):

    # Read gold file to a list
    word_pairs = []
    gold = []
    with open(gold_path + gold_file) as f:
        next(f)
        for line in f:
            fields = line.split('\t')
            word_pairs.append([fields[0].lower(), fields[1].lower()])
            gold.append(float(fields[2]))

    # Load the model and get vocabulary
    wv_from_text = KeyedVectors.load_word2vec_format(gold_path + 'model.txt', binary=False)
    vocab = wv_from_text.wv.vocab

    # Find words in vocab
    cossine = []
    for i in word_pairs:
        word1 = i[0]
        word2 = i[1]

        # Get all related words in vocab
        word1_list = []
        word2_list = []
        for k in vocab.keys():
            k_ = k.split('_')[0]
            if k_ in (word1, word2):
                if k_ == word1:
                    word1_list.append(k)
                else:
                    word2_list.append(k)

        # Get similarities
        sim = [-1]
        for i in word1_list:
            for j in word2_list:
                sim.append(wv_from_text.wv.similarity(i, j))

        # Maximum similarity
        cossine.append(max(sim))

        # Spearman correlation
        corr, _ = spearmanr(gold, cossine)

    return corr


if __name__ == '__main__':
    args = parse_args()

    # Calculate correlation
    corr = score(gold_path='../resources/', gold_file='dataset/combined.tab')

    print("Final Score:\t", corr)

