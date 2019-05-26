from argparse import ArgumentParser
from gensim.models import KeyedVectors
from scipy.stats import spearmanr
import glob
from preprocess import load, save

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("prediction_file", nargs='?',
                        default='../resources/dataset/predicted/cityu_test_gold_prediction.txt',
                        help="The path to the prediction file (in BIES format)")
    parser.add_argument("gold_file", nargs='?', default='../resources/dataset/predicted/cityu_test_gold.txt',
                        help="The path to the gold file (in BIES format)")

    return parser.parse_args()


def score(gold_path='../resources/', gold_file='combined.tab', model=None):

    # Read gold file to a list
    word_pairs = []
    gold = []
    with open(gold_path + gold_file) as f:
        next(f)
        for line in f:
            fields = line.split('\t')
            word_pairs.append([fields[0].lower(), fields[1].lower()])
            gold.append(float(fields[2]))

    # Check if dictionary exists
    if glob.glob(gold_path + 'gold' + '.pkl'):
        gold_dic = load(gold_path + 'gold')
    else:
        # Load the model and get vocabulary
        if model is None:
            model = KeyedVectors.load_word2vec_format(gold_path + 'embeddings.vec', binary=False)
        vocab = model.wv.vocab

        # Distinct words in gold
        distinct = []
        for tuple in word_pairs:
            for word in tuple:
                if word not in distinct:
                    distinct.append(word)

        # Search gold in model
        gold_dic = {}
        for k in vocab.keys():
            k_ = k.split('_')[0]
            if k_ in distinct:
                try:
                    gold_dic[k_].append(k)
                except:
                    gold_dic[k_] = [k]

        # Save dictionary
        save(gold_dic, gold_path + 'gold')

    # Calculate cossine similarity
    cossine = []
    for tuple in word_pairs:
        word1 = tuple[0]
        word2 = tuple[1]

        # Get all related words in gold dictionary
        try:
            word1_list = gold_dic[word1]
            word2_list = gold_dic[word2]
        except:
            word1_list = []
            word2_list = []

        # Get similarities
        sim = [-1]
        for i in word1_list:
            for j in word2_list:
                sim.append(model.wv.similarity(i, j))

        # Maximum similarity
        cossine.append(max(sim))

    # Spearman correlation
    corr, _ = spearmanr(gold, cossine)

    return corr


if __name__ == '__main__':
    args = parse_args()

    # Calculate correlation
    corr = score(gold_path='../resources/', gold_file='combined.tab')

    print("Final Score:\t", corr)

