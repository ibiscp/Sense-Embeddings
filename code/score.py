from argparse import ArgumentParser
from gensim.models import KeyedVectors
from scipy.stats import spearmanr
import glob
from preprocess import load, save

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("resource_folder", nargs='?', default='../resources/', help="Resource folder path")
    parser.add_argument("gold_file", nargs='?', default='combined.tab', help="Name of the gold file to use")
    parser.add_argument("model_name", nargs='?', default='embeddings.vec', help="Name of the embedding file to use")

    return parser.parse_args()


def score(resource_folder='../resources/', gold_file='combined.tab', model_name='embeddings.vec', model=None,  debug=False):

    # Read gold file to a list
    word_pairs = []
    gold = []
    with open(resource_folder + gold_file) as f:
        next(f)
        for line in f:
            fields = line.split('\t')
            word_pairs.append([fields[0].lower(), fields[1].lower()])
            gold.append(float(fields[2]))

    # Load the model and get vocabulary
    if model is None:
        model = KeyedVectors.load_word2vec_format(resource_folder + model_name, binary=False)

    # Check if dictionary exists
    if glob.glob(resource_folder + 'gold' + '.pkl'):
        gold_dic = load(resource_folder + 'gold')
    else:

        # Get vocabulary from model
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
        # save(gold_dic, resource_folder + 'gold')

    # Calculate cossine similarity
    none = 0
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
            none += 1

        # Get similarities
        sim = [-1]
        for i in word1_list:
            for j in word2_list:
                sim.append(model.wv.similarity(i, j))

        # Maximum similarity
        cossine.append(max(sim))

    if debug:
        print('\nPairs not found: ' + str(none) + " ({:.2%})".format(none / len(word_pairs)))

    # Spearman correlation
    corr, _ = spearmanr(gold, cossine)

    return corr


if __name__ == '__main__':
    args = parse_args()

    # Calculate correlation
    corr = score(resource_folder=args.resource_folder, gold_file=args.gold_file, model_name=args.model_name, debug=True)

    print("Final Score:\t", corr)
