import xml.etree.cElementTree as etree
import pickle
import glob
import re
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("resource_folder", nargs='?', default='../resources/', help="Resource folder path")
    parser.add_argument("dictionary_name", nargs='?', default='coverage', help="Name of the dictionary file to use")
    parser.add_argument("mapping_name", nargs='?', default='mapping', help="Name of the mapping file to use")

    return parser.parse_args()

# Save dictionary to file
def save(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)

# Load dictionary from file
def load(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def create_mapping(filepath="../resources/bn2wn_mapping.txt"):
    mapping = {}

    with open(filepath) as f:
        lines = f.readlines()

        for l in lines:
            map = l.split()

            mapping[map[0]] = map[1]

    save(mapping, '../resources/mapping')
    return mapping

def create_dictionary(dic_name='precision'):
    dictionary = {}

    # get an iterable
    file = '../resources/dataset/eurosense.v1.0.high-' + dic_name + '.xml'
    context = etree.iterparse(file, events=['start', 'end'])

    # turn it into an iterator
    context = iter(context)

    # get the root element
    event, root = context.__next__()

    for event, elem in context:
        if elem.tag == "sentence" and event == 'start':
            id = elem.attrib["id"]

            if int(id) % 1000 == 0 and int(id) != 0:
                print(id)

            dictionary[id] = {}

        elif elem.tag == "text" and elem.attrib["lang"] == "en" and event == 'end':
            text = elem.text
            dictionary[id]["text"] = text
            dictionary[id]["annotations"] = {}
        elif elem.tag == "annotation" and elem.attrib["lang"] == "en" and event == 'end':
            annotation = {}
            annotation["anchor"] = elem.attrib["anchor"]
            annotation["lemma"] = elem.attrib["lemma"].lower()
            annotation["coherenceScore"] = elem.attrib["coherenceScore"]
            annotation["babelnet"] = elem.text
            dictionary[id]["annotations"][annotation["anchor"]] = annotation

        root.clear()

    save(dictionary, '../resources/' + dic_name)
    return dictionary

def load_data(dictionary_name, path="../resources/", mapping_name='mapping'):

    # Check if sentences exists
    if glob.glob(path + dictionary_name + '_sentences.pkl'):
        print('\nSentences found!')
        sentences = load(path + dictionary_name + '_sentences.pkl')
    else:
        # Check if dictionary exists
        if glob.glob(path + dictionary_name + '.pkl'):
            print('\nDictionary found!')
            dictionary = load(path + dictionary_name)
        else:
            print('\nDictionary not found!')
            print('\nBuilding dataset from file')
            dictionary = create_dictionary()

        # Check if mapping exists
        if glob.glob(path + mapping_name + '.pkl'):
            print('\nMapping found!')
            mapping = load(path + mapping_name)
        else:
            print('\nMapping not found!')
            print('\nBuilding mapping from file')
            mapping = create_mapping()

        sentences = []

        none = 0
        for key, value in dictionary.items():
            if value['text'] is None:
                none += 1
                continue

            text = re.sub("[^a-zA-Z]+", " ", value['text'])

            annotations = value['annotations']

            sentence = []

            for word in text.split():
                if word in annotations and annotations[word]['babelnet'] in mapping:
                    sentence.append(annotations[word]['lemma'] + '_' + annotations[word]['babelnet'])
                else:
                    sentence.append(word.lower())

            sentences.append(sentence)

        print('\nSentences not found: ' + str(none) + " ({:.2%})".format(none/len(dictionary)))

        # Save sentences
        save(sentences, path + dictionary_name + '_sentences')

    # Print sample sentences
    print('\nSample sentences')
    for i in sentences[0:10]:
        print(i)

    return sentences

if __name__ == '__main__':
    args = parse_args()

    _ = load_data(dictionary_name=args.dictionary_name, path=args.resource_folder, mapping_name=args.mapping_name)