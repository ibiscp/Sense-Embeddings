import xml.etree.cElementTree as etree
import pickle
import glob
import re

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

def create_dictionary(dic_name='coverage'):
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
            # if int(id) in [2031, 3171, 3937, 4449]:
            #     ibis = 1

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

    save(dictionary, '../resources/', dic_name)
    return dictionary

def load_data(path="../resources/", dictionary_name='dictionary2', mapping_name='mapping'):

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

    for key, value in dictionary.items():
        if value['text'] is None:
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

    # Print sample sentences
    print('\nSample sentences')
    for i in sentences[0:10]:
        print(i)

    return sentences
