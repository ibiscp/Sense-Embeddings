import xml.etree.cElementTree as etree
#import cElementTree as ElementTree
import pickle

# Save dictionary to file
def save(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)

# Load dictionary from file
def load(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def load_data(file="../dataset/eurosense.v1.0.high-coverage.xml"):

    dictionary = {}
    dictionary_number = 0
    save_every = 100000

    # get an iterable
    context = etree.iterparse(file, events=['start'])

    # turn it into an iterator
    context = iter(context)

    # get the root element
    event, root = context.__next__()

    for event, elem in context:
        if event == 'start':
            if elem.tag == "sentence":
                id = elem.attrib["id"]

                # if int(id) % save_every == save_every-1:
                #     save(dictionary, '../dataset/dictionary_' + str(dictionary_number))
                #     dictionary = {}
                #     dictionary_number += 1
                dictionary[id] = {}
                if int(id) % 1000 == 0:
                    print(id)

            if elem.tag == "text" and elem.attrib["lang"] == "en":
                text = elem.text
                dictionary[id]["text"] = text
                dictionary[id]["annotations"] = {}
            elif elem.tag == "annotation" and elem.attrib["lang"] == "en":
                annotation = {}
                #annotation["type"] = elem.attrib["type"]
                annotation["anchor"] = elem.attrib["anchor"]
                annotation["lemma"] = elem.attrib["lemma"]
                annotation["coherenceScore"] = elem.attrib["coherenceScore"]
                #annotation["nasariScore"] = elem.attrib["nasariScore"]
                annotation["babelnet"] = elem.text
                dictionary[id]["annotations"][annotation["lemma"]] = annotation

            root.clear()

    save(dictionary, '../dataset/dictionary')

load_data()