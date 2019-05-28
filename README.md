# Sense Embeddings

The goal of this project is to train a Continuous Bag of Words (CBOW) model using Gensim Word2Vec to create a sense embedding.

The dataset used for the training was the [EuroSense dataset](http://lcl.uniroma1.it/eurosense/), which is a multilingual sense-annotated resource in 21 languages, however only the English language was used for this task.

For the correlation evaluation, the dataset [WordSimilarity-353](http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/) is used.

The training was done using a Google Compute Engine instance running a Tesla K80 GPU.

<p align="center">
<img src="report/pca.png" width="800"/></br>
<i>Dimensionality reduction of the 40 words of the BabelNet synset with the highest number of samples</i>
</p>

## Instructions

* Generate dictionary

`python preprocess.py [dictionary_name] [path] [mapping_name]`

* Train

`python train.py [dictionary_name]`

* Score

`python train.py [resource_folder] [gold_file] [model_name] [debug]`

* Filter vec file to keep only BabelNet words

`python convert.py [resource_folder] [vec_name] [filtered_vec_name]`

* Plot PCA

`python pca.py [resource_folder] [filtered_vec_name] [topnumber]`