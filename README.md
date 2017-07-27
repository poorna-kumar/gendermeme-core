# gendermeme-core
A first open-sourced version of GenderMeme: https://gendermeme.org/

This repo contains the code required to run GenderMeme. 

## Pre-requisites:

- CoreNLP version 3.8.0. 
Get it [here](https://stanfordnlp.github.io/CoreNLP/); download and extract.
- Python libraries (to be `pip install`’ed)
  - pycorenlp
  - numpy

## Rough guide for understanding this repo:

This repo contains two folders:
- `nlp`: This folder contains a file called `utils.py`. `utils.py` calls `annotate_corenlp`, a function which takes the text of an article as an input, and calls `pycorenlp`, which is a Python wrapper around CoreNLP. The main NLP tasks are performed here (coreference resolution, named entity recognition, dependency parsing, quote detection, etc.) by CoreNLP, and the CoreNLP annotations are returned.
- `analysis`: This folder contains the following files:
  - `utils.py`: Contains functions that parse the output of `annotate_corenlp` to:
    - Identify mentions of people in our article 
    - Figure out which mentions refer to the same person, and hence get a list of individuals mentioned in our article (entity resolution)
    - Guess the gender of each individual
    - Attribute quotes and associated verbs to people to figure out who says something in the article.
  - `gender.py` and `gender_babynames.py`: Two files of first names and their most likely gender (we use first names to infer gender in some cases). gender.py has been taken from this useful public repo, and `gender_babynames.py` has been derived by us from R’s library `babynames`, based on Census data in the US over the years. 
  - analysis.py: Contains a utility wrapper function called `get_article_info`, which allows the user to pass a piece of text and run the whole GenderMeme pipeline on it, and get a JSON output.

## How to run this code:

- Start a CoreNLP server (more details [here](https://stanfordnlp.github.io/CoreNLP/corenlp-server.html)):
  - cd into the directory that you unzipped CoreNLP to, and run:

  `java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 150000`
- import the function `get_article_info` from `analysis.py` and call it as `get_article_info(text_to_analyze)` with the text to be analyzed passed as a string.
- The output is a JSON string, structured as follows. For each individual, we assign a unique id, and produce a JSON object with the following keys: 
```
{
‘name’: full name of person
‘mentions’: a list of positions 
‘num_times_mentioned’: int
‘gender’: string
‘gender_method’: string, one of ‘HONORIFIC’, ‘COREF’ or ‘NAME_ONLY’
‘quotes’: list of CoreNLP tokens (words) that this person said
‘is_speaker’: whether the person speaks in the article, and a list of reasons for why we think so
‘associated_verbs’: list
}
```
 The overall output is a JSON object with each individual's id as a key, and the object above as the value.
 
 ### Example Usage
 
 Run the following from the `analysis` directory:
 ```
 >>> from analysis import get_article_info
 >>> text = 'Ann Smith and her husband Jim went to the movies. "It was okay," he said.'
 >>> get_article_info(text)
 ```
