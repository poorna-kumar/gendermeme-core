# gendermeme-core
A first open-sourced version of GenderMeme: https://gendermeme.org/

This repo contains the code required to run GenderMeme. 

## Pre-requisites:

- CoreNLP version 3.8.0. 
Get it [here](https://stanfordnlp.github.io/CoreNLP/); download and extract.
- Python libraries (install with `pip install`)
  - `pycorenlp`
  - `numpy`

## Rough guide for understanding this repo:

This repo contains two folders:
- `nlp`: This folder contains a file called `utils.py`. `utils.py` calls `annotate_corenlp`, a function which takes the text of an article as an input, and calls `pycorenlp`, which is a Python wrapper around CoreNLP. The main NLP tasks are performed here (coreference resolution, named entity recognition, dependency parsing, quote detection, etc.) by CoreNLP, and the CoreNLP annotations are returned.
- `analysis`: This folder contains the following files:
  - `utils.py`: Contains functions that parse the output of `annotate_corenlp` to:
    - Identify mentions of people in our article 
    - Figure out which mentions refer to the same person, and hence get a list of individuals mentioned in our article (entity resolution)
    - Guess the gender of each individual
    - Attribute quotes and associated verbs to people to figure out who says something in the article.
  - `gender.py` and `gender_babynames.py`: Two files of first names and their most likely gender (we use first names to infer gender in some cases). `gender.py` has been taken from [this useful public repo](https://github.com/Bemmu/gender-from-name), and `gender_babynames.py` has been derived by us from R’s library `babynames`, based on Census data in the US over the years. 
  - `analysis.py`: Contains a utility wrapper function called `get_article_info`, which allows the user to pass a piece of text and run the whole GenderMeme pipeline on it, and get a JSON output.

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
  The overall output is a list of such objects
 
 ### Example Usage
 
 Run the following from the `analysis` directory:
 ```
 >>> from analysis import get_article_info
 >>> text = 'Ann Smith and her husband Jim went to the movies. "It was okay," he said.'
 >>> get_article_info(text)
 ```
 
 It will return a JSON object created from a Python dictionary which, on prettifying, looks like:
 ```
[{'associated_verbs': [u'go'],
     'gender': 'FEMALE',
     'gender_method': 'COREF',
     'is_speaker': (False, {'Reasons': []}),
     'mentions': [{'end': 2, 'sent_num': 1, 'start': 1}],
     'name': u'Ann Smith',
     'num_times_mentioned': 1,
     'quotes': []},
 {'associated_verbs': [u'go', u'say'],
     'gender': 'MALE',
     'gender_method': 'COREF',
     'is_speaker': (True,
                    {'Reasons': ['Quoted saying 4 words', 'Subject of say']}),
     'mentions': [{'end': 6, 'sent_num': 1, 'start': 6}],
     'name': u'Jim Smith',
     'num_times_mentioned': 1,
     'quotes': [{u'after': u' ',
                 u'before': u'',
                 u'characterOffsetBegin': 51,
                 u'characterOffsetEnd': 53,
                 u'index': 2,
                 u'lemma': u'it',
                 u'ner': u'O',
                 u'originalText': u'It',
                 u'pos': u'PRP',
                 u'speaker': u'7',
                 u'word': u'It'},
                {u'after': u' ',
                 u'before': u' ',
                 u'characterOffsetBegin': 54,
                 u'characterOffsetEnd': 57,
                 u'index': 3,
                 u'lemma': u'be',
                 u'ner': u'O',
                 u'originalText': u'was',
                 u'pos': u'VBD',
                 u'speaker': u'7',
                 u'word': u'was'},
                {u'after': u'',
                 u'before': u' ',
                 u'characterOffsetBegin': 58,
                 u'characterOffsetEnd': 62,
                 u'index': 4,
                 u'lemma': u'okay',
                 u'ner': u'O',
                 u'originalText': u'okay',
                 u'pos': u'JJ',
                 u'speaker': u'7',
                 u'word': u'okay'},
                {u'after': u'',
                 u'before': u'',
                 u'characterOffsetBegin': 62,
                 u'characterOffsetEnd': 63,
                 u'index': 5,
                 u'lemma': u',',
                 u'ner': u'O',
                 u'originalText': u',',
                 u'pos': u',',
                 u'speaker': u'7',
                 u'word': u','}]}]

 ```
