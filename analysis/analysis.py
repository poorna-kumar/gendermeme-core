import sys
from pprint import pprint
sys.path.append('../')
from nlp import utils as nlp_utils
from utils import get_people_mentioned, get_quotes, get_associated_verbs, \
    identify_sources, get_associated_adjectives, get_people_mentioned_new


def get_article_info(article_text, ann=None, verbose=False):
    """
    Helper function that applies our techniques on a given piece of text,
    first annotating it with CoreNLP then doing other stuff.

    Primarily used by the web app right now.
    """
    if ann is None:
        ann = nlp_utils.annotate_corenlp(
                article_text,
                annotators=['pos', 'lemma', 'ner', 'parse',
                            'depparse', 'dcoref', 'quote',
                            'openie'])

    sentences, corefs = ann['sentences'], ann['corefs']
    if verbose:
        pprint(sentences)
        pprint(corefs)

    id_to_info = get_people_mentioned_new(sentences, corefs)

    people_mentioned = {}
    quotes = {}
    verbs = {}
    sources = {}
    for _id, info_dict in id_to_info.iteritems():
        method_name_map = {
            None: None,
            'hon': 'HONORIFIC',
            'coref': 'COREF',
            'name_only': 'NAME_ONLY'
        }
        method = method_name_map.get(info_dict['gender_method'])
        people_mentioned[info_dict['name']] = \
            (info_dict['count'], (info_dict['gender'],
                                  method))
        quotes[info_dict['name']] = info_dict['quotes']
        verbs[info_dict['name']] = info_dict['associated_verbs']
        sources[info_dict['name']] = info_dict['is_source'][1]

    # return people_mentioned, quotes, None, None, None

    # people_mentioned = get_people_mentioned(sentences, corefs,
    #                                         include_gender=True)
    # quotes = get_quotes(people_mentioned, sentences, corefs)
    # verbs = get_associated_verbs(people_mentioned, sentences, corefs)
    adjectives = get_associated_adjectives(people_mentioned, sentences, corefs)
    return people_mentioned, quotes, verbs, sources, adjectives
