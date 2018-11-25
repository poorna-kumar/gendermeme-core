from collections import defaultdict, Counter
from gender_babynames import gender
from gender import gender_special
from pprint import pprint
import numpy as np
import string

VERBOSE = False

HONORIFICS = {
    'Mr': 'MALE',
    'Ms': 'FEMALE',
    'Mrs': 'FEMALE',
    'Sir': 'MALE',
    'Dr': None,
    'Dr.': None,
    'Mr.': 'MALE',
    'Mrs.': 'FEMALE',
    'Ms.': 'FEMALE'
}

RELATIONSHIP_WORDS = {
    'wife',
    'husband',
    'daughter',
    'son',
    'brother',
    'sister',
    'mother',
    'father',
}

CONF_KEYS = {
    'high_conf': 2,
    'med_conf': 1,
    'low_conf': 0
}


# In the dependencies part of the annotation, which of
# the dependency objects do we use? Up to corenlp 3.6.0,
# we were using collapsed-ccprocessed-dependencies
DEPENDENCIES_KEY = 'enhancedPlusPlusDependencies'


def get_gender(name, verbose=False):
    """
    Get the gender from a name.
    Works with full names by extracting the first name out.
    """
    name = name.upper()

    first_name = name.split()[0]
    if first_name == 'DR.':
        first_name = name.split()[1]
    found = gender.get(first_name, None)
    if not found:
        special_found = gender_special.get(name, None)
        if special_found:
            return special_found.upper()
        if verbose:
            print 'Gender not found:', name
    if type(found) is tuple:
        special_found = gender_special.get(name, None)
        if special_found:
            return special_found.upper()
        if verbose:
            print 'Ambiguous gender:', name, found
    elif type(found) is str:
        found = found.upper()
    return found


def get_people_mentioned(sentences, corefs):
    mentions_dictionary = {}
    # We've assumed that all sentences end with full stops.
    # If this assumption breaks, we might be in trouble: notice that we flush
    # the current mention at the end of every sentence, and if a full stop is
    # removed between two sentences that both start with mentions, for example,
    # then we  don't distinguish between the fact that they're in
    # different sentences.
    for sent_i, sentence in enumerate(sentences):
        tokens = sentence['tokens']
        current_mention = ''
        for token in tokens:
            if token['ner'] == 'PERSON':
                if len(current_mention) > 0:
                    current_mention += ' '
                else:
                    start_pos = (sent_i + 1, token['index'])
                current_mention += token['originalText']
                curr_pos = token['index']
            else:
                if len(current_mention) > 0:
                    key = (start_pos[0], start_pos[1], curr_pos)
                    mentions_dictionary[key] = \
                        {'text': current_mention,
                         'mention_num': 1 + len(mentions_dictionary)}
                    if key[1] > 1:
                        preceding_word = tokens[key[1]-2]['originalText']
                        if preceding_word in HONORIFICS:
                            mentions_dictionary[key]['hon_gender'] = \
                                                    HONORIFICS[preceding_word]
                            mentions_dictionary[key]['hon'] = preceding_word
                current_mention = ""

    # Add coreference information
    add_corefs_info(mentions_dictionary, corefs)

    # Add consensus gender
    add_consensus_gender(mentions_dictionary)

    # Add a flag: Is this the first time we are seeing this name,
    # and is this a single name?
    add_flag_last_name_to_be_inferred(mentions_dictionary, sentences, corefs)

    disjoint_sets_of_mentions, id_to_info, mention_key_to_id = \
        merge_mentions(mentions_dictionary)

    add_quotes(sentences, corefs, mentions_dictionary, mention_key_to_id,
               id_to_info)

    add_associated_verbs(sentences, corefs, mentions_dictionary,
                         mention_key_to_id, id_to_info)

    tag_sources(id_to_info)

    mark_companies_as_non_living(sentences, corefs, mentions_dictionary,
                                 mention_key_to_id, id_to_info)

    return id_to_info


def add_corefs_info(mentions_dictionary, corefs):
    for coref_chain_id, coref_chain in corefs.iteritems():
        mentions_pos = []
        male_pronoun_count = 0
        female_pronoun_count = 0
        it_pronoun_count = 0
        for mention_dict in coref_chain:
            pos = (mention_dict['sentNum'],
                   mention_dict['startIndex'],
                   mention_dict['endIndex'] - 1)

            # The key in mentions_dictionary which corresponds
            # to this particular mention_dict, if any.
            curr_ment_dict_key = None

            # If pos matches one of our mentions
            if pos in mentions_dictionary:
                curr_ment_dict_key = pos

            # Otherwise, if pos contains one or more of our mentions
            elif mention_dict['number'] == 'SINGULAR' and \
                    mention_dict['animacy'] == 'ANIMATE' and \
                    mention_dict['type'] == 'PROPER':
                # prospective_keys collects the keys in mentions_dictionary
                # which are located inside pos.
                # Note that there can be more than one key!
                # For example:
                # The mention text in mention_dict (in the coref chain)
                # could be
                # "David Magerman, a Renaissance research scientist
                # who was recently suspended after criticizing his boss's
                # support for Mr. Trump".
                # In this case, two mentions in mentions_dictionary
                # ('David Magerman' and 'Mr. Trump') have keys which indicate
                # that they lie within this (coref chain's) mention's text.
                # So, we extract both the keys, and store them in a
                # list called prospective_keys.
                prospective_keys = []
                for (sent_num, start_index, end_index) in \
                        mentions_dictionary:
                    if sent_num == pos[0]:
                        if start_index >= pos[1] and \
                                end_index <= pos[2]:
                            prospective_keys.append((sent_num, start_index,
                                                     end_index))
                # We now select the most appropriate mention
                # from prospective_keys in order to map the current
                # mention_dict mention to a key in mentions_dictionary
                # Currently, we pick the key in mentions
                # dictionary which corresponds to
                # the first mention in the coref chain's mention_dict text.
                # For example, if the coref chain's mention_dict's text is:
                # "David Magerman, who <blah blah> Mr. Trump", we select
                # David Magerman, as being the person who this mention
                # is really referring to.
                # Since the keys in prospective_list consist of
                # sentence numbers followed by start and end positions, merely
                # sorting prospective_keys should give us the first
                # mention first.
                if len(prospective_keys) > 0:
                    curr_ment_dict_key = min(prospective_keys)

            if curr_ment_dict_key is not None:
                mentions_pos.append(curr_ment_dict_key)
                curr_ment_dict_val = mentions_dictionary[curr_ment_dict_key]
                if 'coref_mention_ids' not in curr_ment_dict_val:
                    curr_ment_dict_val['coref_mention_ids'] = []
                curr_ment_dict_val['coref_mention_ids'].append(
                    mention_dict['id'])

            if mention_dict['type'] == 'PRONOMINAL':
                if mention_dict['gender'] == 'MALE':
                    male_pronoun_count += 1
                if mention_dict['gender'] == 'FEMALE':
                    female_pronoun_count += 1
                if mention_dict['animacy'] == 'INANIMATE' and \
                        mention_dict['number'] == 'SINGULAR':
                    it_pronoun_count += 1

        if len(mentions_pos) > 0:
            curr_pronoun_count_dict = Counter({
                "MALE": male_pronoun_count,
                "FEMALE": female_pronoun_count,
                "NON-LIVING": it_pronoun_count
            })
            for pos_i, pos in enumerate(mentions_pos):
                if 'coref_gender' not in mentions_dictionary[pos]:
                    mentions_dictionary[pos]['coref_gender'] = Counter({})
                mentions_dictionary[pos]['coref_gender'] += \
                    curr_pronoun_count_dict
                if 'coreferent_mentions' not in mentions_dictionary[pos]:
                    mentions_dictionary[pos]['coreferent_mentions'] = []
                mentions_dictionary[pos]['coreferent_mentions'].extend(
                    mentions_pos[:pos_i] + mentions_pos[pos_i + 1:])


def add_consensus_gender(mentions_dictionary):
    high_conf_thres_coref = 3
    for mention in mentions_dictionary.values():
        hon_gender = None
        coref_gender = None
        num_nonzero_conf_counts = 0
        if mention.get('hon_gender', None):
            hon_gender = mention['hon_gender']
        if mention.get('coref_gender', None):
            # get (gender, count) as a list of tuples.
            coref_counts = \
                       sorted(mention['coref_gender'].items(),
                              key=lambda tup: tup[1],
                              reverse=True)
            # find number of nonzero gender counts
            num_nonzero_coref_counts = \
                len([tup[1] for tup in coref_counts
                     if tup[1] != 0])
            coref_gender = coref_counts[0][0]
            coref_gender_count = coref_counts[0][1]
        if hon_gender:
            mention['consensus_gender'] = (hon_gender,
                                           'high_conf',
                                           'hon')
        elif coref_gender and num_nonzero_coref_counts == 1:
            if coref_gender_count >= high_conf_thres_coref:
                mention['consensus_gender'] = (coref_gender,
                                               'high_conf',
                                               'coref')
            elif coref_gender_count < high_conf_thres_coref:
                mention['consensus_gender'] = (coref_gender,
                                               'med_conf',
                                               'coref')
        elif coref_gender and num_nonzero_coref_counts > 1:
            mention['consensus_gender'] = (coref_gender,
                                           'low_conf',
                                           'coref')
        if num_nonzero_conf_counts > 1:
            mention['coref_gender_conflict'] = True
        # Haven't included name-based gender detection here
        # because that would ideally only be necessitated in
        # a later step (if no gender is found during merging)
        # In this step, it is likely to fail a lot
        # because many people could be referred to by
        # surname.


def detect_relationships(mentions_dictionary, key_to_detect, sentences,
                         corefs):

    sent_idx, start_pos, end_pos = key_to_detect
    curr_sentence = sentences[sent_idx - 1]
    dep_parse = curr_sentence[DEPENDENCIES_KEY]

    other_mentions_in_sentence = [key for key in mentions_dictionary if
                                  key[0] == sent_idx]

    related_mention_info = None

    # Trying to detect John and Jane Smith

    # We only want to do this if it's a single name (i.e, the John)
    # This is currently redundant since flag_last_name_to_infer
    # does take this into account, but kept this here in case
    # we change that function.
    if start_pos == end_pos:

        # Check if the next word is 'and'
        # Note that the index for start_pos is 1-based, so
        # we don't need to add 1 to get the element of tokens
        # that is after start_pos.
        if curr_sentence['tokens'][start_pos]['lemma'] == 'and':

            # The next word is 'and'. Now, we need to check if
            # the words right after is one of our mentions.
            mention_after_and = None
            for key in other_mentions_in_sentence:
                if key[1] == end_pos + 2:
                    mention_after_and = key
                    break

            if mention_after_and is not None:
                related_mention_info = {
                    'key': mention_after_and,
                    'rel': 'and_surname_sharing'
                }

    if related_mention_info:
        return related_mention_info

    # Detecting possessives!
    if start_pos == end_pos:

        # prev_token_idx is 0-based
        prev_token_idx = start_pos - 2
        tokens = curr_sentence['tokens']
        while tokens[prev_token_idx]['lemma'] in string.punctuation:
            prev_token_idx -= 1

        if prev_token_idx < 0:
            prev_lemma = None
        else:
            prev_lemma = tokens[prev_token_idx]['lemma']

        if prev_lemma in RELATIONSHIP_WORDS:

            possessor_idxs = []
            for dep in dep_parse:
                if dep['dep'] == 'nmod:poss':
                    # The possession link could be
                    # either 'his' -> 'Michelle' or
                    # 'his' -> 'wife' or
                    # 'Obama' -> 'Michelle' or
                    # 'Obama' -> 'wife'
                    # (The first item is the dependent
                    # and the second is the governor)
                    gov_idx = dep['governor']

                    if gov_idx == start_pos:
                        possessor_idxs.append(dep['dependent'])
                    # Since prev_token_idx is 0-based
                    elif gov_idx == prev_token_idx + 1:
                        possessor_idxs.append(dep['dependent'])

            # print key_to_detect, possessor_idxs

            if len(possessor_idxs) > 1:
                print 'TWO POSSESSORS OF THIS PERSON!'

            if len(possessor_idxs) == 1:
                possessor_idx = possessor_idxs[0]
                possessor_mention = None
                for key in other_mentions_in_sentence:
                    if key[1] <= possessor_idx <= key[2]:
                        possessor_mention = key
                        break

                if possessor_mention is None:
                    candidate_mentions =  \
                        _get_mentions_coreferent_with_word(
                            mentions_dictionary, corefs, sent_idx,
                            possessor_idx)

                    if len(candidate_mentions) > 1:
                        candidate_mentions = \
                            [cm for cm in candidate_mentions if
                             len(mentions_dictionary[cm]['text'].split()) > 1]

                    if len(candidate_mentions) > 1:
                        print 'TOO MANY CANDIDATES SURNAMES'

                    # FIXME: Should do something different if multiple
                    # surnames
                    if len(candidate_mentions) >= 1:
                        possessor_mention = candidate_mentions[0]

                if possessor_mention:
                    related_mention_info = {
                        'key': possessor_mention,
                        'rel': prev_lemma
                    }

    return related_mention_info


def add_flag_last_name_to_be_inferred(mentions_dictionary, sentences,
                                      corefs):
    """
    Add a flag: Is this the first time we are seeing this name,
    and is this a single name?
    If yes, we are on the alert for a person who is related to another person,
    and whose last name is to be inferred from the text.
    For example, in the sentence "President Barack Obama and his wife Michelle
    spoke at the gathering," we have that Michelle's last name is to be
    inferred from her relationship with her husband. Then, a "Ms. Obama" in the
    text refers to Michelle, but this connection is not made explicit.
    This is, of course, just a rough heuristic. There are cases (e.g. Lorde)
    where a person is referred to exclusively by just one name.
    """
    set_of_mentions = set()
    for key in sorted(mentions_dictionary):
        mention = mentions_dictionary[key]['text']
        if len(mention.split()) == 1:
            first_time = True
            for el in set_of_mentions:
                if mention in el:
                    first_time = False
            if first_time:
                mentions_dictionary[key]['flag_last_name_to_infer'] = True
        set_of_mentions.add(mention)

    for key in sorted(mentions_dictionary):
        if not mentions_dictionary[key].get('flag_last_name_to_infer'):
            continue

        related_mention_info = detect_relationships(
            mentions_dictionary, key, sentences, corefs)

        if related_mention_info is not None:
            related_mention = mentions_dictionary[related_mention_info['key']]
            if related_mention_info['rel'] == 'and_surname_sharing':
                mentions_dictionary[key]['potential_surname'] = \
                        related_mention['text'].split()[-1]
            elif related_mention_info['rel'] in RELATIONSHIP_WORDS:
                mentions_dictionary[key]['potential_surname'] = \
                        related_mention['text'].split()[-1]


def merge_mentions(mentions_dictionary):
    disjoint_sets_of_mentions = {}
    for key in sorted(mentions_dictionary):
        new_mention = mentions_dictionary[key]
        new_mention_text = new_mention['text']
        intersection_idx = []
        for idx, set_of_mentions in disjoint_sets_of_mentions.iteritems():
            for key_m in set_of_mentions:
                mention_text = mentions_dictionary[key_m]['text']
                # Determine whether the new mention is a subset of
                # an old mention.
                if is_mention_subset(new_mention_text, mention_text):
                    intersection_idx.append(idx)
                    break

        # This is for potential (ie, inferred) surnames
        potential_intersection_idx = []
        for idx, set_of_mentions in disjoint_sets_of_mentions.iteritems():
            if idx in intersection_idx:
                continue
            for key_m in set_of_mentions:
                mention_m = mentions_dictionary[key_m]
                if 'potential_surname' not in mention_m:
                    continue
                mention_text = '{} {}'.format(
                        mention_m['text'], mention_m['potential_surname'])
                # Determine whether the new mention is a subset of
                # an old mention.
                if is_mention_subset(new_mention_text, mention_text):
                    potential_intersection_idx.append(idx)
                    break

        # If there is an intersection, we merge the new mention into the
        # intersectiong set.
        # FIXME: Most newsrooms have style guidelines that refer to a person
        # by their full name when they first appear in the text.
        # Subsequently, they are referred to by last name alone.
        # Ideally, if everyone followed this convention, we could only
        # consider last-name overlaps (ie, if Smith would overlap with
        # Jane Smith, which would appear first).
        # However, Jane Smith could later on be referred to in a quotation
        # as Jane, and we would miss this. Also, if they style guideline
        # were not followed, and instead Jane Smith were later referred to
        # as Jane, we would miss this.
        # So, we consider any kind of overlap as a sign of life.
        # This opens the door to potential mistakes:
        # Example: "Barack and Sasha Obama took a weeklong vacation. Jim Smith
        # and his wife Sasha wisely stayed away." --> We would incorrectly
        # classify Sasha Obama and Jim Smith's wife Sasha as the same person.
        gender_match = False
        for idx in intersection_idx:
            set_of_mentions = \
                disjoint_sets_of_mentions[idx]
            if is_gender_matched(new_mention,
                                 set_of_mentions,
                                 mentions_dictionary):
                gender_match = True
            if gender_match:
                set_of_mentions.add(key)
                break

        if not gender_match:
            for idx in potential_intersection_idx:
                set_of_mentions = \
                    disjoint_sets_of_mentions[idx]
                if is_gender_matched(new_mention,
                                     set_of_mentions,
                                     mentions_dictionary):
                    gender_match = True
                if gender_match:
                    set_of_mentions.add(key)
                    break

        if not gender_match:
            idx = len(disjoint_sets_of_mentions)
            disjoint_sets_of_mentions[idx] = set([key])

    id_to_info = {}
    mention_key_to_id = {}
    for _id, set_of_mentions in disjoint_sets_of_mentions.iteritems():
        longest_mention = ''
        set_gender = None

        # The method that we used to obtain the gender
        # (This is used only for display in the web app)
        # For now, we pick the method which has the highest
        # associated confidence.
        set_gender_method = None
        highest_conf = -float("inf")

        for key in set_of_mentions:
            mention_key_to_id[key] = _id
            mention = mentions_dictionary[key]
            text = mention['text']
            if 'potential_surname' in mention:
                text += ' {}'.format(mention['potential_surname'])
            if len(text) > len(longest_mention):
                longest_mention = text

            curr_gender_tup = mention.get('consensus_gender', None)
            if curr_gender_tup:
                curr_gender = curr_gender_tup[0]
                curr_method = curr_gender_tup[2]
                curr_conf = CONF_KEYS[curr_gender_tup[1]]
                # If this is the first entry with a gender
                if not set_gender:
                    set_gender = curr_gender
                    set_gender_method = curr_method
                    highest_conf = curr_conf

                # If a previous entry had a gender,
                # we check if they match.
                else:
                    if curr_gender != set_gender:
                        # TODO: We need to look at the number of
                        # low confidence and high confidence mentions
                        # for each gender, and conclude about which gender
                        # the entity referred to by the set of mentions is.
                        # This is a temporary workaround where we're marking
                        # the gender as UNKNOWN if there is any conflict at all
                        # Of course, our current code ensures no conflict ...
                        set_gender = 'UNKNOWN'

                    else:
                        if curr_conf > highest_conf:
                            highest_conf = curr_conf
                            set_gender_method = curr_method

        if set_gender is None:
            set_gender = get_gender(longest_mention)
            set_gender_method = 'name_only'

        id_to_info[_id] = {'name': longest_mention,
                           'gender': set_gender,
                           'gender_method': set_gender_method,
                           'count': len(set_of_mentions),
                           'mentions': set_of_mentions}

    return disjoint_sets_of_mentions, id_to_info, mention_key_to_id


def is_gender_matched(new_mention, set_of_mentions,
                      mentions_dictionary):
    new_mention_gender = new_mention.get('consensus_gender', None)
    if not new_mention_gender:
        return True
    agree_counts_matrix = np.zeros((3, 3))
    disagree_counts_matrix = np.zeros((3, 3))
    row = CONF_KEYS[new_mention_gender[1]]
    # Check that the gender matches.
    for key_m in set_of_mentions:
        curr_mention = mentions_dictionary[key_m]
        curr_mention_gender = \
            curr_mention.get('consensus_gender', None)
        if not curr_mention_gender:
            continue
        col = CONF_KEYS[new_mention_gender[1]]
        if curr_mention_gender[0] == new_mention_gender[0]:
            agree_counts_matrix[row][col] += 1
        else:
            disagree_counts_matrix[row][col] += 1
    if np.sum(disagree_counts_matrix) > 0:
        return False
    else:
        return True


def is_mention_subset(small_mention_text, large_mention_text):
    """
    Check if the smaller mention is a "subset" of the larger mention.
    We define "subset" in a very specific way:
    1. Subsequence:
       Example: Barack is a subset of Barack Obama,
                John Kelly is a subset of John Kelly Smith,
                Kelly Smith is a subset of John Kelly Smith, etc.
                And, Barack is a subset of Barack.
    2. The smaller string is equal to the larger string minus the words in the
        middle.
       Example: John Smith is a subset of John Jackson Smith.
    """
    small_mention_tokens = small_mention_text.split()
    large_mention_tokens = large_mention_text.split()
    if small_mention_text in large_mention_text:
        return True
    elif len(large_mention_tokens) > 2:
        if small_mention_tokens == \
                [large_mention_tokens[0], large_mention_tokens[-1]]:
            return True
    return False


def tag_sources(id_to_info):

    SPEAKING_LEMMAS = {'say', 'tell', 'speak', 'ask', 'mention', 'suggest',
                       'claim', 'question', 'tweet', 'write'}

    for _id, info_dict in id_to_info.iteritems():
        reasons = []
        num_quotes = len(info_dict.get('quotes', []))
        if num_quotes > 0:
            reasons.append('Quoted saying {} words'.format(num_quotes))

        speaking_verbs = info_dict['associated_verbs'].intersection(
            SPEAKING_LEMMAS)
        if len(speaking_verbs) > 0:
            reasons.append('Subject of {}'.format(', '.join(speaking_verbs)))
        info_dict['is_source'] = (len(reasons) > 0, reasons)


def add_associated_verbs(sentences, corefs, mentions_dictionary,
                         mention_key_to_id, id_to_info):
    for entity_id in id_to_info:
        id_to_info[entity_id]['associated_verbs'] = set()

    word_loc_to_entity_id = _get_word_loc_to_entity_id(
        corefs, mentions_dictionary, mention_key_to_id)

    for i, sentence in enumerate(sentences):

        curr_sent_idx = i + 1  # Since CoreNLP uses 1-based indexing
        tokens = sentence['tokens']
        deps = sentence[DEPENDENCIES_KEY]

        for dep in deps:
            curr_word_loc = (curr_sent_idx, dep['dependent'])
            if curr_word_loc not in word_loc_to_entity_id:
                continue

            curr_entity_id = word_loc_to_entity_id[curr_word_loc]

            if dep['dep'] in ['nsubj', 'nsubjpass']:
                gov_token = tokens[dep['governor'] - 1]
                if gov_token['pos'].startswith('VB'):
                    id_to_info[curr_entity_id]['associated_verbs'].add(
                        gov_token['lemma'])


def mark_companies_as_non_living(sentences, corefs, mentions_dictionary,
                                 mention_key_to_id, id_to_info):

    COMPOUND_INDICATORS = ['executive', 'employee', 'attorney', 'chairman',
                           'executives', 'employees', 'attorneys',
                           'CEO', 'CTO', 'CXO', 'COO', 'founder',
                           'co-founder']
    POSS_INDICATORS = ['CEO', 'CTO', 'CXO']

    word_loc_to_entity_id = _get_word_loc_to_entity_id(
        corefs, mentions_dictionary, mention_key_to_id)

    for i, sentence in enumerate(sentences):

        curr_sent_idx = i + 1  # Since CoreNLP uses 1-based indexing
        deps = sentence[DEPENDENCIES_KEY]

        for dep in deps:
            curr_word_loc = (curr_sent_idx, dep['dependent'])
            if curr_word_loc not in word_loc_to_entity_id:
                continue

            curr_entity_id = word_loc_to_entity_id[curr_word_loc]

            if dep['dep'] == 'compound':
                governor = dep['governorGloss']
                if governor in COMPOUND_INDICATORS:
                    id_to_info[curr_entity_id]['gender'] = 'non-living'
                    id_to_info[curr_entity_id]['gender_method'] = \
                        'compound with {}'.format(governor)

            if 'poss' in dep['dep']:
                governor = dep['governorGloss']
                if governor in POSS_INDICATORS:
                    id_to_info[curr_entity_id]['gender'] = 'non-living'
                    id_to_info[curr_entity_id]['gender_method'] = \
                        'Possesses {}'.format(governor)


def _get_word_loc_to_entity_id(corefs, mentions_dictionary,
                               mention_key_to_id):

    # This dictionary is going to be useful for the task of
    # getting associations using dependency parsing.
    # Basically, we go from individual words, which are identified by
    # a (sent_idx, token_idx) tuple, both 1-based, to the entity id.
    # This will be useful when looking at the dependency parse.
    coref_mention_id_to_entity_id = {}
    for mention_key, mention_dict in mentions_dictionary.iteritems():
        coref_mention_ids = mention_dict.get('coref_mention_ids', [])
        for coref_mention_id in coref_mention_ids:
            coref_mention_id_to_entity_id[coref_mention_id] = \
                    mention_key_to_id[mention_key]

    word_loc_to_entity_id = {}

    for (sent_idx, start_idx, end_idx), _id in mention_key_to_id.iteritems():
        for curr_idx in range(start_idx, end_idx + 1):
            word_loc_to_entity_id[(sent_idx, curr_idx)] = _id

    for coref_id, coref_chain in corefs.iteritems():

        entity_in_chain = None
        for coref_mention_dict in coref_chain:
            _id = coref_mention_dict['id']
            if _id in coref_mention_id_to_entity_id:
                entity_in_chain = coref_mention_id_to_entity_id[_id]
                break

        if entity_in_chain is not None:

            for coref_mention_dict in coref_chain:
                num_words = len(coref_mention_dict['text'].split())
                if coref_mention_dict['type'] == 'PRONOMINAL' \
                        and num_words == 1:
                    sent_idx = coref_mention_dict['sentNum']
                    start_idx = coref_mention_dict['startIndex']
                    word_loc_to_entity_id[(sent_idx, start_idx)] = \
                        entity_in_chain
    return word_loc_to_entity_id


def add_quotes(sentences, corefs, mentions_dictionary,
               mention_key_to_id, id_to_info):

    for entity_id in id_to_info:
        id_to_info[entity_id]['quotes'] = []

    coref_mention_id_to_entity_id = {}
    for mention_key, mention_dict in mentions_dictionary.iteritems():
        coref_mention_ids = mention_dict.get('coref_mention_ids', [])
        for coref_mention_id in coref_mention_ids:
            coref_mention_id_to_entity_id[coref_mention_id] = \
                    mention_key_to_id[mention_key]

    unmatched_speakers_quotes = {}
    for sentence in sentences:
        for token in sentence['tokens']:
            if token.get('speaker', '').isdigit():
                speaker_id = int(token['speaker'])
                if VERBOSE:
                    print 'FOUND QUOTE'
                    print speaker_id
                if speaker_id in coref_mention_id_to_entity_id:
                    entity_id = coref_mention_id_to_entity_id[speaker_id]
                    id_to_info[entity_id]['quotes'].append(token)
                else:
                    if speaker_id not in unmatched_speakers_quotes:
                        unmatched_speakers_quotes[speaker_id] = [token]
                    else:
                        unmatched_speakers_quotes[speaker_id].append(token)

    # Try to match the unmatched speakers to known entities.

    # First, match each mention in a coreference chain to the key of the
    # coreference chain. This gives us an easy way to go back and forth
    # between mentions and the coreference chain that they are in.

    coref_chain_mention_id_to_key = {}
    for coref_id, coref_chain in corefs.iteritems():
        for mention_dict in coref_chain:
            coref_chain_mention_id_to_key[mention_dict['id']] = coref_id

    # For every unmatched speaker, find the coreference chain.

    for unmatched_speaker, curr_quotes in \
            unmatched_speakers_quotes.iteritems():
        coref_chain_key = coref_chain_mention_id_to_key[unmatched_speaker]
        coref_chain = corefs[coref_chain_key]
        # Currently only care if the speaker is singular.
        speaker_mention_dict = [el for el in coref_chain
                                if el['id'] == unmatched_speaker][0]
        if speaker_mention_dict['gender'] == 'NEUTRAL':
            speaker_gender = 'NON-LIVING'
        elif speaker_mention_dict['gender'] == 'UNKNOWN':
            speaker_gender = None
        else:
            speaker_gender = speaker_mention_dict['gender']

        possible_entities = {}  # Dict of ids to (count, gender)
        for mention_dict in coref_chain:
            if mention_dict == speaker_mention_dict:
                continue
            if mention_dict['id'] in coref_mention_id_to_entity_id:
                entity_id = coref_mention_id_to_entity_id[mention_dict['id']]
                entity_gender = id_to_info[entity_id]['gender']
                if speaker_gender and entity_gender != speaker_gender:
                    continue
                if entity_id in possible_entities:
                    possible_entities[entity_id] = (
                            possible_entities[entity_id][0] + 1,
                            possible_entities[entity_id][1])
                else:
                    possible_entities[entity_id] = (1, entity_gender)
        if len(possible_entities) == 1:
            # Go with the only speaker in the list
            for entity_id in possible_entities:
                id_to_info[entity_id]['quotes'].extend(curr_quotes)
        else:
            # More than one possible speaker: either take the speaker with
            # the higher score, or mention both with some uncertainty
            # (the latter is better).
            for entity_id in possible_entities:
                id_to_info[entity_id]['quotes'].extend(
                        [(q, 'UNCERTAIN') for q in curr_quotes])


# PRIVATE UTILITY FUNCTIONS FOLLOW


def _get_mentions_coreferent_with_word(mentions_dictionary, corefs,
                                       word_sent_idx, word_idx):
    """
    Given a particular word (which is identified by its
    sentence number and its word number, both 1-based,
    so as to match the numbering in corefs),
    returns the list of mentions from mentions_dictionary it
    is coreferent with, if any.

    Assumes that every entry in mentions_dictionary has the
    'coref_mention_ids' field populated with a list of coref
    mention ids.

    NOTE: Only matches the word if the entry in corefs contains
    exactly the word -- the idea is, if the word is 'his', then it won't
    match a phrase containing 'his', like 'his mother'.
    """

    keys_of_coreferent_mentions = set()

    coref_id_to_mention_key = {}
    for key, mentions_dict in mentions_dictionary.iteritems():
        coref_ids_of_mention = mentions_dict.get('coref_mention_ids', [])
        for coref_id in coref_ids_of_mention:
            coref_id_to_mention_key[coref_id] = key

    for coref_id, coref_chain in corefs.iteritems():
        chain_contains_word = False
        for coref_mention_dict in coref_chain:
            if coref_mention_dict['sentNum'] == word_sent_idx:
                if coref_mention_dict['startIndex'] == word_idx:
                    if coref_mention_dict['endIndex'] == word_idx + 1:
                        chain_contains_word = True
                        break

        if chain_contains_word:
            ids_in_chain = [coref_mention_dict['id'] for coref_mention_dict
                            in coref_chain]
            for _id in ids_in_chain:
                if _id in coref_id_to_mention_key:
                    keys_of_coreferent_mentions.add(
                        coref_id_to_mention_key[_id])

    return list(keys_of_coreferent_mentions)
