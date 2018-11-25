from collections import namedtuple
from pprint import pprint
import unittest

from analysis import get_article_info

Expectations = namedtuple('Expectations', ["gender", "quotes", "verbs", "mention_count", "is_speaker"])


class TestGetArticleInfo(unittest.TestCase):

    def assert_expectations_are_met(self, entity_dict, expectations):
        """
        :param entity_dict: A dictionary corresponding to one of the people mentioned in a block of text.
                This represents one of the elements in the list returned by GenderMeme.
        :param expectations: An Expectations object representing expectations on the entity_dict.
                These expectations _must_ all be met.
        """
        self.assertIsNotNone(entity_dict)
        self.assertEqual(entity_dict['gender'], expectations.gender)

        quotes = [q['word'] for q in entity_dict['quotes']]
        self.assertListEqual(quotes, expectations.quotes)

        self.assertItemsEqual(entity_dict['associated_verbs'], expectations.verbs)
        self.assertEqual(entity_dict['num_times_mentioned'], expectations.mention_count)
        self.assertEqual(entity_dict['is_speaker'][0], expectations.is_speaker)

    def test_with_simple_input(self):
        text = 'Ann Smith and her husband Jim went to the movies. "It was okay," he said.'
        output = get_article_info(text, hostname='gendermeme.org', port=9000, stringify_json=False)

        # There should be two people.
        self.assertEqual(len(output), 2)

        ann_smith_dict = next(d for d in output if d['name'] == 'Ann Smith')
        self.assert_expectations_are_met(ann_smith_dict, Expectations(
            gender="FEMALE",
            quotes=[],
            verbs=["go"],
            mention_count=1,
            is_speaker=False,
        ))

        jim_smith_dict = next(d for d in output if d['name'] == 'Jim Smith')
        self.assert_expectations_are_met(jim_smith_dict, Expectations(
            gender="MALE",
            quotes=["It", "was", "okay", ","],
            verbs=["go", "say"],
            mention_count=1,
            is_speaker=True,
        ))
