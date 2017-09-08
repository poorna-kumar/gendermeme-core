import json
from pprint import pprint
from analysis.analysis import get_article_info

if __name__ == "__main__":
     text = 'Ann Smith and her husband Jim went to the movies. "It was okay," he said.'
     pprint(json.loads(get_article_info(text)))
