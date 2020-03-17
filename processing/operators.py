import re
import string
from typing import List

import emoji
import numpy as np
import nltk
from defs import STOP_WORDS_FILE_PATH


def demojize(tweet: str) -> str:
    return emoji.demojize(tweet)  # transforms unicode emojis to strings like :smiley_face:


def to_lowercase(tweet: str) -> str:
    return tweet.lower()


def erase_numbers(tweet: str) -> str:
    return re.sub(r'\d+', '', tweet)


def erase_punctuation(tweet: str) -> str:
    return tweet.translate(str.maketrans('', '', string.punctuation))


def tokenize(tweet: str) -> List[str]:
    return tweet.split()


def remove_stop_words(tweet_tokens: List[str]) -> List[str]:
    stop_words = list(np.loadtxt(STOP_WORDS_FILE_PATH, dtype=str))

    return [word for word in tweet_tokens if word not in stop_words]


def pos_tag(tweet_tokens: List[str]) -> List[List[str]]:
    from nltk import pos_tag
    return pos_tag(tweet_tokens)


def lemmatize(tweet_tokens_tagged: List[List[str]]) -> List[str]:
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    return list(lemmatizer.lemmatize(token, pos_tag[0].lower()) if pos_tag[0].lower() in ('a', 'r', 'n', 'v') else
                token for token, pos_tag in tweet_tokens_tagged)


def name_entity_recognition(tweet_tokens: List[str]) -> List[str]:
    pass

print(lemmatize(pos_tag(tokenize("We had a lot of mice ."))))
