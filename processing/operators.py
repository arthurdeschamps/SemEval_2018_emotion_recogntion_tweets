import re
import string
from typing import List

import emoji
import numpy as np
import nltk
from defs import STOP_WORDS_FILE_PATH
from nltk.tokenize import TweetTokenizer
from nltk import pos_tag as pos_tagger
from nltk.stem import WordNetLemmatizer
from nltk import ne_chunk

"""
In this file you will find all the available operators for data pre-processing.
"""

UNKNOWN_TOKEN = "UNK"
tokenizer = TweetTokenizer()
lemmatizer = WordNetLemmatizer()
stop_words = list(np.loadtxt(STOP_WORDS_FILE_PATH, dtype=str))


def demojize(tweet: str) -> str:
    """
    Transforms unicode emojis to strings like :smiley_face:
    """
    return emoji.demojize(tweet)


def to_lowercase(tweet_tokens: List[str]) -> List[str]:
    return list(tweet.lower() for tweet in tweet_tokens)


def substitute_numbers(tweet_tokens: List[str]) -> List[str]:
    """
    Replace all numbers with token "NUM".
    """
    return list(re.sub(r'\d+', 'NUM', tweet) for tweet in tweet_tokens)


def remove_punctuation(tweet_tokens: List[str]) -> List[str]:
    """
    Remove all punctuation tokens.
    """
    return list(token for token in tweet_tokens if token not in string.punctuation)


def tokenize(tweet: str) -> List[str]:
    """
    Transforms a plain string tweet into a list of tokens.
    """
    return tokenizer.tokenize(tweet)


def remove_stop_words(tweet_tokens: List[str]) -> List[str]:
    return [word for word in tweet_tokens if word not in stop_words]


def pos_tagging(tweet_tokens: List[str]) -> List[List[str]]:
    return pos_tagger(tweet_tokens)


def lemmatize(tweet_tokens_tagged: List[List[str]]) -> List[List[str]]:
    return list([lemmatizer.lemmatize(token, pos_tag[0].lower()), pos_tag] if pos_tag[0].lower() in ('a', 'r', 'n', 'v') else
                [token, pos_tag] for token, pos_tag in tweet_tokens_tagged)


def name_entity_recognition(tweet_tokens: List[List[str]]) -> List[str]:
    chunks = ne_chunk(tweet_tokens)
    ner_cleaned_tokens = []
    for chunk in chunks:
        if type(chunk) != nltk.Tree:
            ner_cleaned_tokens.append(chunk[0])
        elif len(chunk) == 1:
            ner_cleaned_tokens.append(chunk.label())
        else:
            ner_cleaned_tokens.append(UNKNOWN_TOKEN)
    return ner_cleaned_tokens


def flatten(tweet_tokens_tagged: List[List[str]]) -> List[str]:
    """
    Removes POS tags (or any tags associated with tokens).
    """
    return list(tweet_token_tagged[0] for tweet_token_tagged in tweet_tokens_tagged)


def substitute_tweeter_usernames(tweet_tokens: List[str]) -> List[str]:
    """
    Substitutes tweeter usernames with the token "USERNAME_TOKEN".
    """
    return list("USERNAME_TOKEN" if token[0] == "@" else token for token in tweet_tokens)


def substitute_underscores(tweet: str) -> str:
    return re.sub(r'(.)_(.)', r'\g<1> \g<2>', tweet)


def strip_hashtags(tweet_tokens: List[str]) -> List[str]:
    return list(token[1:] if (len(token) > 1) and (token[0] == "#") else token for token in tweet_tokens)
