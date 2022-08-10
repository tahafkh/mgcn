import os
import re
import pandas as pd
import numpy as np
import emoji
import wordsegment
from parsivar import Normalizer

DATA_DIRECTORY = 'raw_data/'

wordsegment.load()

def process_tweets(tweets, fa=False):
    # Process tweets
    tweets = emoji2word(tweets)
    tweets = remove_links(tweets)
    tweets = remove_usernames(tweets)
    tweets = replace_rare_words(tweets)
    tweets = remove_replicates(tweets)
    tweets = segment_hashtag(tweets)
    tweets = lower_case(tweets)
    tweets = remove_useless_punctuation(tweets)
    if fa == True:
        tweets = remove_eng(tweets)
        tweets = normalize(tweets)
    tweets = np.array(tweets)
    return tweets

def lower_case(sents):
    for i, sent in enumerate(sents):
        sents[i] = sent.lower()
    return sents

def normalize(sents):
    normalizer = Normalizer()
    for i, sent in enumerate(sents):
      sents[i] = normalizer.normalize(str(sent))
    return sents

def remove_links(sents):
    for i, sent in enumerate(sents):
        sents[i] = re.sub(r'^https?:\/\/.*[\r\n]*', 'http', str(sent), flags=re.MULTILINE)
    return sents

def remove_eng(sents):
    for i, sent in enumerate(sents):
        sents[i] = re.sub('[a-zA-Z0-9]','',str(sent))
    return sents

def remove_usernames(sents):
    for i, sent in enumerate(sents):
        sents[i] = re.sub('@[^\s]+','@USER',str(sent))
    return sents

def emoji2word(sents):
    return [emoji.demojize(str(sent)) for sent in sents]

def remove_useless_punctuation(sents):
    for i, sent in enumerate(sents):
        sent = sent.replace(':', ' ')
        sent = sent.replace('_', ' ')
        sent = sent.replace('...', ' ')
        sent = sent.replace('..', ' ')
        sent = sent.replace('’', '')
        sent = sent.replace('"', '')
        sent = sent.replace(',', '')
        sents[i] = sent
    return sents

def remove_replicates(sents):
    # if there are multiple `@USER` tokens in a tweet, replace it with `@USERS`
    # because some tweets contain so many `@USER` which may cause redundant
    for i, sent in enumerate(sents):
        if sent.find('@USER') != sent.rfind('@USER'):
            sents[i] = sent.replace('@USER ', '')
            sents[i] = '@USERS ' + sents[i]
    return sents

def replace_rare_words(sents):
    rare_words = {
        'URL': 'http'
    }
    for i, sent in enumerate(sents):
        for w in rare_words.keys():
            sents[i] = sent.replace(w, rare_words[w])
    return sents

def segment_hashtag(sents):
    # E.g. '#LunaticLeft' => 'lunatic left'
    for i, sent in enumerate(sents):
        sent_tokens = sent.split(' ')
        for j, t in enumerate(sent_tokens):
            if t.find('#') == 0:
                sent_tokens[j] = ' '.join(wordsegment.segment(t))
        sents[i] = ' '.join(sent_tokens)
    return sents
