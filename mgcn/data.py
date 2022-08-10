import os
import re
from math import *

import pandas as pd
import numpy as np

import emoji
import wordsegment
from parsivar import Normalizer
from sklearn.model_selection import train_test_split
import nltk

from nltk.tokenize import TweetTokenizer
from somajo import SoMaJo

DATA_DIRECTORY = 'raw_data/'

wordsegment.load()
nltk.download('stopwords')

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

def read_file(data):
    fa = (data == 'fa')
    if data == 'en':
        train_name = 'olid-training-v1.0.tsv'
        train = pd.read_csv(os.path.join(DATA_DIRECTORY, train_name), sep='\t', keep_default_na=False)
        train_ids = np.array(train['id'].values)
        train_tweets = np.array(train['tweet'].values)
        train.loc[:, 'subtask_a'] = train['subtask_a'].map({'OFF':1, 'NOT':0})
        train_labels = np.array(train['subtask_a'].values)

        test_name = 'testset-levela.tsv'
        test = pd.read_csv(os.path.join(DATA_DIRECTORY, test_name), sep='\t', keep_default_na=False)
        test_ids = np.array(test['id'].values)
        test_tweets = np.array(test['tweet'].values)
        test.loc[:, 'subtask_a'] = test['subtask_a'].map({'OFF':1, 'NOT':0})
        test_labels = np.array(test['subtask_a'].values)

    elif data == 'de':
        train_name = 'germeval2018.training.txt'
        train = pd.read_csv(os.path.join(DATA_DIRECTORY, train_name), sep='\t', keep_default_na=False, header=None)
        train_ids = np.array(range(1,len(train)+1))
        train_tweets = np.array(train[0].values)
        train.loc[:, 1] = train[1].map({'OFFENSE': 1, 'OTHER': 0})
        train_labels = np.array(train[1].values)

        test_name = 'germeval2018.test.txt'
        test = pd.read_csv(os.path.join(DATA_DIRECTORY, test_name), sep='\t', keep_default_na=False, header=None)
        test_ids = np.array(range(1,len(test)+1))
        test_tweets = np.array(test[0].values)
        test.loc[:, 1] = test[1].map({'OFFENSE': 1, 'OTHER': 0})
        test_labels = np.array(test[1].values)
    
    train_tweets = process_tweets(train_tweets, fa=fa)
    test_tweets = process_tweets(test_tweets, fa=fa)

    train = pd.DataFrame({'id':train_ids, 'tweet':train_tweets, 'label': train_labels})
    test = pd.DataFrame({'id':test_ids, 'tweet':test_tweets, 'label': test_labels})
    return train, test

def sample_data(data, sample_size):
    _, new_data = train_test_split(data, test_size=sample_size, stratify=data['label'], random_state=13)
    return new_data

def pmi(tokenized_tweets, vocab, vocab2id, window_size=10):
    vocab_size = len(vocab)
    pmi = np.ones((vocab_size, vocab_size)) * -1
    windows = []

    for words in tokenized_tweets:
        length = len(words)
        if length <= window_size:
            windows.append(words)
        else:
            for j in range(length - window_size + 1):
                window = words[j: j + window_size]
                windows.append(window)

    word_window_freq = {}
    for window in windows:
        appeared = set()
        for i in range(len(window)):
            if window[i] in appeared:
                continue
            if window[i] in word_window_freq:
                word_window_freq[window[i]] += 1
            else:
                word_window_freq[window[i]] = 1
            appeared.add(window[i])

    word_pair_count = {}
    for window in windows:
        for i in range(1, len(window)):
            for j in range(0, i):
                word_i = window[i]
                word_i_id = vocab2id[word_i]
                word_j = window[j]
                word_j_id = vocab2id[word_j]
                if word_i_id == word_j_id:
                    continue
                word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1
                word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1

    num_window = len(windows)
    for key in word_pair_count:
        temp = key.split(',')
        i = int(temp[0])
        j = int(temp[1])
        count = word_pair_count[key]
        word_freq_i = word_window_freq[vocab[i]]
        word_freq_j = word_window_freq[vocab[j]]
        pmi_val = log((1.0 * count / num_window) /
                (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
        if pmi_val <= 0:
            continue
        pmi[i][j] = pmi_val
    return pmi

def generate_pmi_edges():
    pass

def tokenize(all_tweets, data):
    if data == 'en':
        tokenizer = TweetTokenizer()
        tokenized = [tokenizer.tokenize(tweet) for tweet in all_tweets]
        stopwords = nltk.corpus.stopwords.words('english')
        
    elif data == 'de':
        tokenizer = SoMaJo("de_CMC", split_camel_case=True)
        sentences = tokenizer.tokenize_text(all_tweets)
        tokenized = [[token.text for token in sentence] for sentence in sentences]
        stopwords = nltk.corpus.stopwords.words('german')

    final_tokenized = [[token for token in tweet if token not in stopwords] for tweet in tokenized]
    return final_tokenized


def prepare_data(args):
    layers = args['layers']

    for layer in layers:
        train, test = read_file(layer)
        train = sample_data(train, args['sample_size'])
        all_tweets = np.concatenate((train['tweet'].values, test['tweet'].values))
        tokenized = 


    
