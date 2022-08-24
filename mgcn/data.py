import os
import re
import json
from math import *

import pandas as pd
import numpy as np

import emoji
import wordsegment
from parsivar import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

from nltk.tokenize import TweetTokenizer
from somajo import SoMaJo

from transformers import (
        BertForSequenceClassification, BertTokenizer, \
        XLMForSequenceClassification, XLMTokenizer, \
        XLMRobertaForSequenceClassification, XLMRobertaTokenizer)

import torch
from torch.utils.data import DataLoader, TensorDataset

RAW_DATA_DIRECTORY = 'raw_data'
DATA_DIRECTORY = 'data'
DATASET = 'sample'

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
        train = pd.read_csv(os.path.join(RAW_DATA_DIRECTORY, train_name), sep='\t', keep_default_na=False)
        train_ids = np.array(train['id'].values)
        train_tweets = np.array(train['tweet'].values)
        train.loc[:, 'subtask_a'] = train['subtask_a'].map({'OFF':1, 'NOT':0})
        train_labels = np.array(train['subtask_a'].values)

        test_name = 'testset-levela.tsv'
        test = pd.read_csv(os.path.join(RAW_DATA_DIRECTORY, test_name), sep='\t', keep_default_na=False)
        test_ids = np.array(test['id'].values)
        test_tweets = np.array(test['tweet'].values)
        test.loc[:, 'subtask_a'] = test['subtask_a'].map({'OFF':1, 'NOT':0})
        test_labels = np.array(test['subtask_a'].values)

    elif data == 'de':
        train_name = 'germeval2018.training.txt'
        train = pd.read_csv(os.path.join(RAW_DATA_DIRECTORY, train_name), sep='\t', keep_default_na=False, header=None)
        train_ids = np.array(range(1,len(train)+1))
        train_tweets = np.array(train[0].values)
        train.loc[:, 1] = train[1].map({'OFFENSE': 1, 'OTHER': 0})
        train_labels = np.array(train[1].values)

        test_name = 'germeval2018.test.txt'
        test = pd.read_csv(os.path.join(RAW_DATA_DIRECTORY, test_name), sep='\t', keep_default_na=False, header=None)
        test_ids = np.array(range(1,len(test)+1))
        test_tweets = np.array(test[0].values)
        test.loc[:, 1] = test[1].map({'OFFENSE': 1, 'OTHER': 0})
        test_labels = np.array(test[1].values)
    
    else:
        raise ValueError(f'Data {data} not supported.')

    train_tweets = process_tweets(train_tweets, fa=fa)
    test_tweets = process_tweets(test_tweets, fa=fa)

    train = pd.DataFrame({'id':train_ids, 'tweet':train_tweets, 'label': train_labels})
    test = pd.DataFrame({'id':test_ids, 'tweet':test_tweets, 'label': test_labels})
    return train, test

def sample_data(data, sample_size):
    _, new_data = train_test_split(data, test_size=sample_size, stratify=data['label'], random_state=13)
    return new_data

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

    else:
        raise ValueError(f'Data {data} not supported.')

    final_tokenized = [[token for token in tweet if token not in stopwords] for tweet in tokenized]
    return final_tokenized

def create_tfidf_edges(tokenized, doc_numbers):
    tfidf = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
    tfidf_matrix = tfidf.fit_transform(tokenized)
    vocab = list(tfidf.get_feature_names_out())
    vocab2id = {vocab[i]:i for i in range(len(vocab))}
    id2vocab = {i:vocab[i] for i in range(len(vocab))}
    edges = []
    for i in range(tfidf_matrix.shape[0]):
        for j in range(tfidf_matrix.shape[1]):
            if tfidf_matrix[i,j] > 0:
                edges.append([i, j + doc_numbers])
    print('TFIDF edges created.')
    return edges, vocab, vocab2id, id2vocab

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

def create_pmi_edges(tokenized_tweets, vocab, vocab2id, doc_numbers, window_size=10):
    pmi_matrix = pmi(tokenized_tweets, vocab, vocab2id, window_size)
    edges = []
    for i in range(pmi_matrix.shape[0]):
        for j in range(pmi_matrix.shape[1]):
            if pmi_matrix[i,j] > 0:
                edges.append([i + doc_numbers, j + doc_numbers])
    print('PMI edges created.')
    return edges    

def create_layer_adj(i, layer, args):
    train, test = read_file(layer)
    sampled_train = sample_data(train, args['train_sizes'][i])
    sampled_test = sample_data(test, args['test_sizes'][i])
    train_and_test = pd.concat([sampled_train, sampled_test], ignore_index=True)
    all_tweets = list(train_and_test['tweet'].values)
    doc_numbers = len(all_tweets)
    tokenized = tokenize(all_tweets, layer)
    tfidf_edges, vocab, vocab2id, id2vocab = create_tfidf_edges(tokenized, doc_numbers)
    pmi_edges = create_pmi_edges(tokenized, vocab, vocab2id, doc_numbers)
    edges = tfidf_edges + pmi_edges
    file = f'{DATASET}.adj{i}'
    with open(os.path.join(DATA_DIRECTORY, DATASET, file), 'w') as f:
        for edge in edges:
            f.write(f'{edge[0]} {edge[1]}\n')
    layer_dict = {
        'train': train,
        'test': test,
        'train_and_test': train_and_test,
        'all_tweets': all_tweets,
        'tokenized': tokenized,
        'vocab': vocab,
        'vocab2id': vocab2id,
        'id2vocab': id2vocab,
        'doc_numbers': doc_numbers,
    }
    print('Layer adjacency matrix created.')
    return layer_dict

def initialize():
    if not os.path.exists(DATA_DIRECTORY):
        os.makedirs(DATA_DIRECTORY)
    if not os.path.exists(os.path.join(DATA_DIRECTORY, DATASET)):
        os.makedirs(os.path.join(DATA_DIRECTORY, DATASET))
    
    wordsegment.load()
    nltk.download('stopwords')

def create_adj(args):
    layers = args['layers']
    layers_dict = {}
    for i, layer in enumerate(layers):
        layer_dict = create_layer_adj(i, layer, args)
        layers_dict[layer] = layer_dict
    return layers_dict

def create_bidict_edges(args, layers_dict):
    for file in os.listdir(RAW_DATA_DIRECTORY):
        if file.endswith(".json"):
            layers = file[:-5].split('2')
            create_layer_bidict_edges(layers[0], layers[1], args['layer2id'], layers_dict)
    print('Between layer edges created.')

def create_layer_bidict_edges(first, second, layer2id, layers_dict):
    file = f'{first}2{second}.json'
    with open(os.path.join(RAW_DATA_DIRECTORY, file)) as json_file:
        words_dict = json.load(json_file)
    
    first_doc_numbers = layers_dict[first]['doc_numbers']
    second_doc_numbers = layers_dict[second]['doc_numbers']
    edges = []
    for word in words_dict:
        if word in layers_dict[first]['vocab']:
            first_word_id = layers_dict[first]['vocab2id'][word]
            similar_words = words_dict[word][:3]
            for similar_word in similar_words:
                if similar_word in layers_dict[second]['vocab']:
                    second_word_id = layers_dict[second]['vocab2id'][similar_word]
                    edges.append([first_word_id + first_doc_numbers, second_word_id + second_doc_numbers])

    file = f'{DATASET}.bet{layer2id[first]}_{layer2id[second]}'
    with open(os.path.join(DATA_DIRECTORY, DATASET, file), 'w') as f:
        for edge in edges:
            f.write(f'{edge[0]} {edge[1]}\n')

def create_bet(args, layers_dict):
    method = args['method']
    if method == 'bidict':
        create_bidict_edges(args, layers_dict)
    else:
        raise ValueError(f'Method {method} not implemented.')

def get_model_cls(model):
    if model == "bert":
        return "bert-base-multilingual-uncased", BertTokenizer, BertForSequenceClassification
    elif model == "xlm":
        return "xlm-mlm-17-1280", XLMTokenizer, XLMForSequenceClassification
    elif model == "xlmr":
        return "xlm-roberta-base", XLMRobertaTokenizer, XLMRobertaForSequenceClassification
    else:
        raise NameError(f"Unsupported model {model}.")

def create_dataloader(tokenizer, args, inputs, labels):
    encoded_dict = tokenizer.batch_encode_plus(
            inputs,                           
            add_special_tokens=True,      
            max_length=args['max_length'],        
            pad_to_max_length=True,
            return_attention_mask=True,   
            return_tensors='pt',
    )
    dataset = TensorDataset(encoded_dict["input_ids"], encoded_dict["attention_mask"], torch.tensor(labels))
    return DataLoader(dataset, batch_size=args['batch_size'], shuffle=False)

def create_outputs(model, dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    model.to(device)

    model.eval()
    outputs = []
    for batch in dataloader:
        input_ids = batch[0].to(device)
        input_mask = batch[1].to(device)
        
        with torch.no_grad():        
            out = model(input_ids, 
                        token_type_ids=None, 
                        attention_mask=input_mask)
            hidden_states = out[-1]
            outputs.append(hidden_states[-1][:, 0,:].detach().cpu().numpy())
        
    model.to('cpu')
    return outputs

def create_layer_feature(i, layer, model, tokenizer, layers_dict, args):
    inputs = layers_dict[layer]['all_tweets'] + layers_dict[layer]['vocab']
    labels = list(layers_dict[layer]['train_and_test']['label'].values)
    all_labels = np.array([int(label) for label in labels] + \
        [-1 for _ in range(len(inputs) - len(labels))]).reshape(-1, 1)
    dataloader = create_dataloader(tokenizer, args, inputs, all_labels)
    outputs = create_outputs(model, dataloader)
    outputs = np.concatenate(outputs)
    ids = np.arange(outputs.shape[0]).reshape(-1, 1)
    ids_embeddings = np.append(ids, outputs, 1)
    ids_embeddings = np.append(ids_embeddings, all_labels, 1)
    file = f'{DATASET}.feat{i}'
    file_path = os.path.join(DATA_DIRECTORY, DATASET, file)
    np.savetxt(file_path, ids_embeddings)
    
def finetune(model, tokenizer, layer, args, layers_dict):
    layer_train = sample_data(layers_dict[layer]['train'], 2000)
    dataloader = create_dataloader(tokenizer, args, layer_train['tweet'], layer_train['label'].values)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    model.to(device)

    model.train()
    epochs = 2
    for epoch in range(epochs):
        for batch in dataloader:
            input_ids = batch[0].to(device)
            input_mask = batch[1].to(device)
            labels = batch[2].to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                out = model(input_ids, 
                            token_type_ids=None, 
                            attention_mask=input_mask)
                loss = criterion(out[0], labels)
                loss.backward()
                optimizer.step()
    model.to('cpu')
    print('Finetuned.')


def prepare_model(layer, args, layers_dict):
    model_name, tokenizer_cls, model_cls = get_model_cls(args['model'])
    tokenizer = tokenizer_cls.from_pretrained(model_name, do_lower_case=(args['model'] == "bert"))
    num_labels = layers_dict[layer]['train']['label'].nunique()
    model = model_cls.from_pretrained(model_name, num_labels=num_labels, output_hidden_states=True)
    if args['finetune']:
        finetune(model, tokenizer, layer, args, layers_dict)
    return model, tokenizer

def create_feature(args, layers_dict):
    layers = args['layers']
    for i, layer in enumerate(layers):
        model, tokenizer = prepare_model(layer, args, layers_dict)
        create_layer_feature(i, layer, model, tokenizer, layers_dict, args)
    print('Features created.')

def prepare_data(args):
    initialize()
    layers_dict = create_adj(args)
    create_bet(args, layers_dict)
    create_feature(args, layers_dict)
    print('Data prepared for MGCN.')
