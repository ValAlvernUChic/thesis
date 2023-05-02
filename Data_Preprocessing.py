#!/usr/bin/env python
# coding: utf-8


import os
import format_hansard
import pandas as pd
import json
import gensim
import sklearn
from gensim.models import Word2Vec
from tqdm import tqdm
import string
import copy
import nltk
tqdm.pandas()
import spacy
nlp = spacy.load('en_core_web_sm')

nltk.download('stopwords')

files = []
def scan_folder(parent):
    """
    Collects relevant file names from data folder

    Inputs:
        parent - parent file directory to start your search;
        shouuld end with .../hansard_full 2/

    Returns:
        List of filenames
    """
    files = []
    # iterate over all the files in directory 'parent'
    for file_name in os.listdir(parent):
        if file_name.endswith(".txt"):
            # if it's a txt file, print its name (or do whatever you want)
            files.append("".join((parent, "/", file_name)))
        else:
            current_path = "".join((parent, "/", file_name))
            if os.path.isdir(current_path):
                # if we're checking a sub-directory, recursively call this method
                scan_folder(current_path)
    return files


def word_tokenize(word_list):
    """
    Turns string of words into a list of tokens
    
    Input: 
        word_list - list of strings 
    
    Returns: 
        list of tokens
    """
    tokenized = []
    # pass word list through language model.
    doc = nlp(word_list)
    for token in doc:
        if not token.is_punct and len(token.text.strip()) > 0:
            tokenized.append(token.text)
    return tokenized

def migrant_worker(y):
    """
    Converts 'migrant worker' to one token 
    
    Input: 
        y - list of strings
    
    Returns: 
        string with migrant worker replaced
    """
    if type(y) == float:
        y = str(y)
    if 'migrant worker' in y:
        return '. '.join([x.replace('migrant worker', 'migrant-worker')
                          for x in y.split('. ') if 'migrant worker' in x])
    else:
        return y

def normalizeTokens(word_list, extra_stop=[]):
    """
    Normalizes the tokenized sentences

    Inputs:
        word_list: list of tokenized words

    Returns:
        List of normalized words
    """
    # We can use a generator here as we just need to iterate over it
    normalized = []
    if type(word_list) == list and len(word_list) == 1:
        word_list = word_list[0]

    if type(word_list) == list:
        word_list = ' '.join([str(elem) for elem in word_list])

    doc = nlp(word_list.lower())

    # add the property of stop word to words considered as stop words
    if len(extra_stop) > 0:
        for stopword in extra_stop:
            lexeme = nlp.vocab[stopword]
            lexeme.is_stop = True

    for w in doc:
        # if it's not a stop word or punctuation mark, add it to our article
        if w.text != '\n' and not w.is_stop and not w.is_punct and not w.like_num and len(
                w.text.strip()) > 0:
            # we add the lematized version of the word
            normalized.append(str(w.lemma_))

    return [[x] for x in normalized]

def clean(df):
    """
    Cleans the collected dataframe

    Inputs:
        df - dataframe to be cleaned

    Returns:
        Cleaned df
    """
#     punc = '!()-[]{};:'"\,<>./?@#$%^&*_~'
    df = pd.concat(df, ignore_index=True)
    df = df.dropna()
    df['clean'] = df[df.iloc[:,0].str.len() >= 20]
    df['clean'] = df['clean'].apply(lambda x: migrant_worker(x))
    df['clean'] = df['clean'].progress_apply(lambda x: ' '.join(x.lower().translate(str.maketrans('', '', string.punctuation))
                                    .split() if type(x)==str else x))
    df['tokenized'] = df['clean'].progress_apply(lambda x: word_tokenize(x))
    df['normalized'] = df['tokenized'].progress_apply(
        lambda x: normalizeTokens(x))
    df['normalized_sents'] = df['normalized'].progress_apply(
        lambda x: to_list(x))

    return df.dropna().reset_index()

def create_df(files, savepath):
    """
    Creates a concatenated dataframe

    Files - Collated list of filenames
    Savepath -
    """
    pd12 = []
    pd13 = []
    pd14 = []
    pd15 = []
    pd16 = []
    pd17 = []
    pd18 = []
    pd19 = []
    pd20 = []
    pd21 = []
    massive = []
    for file in tqdm(files):
        if '2012' in file:
            f = open(file, 'r')
            lis = [x.lower().replace('\xa0', '').replace('\n', '').split(':  ')
                   for x in f.readlines()]
            f.close()
            df = pd.DataFrame(lis)
            df.columns = ['text']
            pd12.append(df)
        if '2013' in file:
            f = open(file, 'r')
            lis = [x.lower().replace('\xa0', '').replace('\n', '').split(':  ')
                   for x in f.readlines()]
            f.close()
            df = pd.DataFrame(lis)
            df.columns = ['text']
            pd13.append(df)
        if '2014' in file:
            f = open(file, 'r')
            lis = [x.lower().replace('\xa0', '').replace('\n', '').split(':  ')
                   for x in f.readlines()]
            f.close()
            df = pd.DataFrame(lis)
            df.columns = ['text']
            pd14.append(df)
        if '2015' in file:
            f = open(file, 'r')
            lis = [x.lower().replace('\xa0', '').replace('\n', '').split(':  ')
                   for x in f.readlines()]
            f.close()
            df = pd.DataFrame(lis)
            df.columns = ['text']
            pd15.append(df)
        if '2016' in file:
            f = open(file, 'r')
            lis = [x.lower().replace('\xa0', '').replace('\n', '').split(':  ')
                   for x in f.readlines()]
            f.close()
            df = pd.DataFrame(lis)
            df.columns = ['text']
            pd16.append(df)
        if '2017' in file:
            f = open(file, 'r')
            lis = [x.lower().replace('\xa0', '').replace('\n', '').split(':  ')
                   for x in f.readlines()]
            f.close()
            df = pd.DataFrame(lis)
            df.columns = ['text']
            pd17.append(df)
        if '2018' in file:
            f = open(file, 'r')
            lis = [x.lower().replace('\xa0', '').replace('\n', '').split(':  ')
                   for x in f.readlines()]
            f.close()
            df = pd.DataFrame(lis)
            df.columns = ['text']
            pd18.append(df)
        if '2019' in file:
            f = open(file, 'r')
            lis = [x.lower().replace('\xa0', '').replace('\n', '').split(':  ')
                   for x in f.readlines()]
            f.close()
            df = pd.DataFrame(lis)
            df.columns = ['text']
            pd19.append(df)
        if '2020' in file:
            f = open(file, 'r')
            lis = [x.lower().replace('\xa0', '').replace('\n', '').split(':  ')
                   for x in f.readlines()]
            f.close()
            df = pd.DataFrame(lis)
            df.columns = ['text']
            pd20.append(df)
        if '2021' in file:
            f = open(file, 'r')
            lis = [x.lower().replace('\xa0', '').replace('\n', '').split(':  ')
                   for x in f.readlines()]
            f.close()
            df = pd.DataFrame(lis)
            df.columns = ['text']
            pd21.append(df)

    massive.append(clean(pd12))
    massive.append(clean(pd13))
    massive.append(clean(pd14))
    massive.append(clean(pd15))
    massive.append(clean(pd16))
    massive.append(clean(pd17))
    massive.append(clean(pd18))
    massive.append(clean(pd19))
    massive.append(clean(pd20))
    massive.append(clean(pd21))

    for i, file in zip(range(12,22), massive):
        file.to_csv('{}/{}.csv'.format(savepath, i))

def create_final_df(path, savepath):
    """
    Creates final dataframe for input into word embedding models

    Inpuuts:
        Path - Path where individual yearly csv files are held
        Savepath - Path to save your final df

    Returns:
        None
    """
    speeches = 0
    final_df = pd.DataFrame(columns = ['year', '0', 'clean', 'tokenized'])
    for i in range(12,22):
        file = pd.read_csv(path + '{}.csv'.
                             format(i))
        speeches += len(file)
        file['year'] = int('20{}'.format(i))
        file = file[['year', '0', 'clean', 'tokenized']]
        final_df = pd.concat([final_df, file], ignore_index = True)
        final_df['normalized_sents'] = \
            final_df['tokenized'].apply(lambda x: normalizeTokens(x))

    final_df = final_df.dropna()
    final_df.to_csv(savepath)

