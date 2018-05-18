#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 22:48:08 2018

@author: apple
"""
import re, nltk
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from collections import Counter
from gensim.models import word2vec


def getWordFreq(dataset):
    docs = []
    temp = ""
    for data in dataset:
        sentence = data['sentence']
        if sentence != temp:        
            sent = re.findall('[\w]+',sentence)
            temp = sentence
            docs.extend(sent)
    return docs


def wordEmbedding(dataset):
    sents = []
    temp = ""
    for data in dataset:
        sentence = data['sentence']
        if sentence != temp:
            sent = re.findall('[\w]+',sentence)
            if 'barren' in sent:
                print(sent)
            temp = sentence
            sents.append(sent)
    print(sents[0])
    model=word2vec.Word2Vec(sents, sg=1,size=100,min_count=1)
    return model

def extract_features(object, sent):
    results = []
    word = sent['target_word']
    sentence = sent['sentence']
    
    for feature in object.features:
        if feature == "chars_len":
            len_chars = len(word) / object.avg_word_length
            results.append(len_chars)
            
        elif feature == "tokens_len":
            len_tokens = len(word.split(' '))
            results.append(len_tokens)
            
        elif feature == "vowels_len":
            vowels = 0
            for char in word.lower():
                if char in ['a','e','i','o','u']:
                    vowels += 1
            len_tokens = len(word.split(' '))
            results.append(vowels/len_tokens)
            
        elif feature == "first_upper":
            first_character = word[0]
            if first_character.isupper():
                results.append(1)
            else:
                results.append(0)
                
        elif feature == "word_frequency":
            words = word.split(' ')
            freqs = []
            for w in words:
                freqs.append(object.word_to_freq[word])
            results.append(sorted(freqs)[0])
    
    return results
        
class SVM(object):
    def __init__(self, language,features):
        self.language = language
        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if language == 'english':
            self.avg_word_length = 5.3
        else:  # spanish
            self.avg_word_length = 6.2
        self.features = features
        self.model = SVC()

    def train(self, trainset):
        self.word_to_freq = Counter(getWordFreq(trainset))        
        X = []
        y = []
        for sent in trainset:
            X.append(extract_features(self,sent))
            y.append(sent['gold_label'])

        self.model.fit(X, y)


    def test(self, testset):
        X = []
        for sent in testset:
            X.append(extract_features(self,sent))
        results = self.model.predict(X)
        return results

class LR(object):

    def __init__(self, language, features):
        self.language = language
        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if language == 'english':
            self.avg_word_length = 5.3
        else:  # spanish
            self.avg_word_length = 6.2
        self.features = features
        self.model = LogisticRegression()



    def train(self, trainset):
        self.word_to_freq = Counter(getWordFreq(trainset))
        X = []
        y = []
        for sent in trainset:
            X.append(extract_features(self,sent))
            y.append(sent['gold_label'])

        self.model.fit(X, y)


    def test(self, testset):
        self.word_to_freq.update(Counter(getWordFreq(testset)))
        X = []
        for sent in testset:
            
            X.append(extract_features(self,sent))
            
        results = self.model.predict(X)
        
        
        return results    
        
