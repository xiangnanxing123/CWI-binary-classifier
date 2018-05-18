#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 11:15:55 2018

@author: apple
"""

from sklearn.linear_model import LogisticRegression


class WordLength(object):
    def __init__(self, language, ):
        self.language = language
    

    def test(self,test_sent, length):
        predicts = []
        for sent in test_sent:
            if len(sent['target_word']) >= length :
                predicts.append('1')
            else:
                predicts.append('0')
        return predicts
    



class Regression(object):

    def __init__(self, language):
        self.language = language
        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if language == 'english':
            self.avg_word_length = 5.3
        else:  # spanish
            self.avg_word_length = 6.2

        self.model = LogisticRegression()

    def extract_features(self, word):
        len_chars = len(word) / self.avg_word_length
        len_tokens = len(word.split(' '))
        return [len_chars,len_tokens]

    def train(self, trainset):
        X = []
        y = []
        for sent in trainset:
            X.append(self.extract_features(sent['target_word']))
            y.append(sent['gold_label'])

        self.model.fit(X, y)


    def test(self, testset):
        X = []
        for sent in testset:
            
            X.append(self.extract_features(sent['target_word']))
            
        results = self.model.predict(X)
        
        
                
            
        return results

        
    
    