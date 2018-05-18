"""
Created on Sun May 13 13:37:58 2018

@author: apple
"""
from utils.dataset import Dataset
from utils.scorer import report_score

from utils.baselines import Regression
from utils.baselines import WordLength
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

from utils.improved import getWordFreq
from utils.improved import LR
from utils.improved import SVM

scores = {}
scores_tokens_len = {}

LR_results = {}
SVM_results = {}

LR_baseline_results = {}
TB_baseline_results = {}

def word_identifier(language):
    data = Dataset(language)

    
    print("{}: {} training - {} dev".format(language, len(data.trainset), len(data.testset)))

    # for sent in data.trainset:
    #    print(sent['sentence'], sent['target_word'], sent['gold_label'])
    
    #Define gold labels
    dev_gold_labels = [sent['gold_label'] for sent in data.devset]
    test_gold_labels = [sent['gold_label'] for sent in data.testset]
    train_gold_labels = [sent['gold_label'] for sent in data.trainset]
    train_data_size = len(data.trainset)
    
    #define using of  features for improved systems
    features = ['chars_len','tokens_len','vowels_len','first_upper','word_frequency'] 
    
    LR_classifier = LR(language,features)
    LR_classifier.train(data.trainset)
    LR_predictions = LR_classifier.test(data.testset)
    report_score(test_gold_labels, LR_predictions,True)
    
    SVM_classifier = SVM(language, features)
    SVM_classifier.train(data.trainset)
    SVM_predictions = SVM_classifier.test(data.testset)
    report_score(test_gold_labels, SVM_predictions,True)
    

    ''' 
    scores = []
    data_scale = 0
    while True:
        data_scale += 1000
        if train_data_size <data_scale:
            data_scale = train_data_size
            data_set = data.trainset
        else:
            data_set = data.trainset[0:data_scale]
        
          
        TB_baseline = WordLength(language)
        if language == 'english':
            length = 8
        elif language == 'spanish':
            length =10
        TB_predictions = TB_baseline.test(data.devset,length)
        fscore = report_score(dev_gold_labels, TB_predictions)
        scores.append((data_scale,fscore))
        
        if data_scale == train_data_size:
            break
        
    TB_baseline_results[language] = np.asarray(scores)   
    '''
    
    
    
    '''
    print('old - lr')
    scores = []
    data_scale = 0
    while True:
        data_scale += 1000
        if train_data_size <data_scale:
            data_scale = train_data_size
            data_set = data.trainset
        else:
            data_set = data.trainset[0:data_scale]
        
          
        LR_baseline = Regression(language)
        LR_baseline.train(data_set)
        LR_predictions = LR_baseline.test(data.devset)
        fscore = report_score(dev_gold_labels, LR_predictions)
        scores.append((data_scale,fscore))
        
        if data_scale == train_data_size:
            break
        
    LR_baseline_results[language] = np.asarray(scores)   
    
   
    

    
    print('lr')  
       
    scores = []
    data_scale = 0
    while True:
        data_scale += 1000
        if train_data_size <data_scale:
            data_scale = train_data_size
            data_set = data.trainset
        else:
            data_set = data.trainset[0:data_scale]
        
          
        LR_classifier = LR(language,features)
        LR_classifier.train(data_set)
        LR_predictions = LR_classifier.test(data.devset)
        fscore = report_score(dev_gold_labels, LR_predictions)
        scores.append((data_scale,fscore))
        
        if data_scale == train_data_size:
            break
        
    LR_results[language] = np.asarray(scores)   
    
    print('svm')     
    scores = []
    data_scale = 0
    while True:
        data_scale += 1000
        if train_data_size <data_scale:
            data_scale = train_data_size
            data_set = data.trainset
        else:
            data_set = data.trainset[0:data_scale]
       
        SVM_classifier = SVM(language, features)
        SVM_classifier.train(data_set)
        SVM_predictions = SVM_classifier.test(data.devset)
        fscore = report_score(dev_gold_labels, SVM_predictions)
        
        scores.append((data_scale,fscore))
        
        if data_scale == train_data_size:
            break
        
    SVM_results[language] = np.asarray(scores)   
    
    '''
    
    
if __name__ == '__main__':
    word_identifier('english')
    word_identifier('spanish')
    '''
    fig1, ax1 = plt.subplots(figsize=(8,4))
    ax1.set_title('Spanish learning')
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('F-score')
    
    ax1.plot(LR_baseline_results['spanish'][:,0],LR_baseline_results['spanish'][:,1],'g-',label = 'OLR-LR')
    ax1.plot(LR_results['spanish'][:,0],LR_results['spanish'][:,1],'r-',label = 'LR')
    ax1.plot(SVM_results['spanish'][:,0],SVM_results['spanish'][:,1],'b-',label = 'SVM')
    ax1.legend(loc='lower right')
    
    fig2, ax2 = plt.subplots(figsize=(8,4))
    ax2.set_xlabel('Dataset')
    ax2.set_ylabel('F-score')
    ax2.set_title('English learning')
    ax2.plot(LR_baseline_results['english'][:,0],LR_baseline_results['english'][:,1],'g-',label = 'OLD-LR')
    ax2.plot(LR_results['english'][:,0],LR_results['english'][:,1],'r-',label = 'LR')
    ax2.plot(SVM_results['english'][:,0],SVM_results['english'][:,1],'b-',label = 'SVM')
    ax2.legend(loc='lower right')
    '''
    '''
    fig1, ax1 = plt.subplots(figsize=(8,4))
    ax1.set_title('F1 scores')
    ax1.plot(LR_results['spanish'][:,0],LR_results['spanish'][:,1],'r-',label = 'LR F1-score')
    ax1.plot(SVM_results['spanish'][:,0],SVM_results['spanish'][:,1],'b-',label = 'SVM F1-score')
    ax1.legend(loc='lower right')
    '''
    
    '''
    fig1, ax1 = plt.subplots(figsize=(8,4))
    ax1.set_title('F1 scores')
    ax1.plot(LR_results['english'][:,0],LR_results['english'][:,1],'r-',label = 'LR F1-score')
    ax1.plot(SVM_results['english'][:,0],SVM_results['english'][:,1],'b-',label = 'SVM F1-score')
    ax1.legend(loc='lower right')
    '''
    
    '''
    fig, ax = plt.subplots(figsize=(8,4))
    ax.set_title('Precision and Recall')
    ax.plot(scores['english'][:,0],scores['english'][:,2],'r-',label = 'English CW Precision')
    ax.plot(scores['spanish'][:,0],scores['spanish'][:,2],'b-',label = 'Spanish CW Precision')
    ax.plot(scores['english'][:,0],scores['english'][:,3],'r--',label = 'English CW Recall')
    ax.plot(scores['spanish'][:,0],scores['spanish'][:,3],'b--',label = 'Spanish CW Recall')
    ax.legend(loc='upper right')
    
    
    fig1, ax1 = plt.subplots(figsize=(8,4))
    ax1.set_title('F1 scores')
    ax1.plot(scores['english'][:,0],scores['english'][:,1],'r-',label = 'English F1-score')
    ax1.plot(scores['spanish'][:,0],scores['spanish'][:,1],'b-',label = 'Spanish F1-score')

    ax1.legend(loc='upper right')
    '''