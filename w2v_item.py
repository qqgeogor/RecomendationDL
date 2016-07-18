import pandas as pd
import numpy as np
import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')
from datetime import datetime,timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import logging
import logging.handlers
import codecs
import json
import re
import chardet
import jieba
from gensim import corpora, models, similarities
from collections import defaultdict
seed = 1024
np.random.seed(seed)

if __name__ == '__main__':

    use_all=False

    path = "E:\\RSdata\\"
    features = ['uid','iid']
    score = 'score'

    X_train = pd.read_csv(path+"X_train.csv")
    X_test = pd.read_csv(path+"X_test.csv")
    test = pd.read_csv(path+"test.csv")

    y_train = X_train[score].values
    X_train = X_train[features].values

    y_test = X_test[score].values
    X_test = X_test[features].values
    
    test = test[features].values


    # X = np.concatenate([X_train,X_test,test])
    print('make sentences')
    sentences_5 = defaultdict(list)
    sentences_4 = defaultdict(list)
    sentences_3 = defaultdict(list)
    sentences_2 = defaultdict(list)
    sentences_1 = defaultdict(list)
    sentences = defaultdict(list)


    for u,i,yy in zip(X_train[:,0],X_train[:,1],y_train):
        u = 'u%s'%u
        i = 'i%s'%i
        sentences[u].append(i)
        if yy == 5:
            sentences_5[u].append(i)
        if yy == 4:
            sentences_4[u].append(i)
        if yy == 3:
            sentences_3[u].append(i)
        if yy == 2:
            sentences_2[u].append(i)
        if yy == 1:
            sentences_1[u].append(i)


    corpus = []
    for k in sentences.keys():
        corpus.append(sentences_5[k]+sentences_4[k]+sentences_3[k]+sentences_2[k]+sentences_1[k])

    pd.to_pickle(sentences,path+'sentences.pkl')
    del sentences
    del sentences_5
    del sentences_4
    del sentences_3
    del sentences_2
    del sentences_1
    
    
    print('w2v 128')
    model = models.Word2Vec(corpus, size=32, window=10, min_count=1, workers=8)
    model.save(path+'w2v_item_based.mdl')
    model = models.Word2Vec.load(path+'w2v_item_based.mdl')
    


    


