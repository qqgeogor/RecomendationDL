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

    # y_train = X_train[score].values
    X_train = X_train[features].values

    # y_test = X_test[score].values
    X_test = X_test[features].values

    test = test[features].values


    # X = np.concatenate([X_train,X_test,test])
    print('make sentences')
    sentences = []
    for u,i in zip(X_train[:,0],X_train[:,1]):
        sentences.append([ 'u%s'%u,'i%s'%i])

    del X_train

    for u,i in zip(X_test[:,0],X_test[:,1]):
        sentences.append([ 'u%s'%u,'i%s'%i])

    del X_test

    for u,i in zip(test[:,0],test[:,1]):
        sentences.append([ 'u%s'%u,'i%s'%i])

    del test

    
    # print('w2v 32')
    # model = models.Word2Vec(sentences, size=32, window=2, min_count=1, workers=7)
    # model.save(path+'w2v_32.mdl')
    # model = models.Word2Vec.load(path+'w2v_32.mdl')
    
    # print('w2v 64')
    # model = models.Word2Vec(sentences, size=64, window=2, min_count=1, workers=7)
    # model.save(path+'w2v_64.mdl')
    # model = models.Word2Vec.load(path+'w2v_64.mdl')

    print('w2v 128')
    model = models.Word2Vec(sentences, size=128, window=2, min_count=1, workers=8)
    model.save(path+'w2v_128.mdl')
    model = models.Word2Vec.load(path+'w2v_128.mdl')


    


