import numpy as np
import pandas as pd
import os
import glob
import math
import pickle
import datetime
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,LabelBinarizer
from sklearn.ensemble import RandomForestRegressor
# from keras.callbacks import ModelCheckpoint
# from keras import backend as K
# from keras.layers import Input, Embedding, LSTM, Dense,Flatten, Dropout, merge,Convolution1D,MaxPooling1D,Lambda
# from keras.layers.normalization import BatchNormalization
# from keras.optimizers import SGD
# from keras.layers.advanced_activations import PReLU,LeakyReLU,ELU,SReLU
# from keras.models import Model
# from keras.utils.visualize_util import plot
from gensim import models
from ml_metrics import rmse

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

    
    print('w2v 32')
    model = models.Word2Vec.load(path+'w2v_32.mdl')
    


    print('make sentences')
    sentences = []
    for u,i in zip(X_train[:,0],X_train[:,1]):
        s=[ 'u%s'%u,'i%s'%i]
        if s[0] in model.vocab:
            s0 = model[s[0]].ravel()
        else:
            s0 = np.zeros(32)
            
        if s[0] in model.vocab:
            s1 = model[s[0]].ravel()
        else:
            s1 = np.zeros(32)

        s = np.concatenate([s0,s1]).ravel()
        sentences.append(s)

    X_train = np.array(sentences)
    del sentences


    rf = RandomForestRegressor(n_estimators = 100,max_depth=12,n_jobs=7,random_state=seed)
    rf.fit(X_train,y_train)
    del X_train
    del y_train

    sentences = []
    for u,i in zip(X_test[:,0],X_test[:,1]):
        s=[ 'u%s'%u,'i%s'%i]
        if s[0] in model.vocab:
            s0 = model[s[0]].ravel()
        else:
            s0 = np.zeros(32)

        if s[0] in model.vocab:
            s1 = model[s[0]].ravel()
        else:
            s1 = np.zeros(32)
        s = np.concatenate([s0,s1]).ravel()
        sentences.append(s)
        
    X_test = np.array(sentences)  
    del sentences

    y_preds = rf.predict(X_test).ravel()
    score = rmse(y_test,y_preds)
    print('rf rmse score',score)

