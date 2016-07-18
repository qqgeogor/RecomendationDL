import numpy as np
import pandas as pd
import os
import glob
import math
import pickle
import datetime
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,LabelBinarizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import resample
# from keras.callbacks import ModelCheckpoint
# from keras import backend as K
# from keras.layers import Input, Embedding, LSTM, Dense,Flatten, Dropout, merge,Convolution1D,MaxPooling1D,Lambda
# from keras.layers.normalization import BatchNormalization
# from keras.optimizers import SGD
# from keras.layers.advanced_activations import PReLU,LeakyReLU,ELU,SReLU
# from keras.models import Model
# from keras.utils.visualize_util import plot
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

    
    X_train,y_train = resample(X_train,y_train,n_samples = X_train.shape[0]/10, random_state =seed)
    X_test,y_test = resample(X_test,y_test,n_samples = X_test.shape[0]/10, random_state =seed)
    
    
    X = np.concatenate([X_train,X_test,test])

    user_le = LabelEncoder()
    item_le = LabelEncoder()

    user_le.fit(X[:,0])
    item_le.fit(X[:,1])

    X_train[:,0] = user_le.transform(X_train[:,0])
    X_train[:,1] = item_le.transform(X_train[:,1])

    if use_all:
        X = np.concatenate([X_train,X_test])
        y_train = np.concatenate([y_train,y_test])
        X_train[:,0] = user_le.transform(X[:,0])
        X_train[:,1] = item_le.transform(X[:,1])

    X_test[:,0] = user_le.transform(X_test[:,0])
    X_test[:,1] = item_le.transform(X_test[:,1])
        
        
    rf = RandomForestRegressor(n_estimators = 100,max_depth=12,n_jobs=7,random_state=seed)
    rf.fit(X_train,y_train)
    y_preds = rf.predict(X_test).ravel()
    score = rmse(y_test,y_preds)
    print('rf rmse score',score)
    