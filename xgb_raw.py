import numpy as np
import pandas as pd
from scipy import sparse as ssp
import os
import glob
import math
import pickle
import datetime
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,LabelBinarizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import resample
from sklearn.decomposition import TruncatedSVD
# from keras.callbacks import ModelCheckpoint
# from keras import backend as K
# from keras.layers import Input, Embedding, LSTM, Dense,Flatten, Dropout, merge,Convolution1D,MaxPooling1D,Lambda
# from keras.layers.normalization import BatchNormalization
# from keras.optimizers import SGD
# from keras.layers.advanced_activations import PReLU,LeakyReLU,ELU,SReLU
# from keras.models import Model
# from keras.utils.visualize_util import plot
from ml_metrics import rmse
import xgboost as xgb
seed = 1024
np.random.seed(seed)


def user_matrix(X,y,num_u,num_i):
    return ssp.csr_matrix((y,(X[:,0],X[:,1])),shape=(num_u,num_i), dtype=np.int8)


def item_matrix(X,y,num_u,num_i):
    return ssp.csr_matrix((y,(X[:,1],X[:,0])),shape=(num_i,num_u), dtype=np.int8)

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

    

    
    
    X = np.concatenate([X_train,X_test,test])
    num_u = len(np.unique(X[:,0]))
    num_i = len(np.unique(X[:,1]))
    # print num_u,num_i


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
    
    
    
    um = user_matrix(X_train,y_train,num_u,num_i)
    im = item_matrix(X_train,y_train,num_u,num_i)
    # print um.shape,im.shape
    um_svd = TruncatedSVD(n_components=32)
    um = um_svd.fit_transform(um)
    print 'um',um.shape
    im_svd = TruncatedSVD(n_components=32)
    im = im_svd.fit_transform(im)
    print 'im',im.shape

    X_train,y_train = resample(X_train,y_train,n_samples = X_train.shape[0]/10, random_state =seed)
    X_test,y_test = resample(X_test,y_test,n_samples = X_test.shape[0]/10, random_state =seed)
    
    s = []
    
    for xx in X_train:
        tmp = np.concatenate([um[xx[0]],im[xx[1]]]).ravel()
        s.append(tmp)

    X_train = np.array(s)
    del s

    s = []
    for xx in X_test:
        tmp = np.concatenate([um[xx[0]],im[xx[1]]]).ravel()
        s.append(tmp)
        
    X_test = np.array(s)
    del s
    
    print(X_train.shape,X_test.shape)
    
    clf = xgb.XGBRegressor(
            learning_rate=0.3,
            n_estimators=500,
            min_child_weight=1,
            max_depth=6,
            subsample=0.7,
            colsample_bytree= 0.7,
            reg_alpha=0,
            reg_lambda=0.1,
            gamma=0,
            max_delta_step=0,
            seed=seed,
        )
        
    clf.fit(
        X_train,
        y_train,
        early_stopping_rounds=150,
        eval_set=[(X_train, y_train),(X_test, y_test)],
        eval_metric='rmse',
    )

    y_preds = clf.predict(X_test).ravel()
    score = rmse(y_test,y_preds)
    print('xgb rmse score',score)
    