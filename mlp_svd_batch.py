import numpy as np
import pandas as pd
from scipy import sparse as ssp
import os
import glob
import math
import pickle
import time
import datetime
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,LabelBinarizer
from sklearn.utils import resample
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers import Input, Embedding, LSTM, Dense,Flatten, Dropout, merge,Convolution1D,MaxPooling1D,Lambda
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.layers.advanced_activations import PReLU,LeakyReLU,ELU,SReLU
from keras.models import Model
from keras.utils.visualize_util import plot
from gensim import models
from ml_metrics import rmse
seed = 1024
np.random.seed(seed)
dim = 128

def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size/float(batch_size)))
    return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch)]


def convert_vector(X_train,y_train,batch_start,batch_end,model):
    x = X_train[batch_start:batch_end]
    y = y_train[batch_start:batch_end]
    s = []
    for u,i in zip(x[:,0],x[:,1]):
        s=[ 'u%s'%u,'i%s'%i]
        if s[0] in model.vocab:
            s0 = model[s[0]].ravel()
        else:
            s0 = np.zeros(dim)
            
        if s[0] in model.vocab:
            s1 = model[s[0]].ravel()
        else:
            s1 = np.zeros(dim)
        s.append(np.concatenate([s0,s1]).flatten())

    s = np.array(s)
    return (s,y)

def X_train_generatetor(dim=128,um=None,im=None,batch_size=128,name="X_train.csv",shuffle=True):
    # model = models.Word2Vec.load(path+'w2v_%s.mdl'%dim)
    data = pd.read_csv(path+name)
    # batches = make_batches(data.shape[0],batch_size=batch_size)
    index_array = np.arange(data.shape[0])
    if shuffle:
        np.random.shuffle(index_array)
    batches = make_batches(data.shape[0], batch_size)

    for batch_index, (batch_start, batch_end) in enumerate(batches):
        batch_ids = index_array[batch_start:batch_end]
        X_train = data.iloc[batch_ids]

        y_tr = X_train[score].values
        x = X_train[features].values
        x[:,0] = user_le.transform(x[:,0])
        x[:,1] = item_le.transform(x[:,1])
        x_tr = []
        for u,i in zip(x[:,0],x[:,1]):
            tmp = np.concatenate([um[u],im[i]]).ravel()
            x_tr.append(tmp)
        x_tr = np.array(x_tr)
        yield (x_tr,y_tr)
        
def get_test(dim=128,um=None,im=None,name="test.csv"):
    model = models.Word2Vec.load(path+'w2v_%s.mdl'%dim)
    x = pd.read_csv(path+name)
    x = X_test[features].values

    x[:,0] = user_le.transform(x[:,0])
    x[:,1] = item_le.transform(x[:,1])
    x_train = []
    for u,i in zip(x[:,0],x[:,1]):
        tmp = np.concatenate([um[u],im[i]]).ravel()
        x_train.append(tmp)
    x_train = np.array(x_train)
    return x_train

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
    um_svd = TruncatedSVD(n_components=dim)
    um = um_svd.fit_transform(um)
    print 'um',um.shape
    im_svd = TruncatedSVD(n_components=dim)
    im = im_svd.fit_transform(im)
    print 'im',im.shape


    input = Input(shape=(dim*2,))

    fc1 = Dense(512)(input)
    fc1 = SReLU()(fc1)
    fc1 = BatchNormalization(mode=0)(fc1)
    dp1 = Dropout(0.5)(fc1)

    fc2 = Dense(128)(dp1)
    fc2 = SReLU()(fc2)
    fc2 = BatchNormalization(mode=0)(fc2)
    dp2 = Dropout(0.2)(fc2)

    output = Dense(1)(dp2)

    model = Model(input=input, output=output)
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer='nadam',
              loss='mse')

    model_name = 'mlp_w2v.hdf5'
    model_checkpoint = ModelCheckpoint(path+model_name, monitor='val_loss', save_best_only=True)
    plot(model, to_file=path+'%s.png'%model_name.replace('.hdf5',''),show_shapes=True)


    nb_epoch = 10
    batch_size = 1024*6
    load_model = True
    
    if load_model:
        print('Load Model')
        model.load_weights(path+model_name)

        y_tr_preds = []
        y_te_preds = []
        
        tr_gen = X_train_generatetor(128,um,im,batch_size=batch_size,name='X_train.csv')
        te_gen = X_train_generatetor(128,um,im,batch_size=batch_size,name='X_test.csv')

        for X_tr,y_tr in tr_gen:
            p_tr = model.predict_on_batch(X_tr)
            y_tr_preds.append(p_tr)

        for X_te,y_te in te_gen:
            p_te = model.predict_on_batch(X_te)
            y_te_preds.append(p_te)

        y_tr_preds = np.concatenate(y_tr_preds).ravel()
        y_te_preds = np.concatenate(y_te_preds).ravel()

        train_score = rmse(y_train,y_tr_preds)
        print('rmse train',train_score)
        test_score = rmse(y_test,y_te_preds)
        print('rmse test',test_score)

    print('Start Training')
    for epoch in range(nb_epoch):
        tr_gen = X_train_generatetor(128,um,im,batch_size=batch_size,name='X_train.csv')
        te_gen = X_train_generatetor(128,um,im,batch_size=batch_size,name='X_test.csv')
        # train
        # y_tr_preds = []
        # y_te_preds = []

        start_time = time.time()
        for X_tr,y_tr in tr_gen:
            model.train_on_batch(X_tr,y_tr)

        model.save_weights(path+model_name,overwrite=True)

        # evaluate
        y_tr_preds = []
        y_te_preds = []
        
        tr_gen = X_train_generatetor(128,um,im,batch_size=batch_size,name='X_train.csv')
        te_gen = X_train_generatetor(128,um,im,batch_size=batch_size,name='X_test.csv')
        
        for X_tr,y_tr in tr_gen:
            p_tr = model.predict_on_batch(X_tr)
            y_tr_preds.append(p_tr)

        for X_te,y_te in te_gen:
            p_te = model.predict_on_batch(X_te)
            y_te_preds.append(p_te)
            
        y_tr_preds = np.concatenate(y_tr_preds).ravel()
        y_te_preds = np.concatenate(y_te_preds).ravel()

        train_score = rmse(y_train,y_tr_preds)
        test_score = rmse(y_test,y_te_preds)

        print("Epoch {} of {} took {:.3f}s".format(
                    epoch + 1, nb_epoch, time.time() - start_time))
        print('rmse train',train_score)
        print('rmse test',test_score)