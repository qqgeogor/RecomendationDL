import numpy as np
import pandas as pd
import os
import glob
import math
import pickle
import datetime
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

def X_train_generatetor(dim=128,name="X_train.csv"):
    model = models.Word2Vec.load(path+'w2v_%s.mdl'%dim)
    X_train = pd.read_csv(path+name)
    y_train = X_train[score].values
    X_train = X_train[features].values
    while 1:
        for u,i,yy in zip(X_train[:,0],X_train[:,1],y_train):
            s=[ 'u%s'%u,'i%s'%i]
            if s[0] in model.vocab:
                s0 = model[s[0]].ravel()
            else:
                s0 = np.zeros(dim)
                
            if s[0] in model.vocab:
                s1 = model[s[0]].ravel()
            else:
                s1 = np.zeros(dim)

            s = np.concatenate([s0,s1]).flatten()
            # print(s.shape)
            yield (np.array([s]),np.array([yy]))

def X_test_generatetor(dim=128,name="X_test.csv"):
    model = models.Word2Vec.load(path+'w2v_%s.mdl'%dim)
    X_test = pd.read_csv(path+name)
    y_test = X_test[score].values
    X_test = X_test[features].values
    for u,i,yy in zip(X_test[:,0],X_test[:,1],y_train):
        s=[ 'u%s'%u,'i%s'%i]
        if s[0] in model.vocab:
            s0 = model[s[0]].ravel()
        else:
            s0 = np.zeros(dim)
            
        if s[0] in model.vocab:
            s1 = model[s[0]].ravel()
        else:
            s1 = np.zeros(dim)

        s = np.concatenate([s0,s1]).flatten()
        # print(s.shape)
        yield (np.array([s]),np.array([yy]))


def get_test(dim=128,name="test.csv"):
    model = models.Word2Vec.load(path+'w2v_%s.mdl'%dim)
    X_test = pd.read_csv(path+name)
    X_test = X_test[features].values
    d = []
    for u,i,yy in zip(X_test[:,0],X_test[:,1],y_train):
        s=[ 'u%s'%u,'i%s'%i]
        if s[0] in model.vocab:
            s0 = model[s[0]].ravel()
        else:
            s0 = np.zeros(dim)
            
        if s[0] in model.vocab:
            s1 = model[s[0]].ravel()
        else:
            s1 = np.zeros(dim)

        s = np.concatenate([s0,s1]).flatten()
        d.append(s)
    d = np.array(d)
    return d



if __name__ == '__main__':

    use_all=False

    path = "E:\\RSdata\\"
    features = ['uid','iid']
    score = 'score'

    X_train = pd.read_csv(path+"X_train.csv")
    X_test = pd.read_csv(path+"X_test.csv")
    test = pd.read_csv(path+"test.csv")

    input = Input(shape=(dim*2,))

    fc1 = Dense(512)(input)
    fc1 = SReLU()(fc1)
    # fc1 = BatchNormalization(mode=0)(fc1)
    dp1 = Dropout(0.5)(fc1)
    
    fc2 = Dense(128)(dp1)
    fc2 = SReLU()(fc2)
    # fc2 = BatchNormalization(mode=0)(fc2)
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
    load_model = False

    if load_model:
        model.load_weights(path+model_name)

    tr_gen = X_train_generatetor(128,name='X_train.csv')
    te_gen = X_test_generatetor(128,name='X_test.csv')
    test_gen = X_test_generatetor(128,name='test.csv')

    model.fit_generator(
        tr_gen, 
        samples_per_epoch=X_train.shape[0], 
        nb_epoch=nb_epoch, 
        verbose=1, 
        callbacks=[model_checkpoint], 
        validation_data=te_gen, 
        nb_val_samples=X_test.shape[0], 
        class_weight={}, 
        max_q_size=batch_size
        )
    
    # model.fit([u_train,i_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, shuffle=True,
    #                   callbacks=[model_checkpoint],
    #                   validation_data=([u_test,i_test],y_test)
    #                   )
    
    
    test = get_test()
    y_preds = model.predict(test)
    print('y_preds shape',y_preds.shape)
    # d = {'score':y_preds}
    submission = pd.DataFrame()
    submission['score'] = y_preds
    submission.to_csv(path+"submission_mlp.csv",index=False)




