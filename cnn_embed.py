import numpy as np
import pandas as pd
import os
import glob
import math
import pickle
import datetime
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers import Input, Embedding, LSTM, Dense,Flatten, Dropout, merge,Convolution1D,MaxPooling1D,Lambda
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.layers.advanced_activations import PReLU,LeakyReLU,ELU,SReLU
from keras.models import Model
from keras.utils.visualize_util import plot
from ml_metrics import rmse
seed = 1024
np.random.seed(seed)

if __name__ == '__main__':

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

    user_le = LabelEncoder()
    item_le = LabelEncoder()

    user_le.fit(X[:,0])
    item_le.fit(X[:,1])

    X_train[:,0] = user_le.transform(X_train[:,0])
    X_train[:,1] = item_le.transform(X_train[:,1])

    X_test[:,0] = user_le.transform(X_test[:,0])
    X_test[:,1] = item_le.transform(X_test[:,1])

    test[:,0] = user_le.transform(test[:,0])
    test[:,1] = item_le.transform(test[:,1])

    u_train,i_train = X_train[:,0],X_train[:,1]
    u_test,i_test = X_test[:,0],X_test[:,1]



    num_u = len(np.unique(X[:,0]))
    num_i = len(np.unique(X[:,1]))

    del X

    u_train = np.expand_dims(u_train,1)
    i_train = np.expand_dims(i_train,1)
    u_test = np.expand_dims(u_test,1)
    i_test = np.expand_dims(i_test,1)

    uinput = Input(shape=(1,), dtype='int32')
    iinput = Input(shape=(1,), dtype='int32')

    embed_u = Embedding(
                    output_dim=128, 
                    input_dim=num_u,
                    # dropout=0.0
                    input_length=1,
                    )(uinput)
    embed_i = Embedding(
                    output_dim=128, 
                    input_dim=num_i,
                    # dropout=0.0
                    input_length=1,
                    )(iinput)

    conv_1u = Convolution1D(nb_filter=128,
                        filter_length=3,
                        border_mode='same',
                        # activation='relu',
                        subsample_length=1)(embed_u)
    conv_1u = SReLU()(conv_1u)
    conv_1i = Convolution1D(nb_filter=128,
                        filter_length=3,
                        border_mode='same',
                        # activation='relu',
                        subsample_length=1)(embed_i)
    conv_1i = SReLU()(conv_1i)
    # pool_1u = MaxPooling1D(pool_length=2)(conv_1u)
    # pool_1i = MaxPooling1D(pool_length=2)(conv_1i)


    # pool_1u = MaxPooling1D(pool_length=2)(conv_1u)
    # pool_1i = MaxPooling1D(pool_length=2)(conv_1i)

    # flatten_1u = Flatten()(pool_1u)
    # flatten_1i = Flatten()(pool_1i)

    # merge_ui =  merge([flatten_1u,flatten_1i],mode="concat")

    
    def max_1d(X):
        return K.max(X, axis=1)

    lambda_1u = Lambda(max_1d, output_shape=(128,))(conv_1u)
    lambda_1i = Lambda(max_1d, output_shape=(128,))(conv_1i)

    merge_ui =  merge([lambda_1u,lambda_1i],mode="concat")

    fc1 = Dense(512)(merge_ui)
    fc1 = SReLU()(fc1)
    # fc1 = BatchNormalization(mode=2)(fc1)
    dp1 = Dropout(0.5)(fc1)

    fc2 = Dense(64)(dp1)
    fc2 = SReLU()(fc2)
    # fc2 = BatchNormalization(mode=2)(fc2)
    dp2 = Dropout(0.2)(fc2)

    output = Dense(1)(dp2)

    model = Model(input=[uinput, iinput], output=output)
    sgd = SGD(lr=0.007, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,
              loss='mse')
    
    model_name = 'cnn_len_1.hdf5'
    model_checkpoint = ModelCheckpoint(path+model_name, monitor='val_loss', save_best_only=True)
    plot(model, to_file=path+'%s.png'%model_name.replace('.hdf5',''),show_shapes=True)
    
    
    nb_epoch = 20
    batch_size = 1024*6
    load_model = True
    
    if load_model:
        model.load_weights(path+model_name)
        
        
    model.fit([u_train,i_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, shuffle=True,
                      callbacks=[model_checkpoint],
                      validation_data=([u_test,i_test],y_test)
                      )
   
    
    
    u_test_t,i_test_t = test[:,0],test[:,1]
    y_preds = model.predict([u_test_t,i_test_t]).ravel()
    print('y_preds shape',y_preds.shape)
    # d = {'score':y_preds}
    submission = pd.DataFrame()
    submission['score'] = y_preds
    submission.to_csv(path+"submission_mlp.csv",index=False)
    
    
    # 0.83557
    y_preds = model.predict([u_test,i_test]).ravel()
    score = rmse(y_test,y_preds)
    print('cnn rmse score',score)