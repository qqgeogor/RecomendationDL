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
from ml_metrics import rmse
seed = 1024
np.random.seed(seed)


def build_model():
    uinput = Input(shape=(1,), dtype='int32')
    iinput = Input(shape=(1,), dtype='int32')

    embed_u = Embedding(
                    num_u,
                    128,
                    # dropout=0.2,
                    input_length=1
                    )(uinput)

    embed_i = Embedding(num_i,
                    128,
                    # dropout=0.2,
                    input_length=1
                    )(iinput)

    flatten_u = Flatten()(embed_u)
    flatten_i = Flatten()(embed_i)


    merge_ui =  merge([flatten_u,flatten_i],mode="concat")

    fc1 = Dense(512)(merge_ui)
    fc1 = SReLU()(fc1)
    # fc1 = BatchNormalization(mode=2)(fc1)
    dp1 = Dropout(0.5)(fc1)

    fc2 = Dense(128)(dp1)
    fc2 = SReLU()(fc2)
    # fc2 = BatchNormalization(mode=2)(fc2)
    dp2 = Dropout(0.2)(fc2)

    output = Dense(1)(dp2)

    model = Model(input=[uinput, iinput], output=output)
    sgd = SGD(lr=0.5, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,
              loss='mse')
    
    return model
    
    
def make_mf_regression(X ,y, qid, X_test, n_round=5,batch_size=1024*6,nb_epoch=10):
    
    u,i = X
    n = u.shape[0]
    '''
    Fit metafeature by @clf and get prediction for test. Assumed that @clf -- regressor
    '''
    mf_tr = np.zeros(u.shape[0])
    mf_te = np.zeros(X_test[:,0].shape[0])
    for i in range(n_round):
        skf = KFold(n, n_folds=2, shuffle=True, random_state=42+i*1000)
        for ind_tr, ind_te in skf:
            clf = build_model()

            u_tr = u[ind_tr]
            u_te = u[ind_te]

            i_tr = i[ind_tr]
            i_te = i[ind_te]
            
            y_tr = y[ind_tr]
            y_te = y[ind_te]
            
            clf.fit([u_t,i_tr], y_tr, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, shuffle=True,
                      validation_data=([u_t,i_te],y_te)
                      )
            mf_tr[ind_te] += clf.predict(X_te).ravel()
            mf_te += clf.predict(X_test).ravel()*0.5
            y_pred = np.clip(clf.predict(X_te).ravel(),1,3)
            score = rmse(y_te, y_pred)
            print('round',i,'finished')
            # print 'pred[{}] score:{}'.format(i, score)
    return (mf_tr / n_round, mf_te / n_round)


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


    user_le = LabelEncoder()
    item_le = LabelEncoder()

    user_le.fit(X[:,0])
    item_le.fit(X[:,1])


    train = np.concatenate([X_train,X_test])
    y = np.concatenate([y_train,y_test])

    train[:,0] = user_le.transform(train[:,0])
    train[:,1] = item_le.transform(train[:,1])


    X_train[:,0] = user_le.transform(X_train[:,0])
    X_train[:,1] = item_le.transform(X_train[:,1])

    if use_all:
        X = np.concatenate([X_train,X_test])
        y_train = np.concatenate([y_train,y_test])
        X_train[:,0] = user_le.transform(X[:,0])
        X_train[:,1] = item_le.transform(X[:,1])

    X_test[:,0] = user_le.transform(X_test[:,0])
    X_test[:,1] = item_le.transform(X_test[:,1])

    test[:,0] = user_le.transform(test[:,0])
    test[:,1] = item_le.transform(test[:,1])

    u_train,i_train = X_train[:,0],X_train[:,1]
    u_test,i_test = X_test[:,0],X_test[:,1]

    # meta features

    mf = make_mf_regression((train[:,0],train[:,1]) ,y, qid, (test[:,0],test[:,1]), n_round=5,batch_size=1024*6,nb_epoch=10)
    pd.to_pickle(mf,path+'mf.pkl')


    # num_u = len(np.unique(X[:,0]))
    # num_i = len(np.unique(X[:,1]))

    # del X

    # u_train = np.expand_dims(u_train,1)
    # i_train = np.expand_dims(i_train,1)
    # u_test = np.expand_dims(u_test,1)
    # i_test = np.expand_dims(i_test,1)

    # model = build_model()

    # model_name = 'mlp.hdf5'
    # model_checkpoint = ModelCheckpoint(path+model_name, monitor='val_loss', save_best_only=True)
    # plot(model, to_file=path+'%s.png'%model_name.replace('.hdf5',''),show_shapes=True)


    # nb_epoch = 10
    # batch_size = 1024*6
    # load_model = True

    # if load_model:
    #     model.load_weights(path+model_name)

        
    # model.fit([u_train,i_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, shuffle=True,
    #                   callbacks=[model_checkpoint],
    #                   validation_data=([u_test,i_test],y_test)
    #                   )
    
    
    
    # u_test_t,i_test_t = test[:,0],test[:,1]
    # y_preds = model.predict([u_test_t,i_test_t]).ravel()
    # print('y_preds shape',y_preds.shape)
    # # d = {'score':y_preds}
    # submission = pd.DataFrame()
    # submission['score'] = y_preds
    # submission.to_csv(path+"submission_mlp.csv",index=False)


    # # 0.83557
    # # y_preds = model.predict([u_test,i_test]).ravel()
    # # score = rmse(y_test,y_preds)
    # # print('mlp rmse score',score)


    # inputs = [uinput,iinput,K.learning_phase()]
    # outputs = [fc2]
    # # print inputs,outputs
    # func = K.function(inputs, outputs)


    # print('X_train before shape',X_train.shape)
    # i = 0
    # s = []
    # while i*batch_size<X_train.shape[0]:
    #     tmp = func([u_train[i*batch_size:(i+1)*batch_size],i_train[i*batch_size:(i+1)*batch_size],0])[0]
    #     s.append(tmp)
    # X_train = np.vstack(s)
    # print('X_train after shape',X_train.shape)
    
    
    # i = 0
    # s = []
    # while i*batch_size<X_train.shape[0]:
    #     tmp = func([u_test[i*batch_size:(i+1)*batch_size],i_test[i*batch_size:(i+1)*batch_size],0])[0]
    #     s.append(tmp)
    # X_test = np.vstack(s)
    
    # # X_test = func([u_test,i_test,0])[0]
    
    
    # rf = RandomForestRegressor(n_estimators = 100,max_depth=12,n_jobs=7,random_state=seed)
    # rf.fit(X_train,y_train)
    # y_preds = rf.predict(X_test).ravel()
    # score = rmse(y_test,y_preds)
    # print('rf rmse score',score)

