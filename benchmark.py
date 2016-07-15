import pandas as pd
import scipy.sparse as ssp
import numpy as np
from sklearn.cross_validation import StratifiedKFold,KFold
from ml_metrics import rmse
import mrf

seed = 1024
np.random.seed(seed)


def main():
    path = "E:\\RSdata\\"
    # sort = False

    # if sort:
    #     print 'sort train'
    #     # train = pd.read_csv(path+"train.csv")
    #     # train = train.sort(columns=['time'])
    #     # train.to_csv(path+"train_sorted.csv",index=False)
    #     train = pd.read_csv(path+"train_sorted.csv")
    # else:
    #     train = pd.read_csv(path+"train.csv")

        
    # # test = pd.read_csv(path+"test.csv")

    # skf = StratifiedKFold(train['uid'].values, n_folds=5, shuffle=False, random_state=seed)
    # # skf = KFold(train.shape[0],n_folds=5, shuffle=True, random_state=seed)
    # for ind_tr, ind_te in skf:
    #     X_train = train.iloc[ind_tr]
    #     X_test = train.iloc[ind_te]
    #     break
    # print X_train.head()
    # print X_test.head()

    # X_train.to_csv(path+"X_train.csv",index=False)
    # X_test.to_csv(path+"X_test.csv",index=False)


    X_train = pd.read_csv(path+"X_train.csv")
    X_test = pd.read_csv(path+"X_test.csv")
    test = pd.read_csv(path+"test.csv")
    # print X_train.head
    
    X_train = X_train[:100000]
    X_test = X_test[:100000]
    
    features = ['uid','iid']
    score = 'score'
    y_train = X_train[score].values
    X_train = X_train[features].values

    y_test = X_test[score].values
    X_test = X_test[features].values

    model = mrf.UserMRF()
    model.fit(X_train,y_train)
    # pd.to_pickle(model,path+'usermrf.pkl')
    # model = pd.read_pickle(path+'usermrf.pkl')
    # print('neighbour_user',model.neighbour_user)
    # pd.to_pickle(model,path+'usermrf.pkl')
    # model = pd.read_pickle(path+'usermrf.pkl')
    # model.save_weights()
    # model.load_weights()
    model.loss()


    


if __name__ == '__main__':
    main()