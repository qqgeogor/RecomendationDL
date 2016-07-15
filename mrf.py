import numpy as np
import scipy as sp
from scipy import sparse as ssp
import pandas as pd
from datetime import datetime
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from multiprocessing import Pool
import networkx as nx
# import theano

class MRF(BaseEstimator):

    def __init__(self):
        self.G = nx.Graph()
        pass


    def neighbour(self,N,u,i,neighbour_type):
        if neighbour_type=="item":
            pass
        if neighbour_type=="user":
            pass

    def fit(self,X,y):
        self.user_le = LabelEncoder()
        self.item_le = LabelEncoder()
        X[:,0] = self.user_le.fit_transform(X[:,0])
        X[:,1] = self.item_le.fit_transform(X[:,1])


        self.num_u = len(np.unique(X[:,0]))
        self.num_i = len(np.unique(X[:,1]))

        self.alpha = np.zeros(self.num_i)
        self.beta = np.zeros(self.num_u)
        self.omega = ssp.dok_matrix((self.num_i,self.num_i),dtype=np.float32)
        self.w = ssp.dok_matrix((self.num_u,self.num_u),dtype=np.float32)
        

        

        self.R_u_i = ssp.csr_matrix((y,(X[:,0],X[:,1])), dtype=np.int8)


        # self.R_u_i = self.R_u_i.todok()
        # print self.R_u_i
        # for u,i,yy in zip(X[:,0],X[:,1],y):
        #     self.G.add_edge('u%s'%u,'i%s'%i, weight=yy)
            # print('nodes',self.G.nodes())
            # print('edges',self.G.edges())

        # ratings = np.unique(y)
        # for u,i,yy in zip(X[:,0],X[:,1],y):
        #     self.R_u_i[u,i] = yy

        pass

    def predit(self,X):

        pass


    def loss(self,R_u_i):
        E = 0
        for u in range(self.num_u):
            for i in range(self.num_i):
                r_u_i = R_u_i[u][i]
                phi = self.phi_ui(r_u_i,i,u)
                phi = np.log(phi)

                neighbour_item = self.neighbour(N,u,i,neighbour_type="item")
                psi = 0
                for j,i in neighbour_item:
                    r_u_j = R_u_i[u][j]
                    psi += self.psi_ij(r_u_i,r_u_j,i,j)

                neighbour_user = self.neighbour(N,u,i,neighbour_type="user")
                varphi = 0
                for u,v in neighbour_user:
                    r_v_i = R_u_i[v][i]
                    varphi += self.varphi_uv(r_u_i,r_v_i,u,v)

            E+=-(phi+psi+varphi)
        return E


    def predict_prob(self,X):
        Z = 1.
        prob = np.exp(-self.loss(X))
        prob /=Z
        return prob


    def phi_ui(self,r_u_i,u,i):
        phi = -(r_u_i-self.alpha[i]-self.beta[u])*(r_u_i-self.alpha[i]-self.beta[u])
        phi /=2.0
        phi = np.exp(phi)
        return phi

    def psi_ij(self,r_u_i,r_u_j,i,j):
        # print('psi_ij',self.omega[i,j]*r_u_i*r_u_j)
        psi = np.exp(self.omega[i,j]*r_u_i*r_u_j)
        return psi


    def varphi_uv(self,r_u_i,r_v_i,u,v):
        varphi = np.exp(self.w[u,v]*r_u_i*r_v_i)
        return varphi



class UserMRF(MRF):

    def neighbours(self,target_u,target_i):
        # tmp_u = self.X[0][0]
        # tmp_i = self.X[0][1]
        neighbour_user = dict()
        neighbour_user[target_u]=[]
        t1 = datetime.now()
        # print('neighbours starts','t1',t1)
        for row,yy in zip(self.X,self.y):
            u,i,r_u_i = row[0],row[1],yy
            if u!=target_u:
                # tmp_u = u
                # neighbour_user[u]=[(i,yy)]
                continue
                # print('N_%s'%(u-1),self.neighbour_user[u-1])
            else:
                if target_i!=i:
                    neighbour_user[u].append((i,yy))
                else:
                    continue

            # print('u:%s,i:%s,yy:%s'%(u,i,yy))
        t2 = datetime.now()
        # print('neighbours','t2-t1',(t2-t1).total_seconds())
        return neighbour_user



    def neighbour_all(self):
        tmp_u = self.X[0][0]
        tmp_i = self.X[0][1]
        self.neighbour_user = dict()
        self.neighbour_user[tmp_u]=dict(start=0)
        t1 = datetime.now()
        # print('neighbours starts','t1',t1)
        count=0
        for row,yy in zip(self.X,self.y):
            u,i,r_u_i = row[0],row[1],yy
            if u!=tmp_u:
                self.neighbour_user[tmp_u]['end']=count
                tmp_u = u
                self.neighbour_user[tmp_u]=dict(start=count)
            count+=1

        self.neighbour_user[tmp_u]['end']=count+1
        t2 = datetime.now()
        # print('neighbours','t2-t1',(t2-t1).total_seconds())
        # return neighbour_user


    def fit(self,X,y):

        self.user_le = LabelEncoder()
        self.item_le = LabelEncoder()
        X[:,0] = self.user_le.fit_transform(X[:,0])
        X[:,1] = self.item_le.fit_transform(X[:,1])
        self.X = X
        self.y = y

        self.num_u = len(np.unique(X[:,0]))
        self.num_i = len(np.unique(X[:,1]))

        print("number of user:",self.num_u)
        print("number of item:",self.num_i)

        self.alpha = np.zeros(self.num_i)
        self.beta = np.zeros(self.num_u)
        self.omega = ssp.dok_matrix((self.num_i,self.num_i),dtype=np.float32)

        # self.varw = defaultdict(dict)

        self.R_u_i = ssp.csr_matrix((y,(X[:,0],X[:,1])), dtype=np.int8)
        self.neighbour_all()

    def save_weights(self,path):
        pass

    def load_weights(self,path):
        pass

    def loss(self):
        E = 0
        phi_sum = 0
        psi_sum = 0
        t1 = datetime.now()
        print('loss starts','t1',t1)
        for row,yy in zip(self.X,self.y):
            u,i,r_u_i = row[0],row[1],yy
            phi = self.phi_ui(r_u_i,u,i)
            phi = np.log(phi)
            phi_sum += phi
            # get neighbor

            N = self.neighbour_user[u] 
            start = N['start'] 
            end = N['end'] 
            R_N = self.X[start:end] 

            
            for row in R_N:
                j,r_u_j = row[0],row[1]
                psi = self.psi_ij(r_u_i,r_u_j,i,j)
                psi_sum+=psi

            # neighbour_user = self.neighbours(u,i)
            # print('neighbour_user',self.neighbour_user)

        t2 = datetime.now()
        print('phi_sum',phi_sum,'psi_sum',psi_sum,'t2-t1',(t2-t1).total_seconds())
        
        return E


class ItemMRF(MRF):
    
    def neighbours(self,target_u,target_i):
        # tmp_u = self.X[0][0]
        # tmp_i = self.X[0][1]
        neighbour_user = dict()
        neighbour_user[target_u]=[]
        t1 = datetime.now()
        # print('neighbours starts','t1',t1)
        for row,yy in zip(self.X,self.y):
            u,i,r_u_i = row[0],row[1],yy
            if u!=target_u:
                # tmp_u = u
                # neighbour_user[u]=[(i,yy)]
                continue
                # print('N_%s'%(u-1),self.neighbour_user[u-1])
            else:
                if target_i!=i:
                    neighbour_user[u].append((i,yy))
                else:
                    continue

            # print('u:%s,i:%s,yy:%s'%(u,i,yy))
        t2 = datetime.now()
        # print('neighbours','t2-t1',(t2-t1).total_seconds())
        return neighbour_user



    def neighbour_all(self):
        tmp_u = self.X[0][0]
        tmp_i = self.X[0][1]
        self.neighbour_user = dict()
        self.neighbour_user[tmp_u]=dict(start=0)
        t1 = datetime.now()
        # print('neighbours starts','t1',t1)
        count=0
        for row,yy in zip(self.X,self.y):
            u,i,r_u_i = row[0],row[1],yy
            if u!=tmp_u:
                self.neighbour_user[tmp_u]['end']=count
                tmp_u = u
                self.neighbour_user[tmp_u]=dict(start=count)
            count+=1

        self.neighbour_user[tmp_u]['end']=count+1
        t2 = datetime.now()
        # print('neighbours','t2-t1',(t2-t1).total_seconds())
        # return neighbour_user


    def fit(self,X,y):

        self.user_le = LabelEncoder()
        self.item_le = LabelEncoder()
        X[:,0] = self.user_le.fit_transform(X[:,0])
        X[:,1] = self.item_le.fit_transform(X[:,1])
        self.X = X
        self.y = y

        self.num_u = len(np.unique(X[:,0]))
        self.num_i = len(np.unique(X[:,1]))

        print("number of user:",self.num_u)
        print("number of item:",self.num_i)

        self.alpha = np.zeros(self.num_i)
        self.beta = np.zeros(self.num_u)
        self.omega = ssp.dok_matrix((self.num_i,self.num_i),dtype=np.float32)

        # self.varw = defaultdict(dict)

        self.R_u_i = ssp.csr_matrix((y,(X[:,0],X[:,1])), dtype=np.int8)
        self.neighbour_all()

    def save_weights(self,path):
        pass

    def load_weights(self,path):
        pass

    def loss(self):
        E = 0
        phi_sum = 0
        psi_sum = 0
        t1 = datetime.now()
        print('loss starts','t1',t1)
        for row,yy in zip(self.X,self.y):
            u,i,r_u_i = row[0],row[1],yy
            phi = self.phi_ui(r_u_i,u,i)
            phi = np.log(phi)
            phi_sum += phi
            # get neighbor

            N = self.neighbour_user[u] 
            start = N['start'] 
            end = N['end'] 
            R_N = self.X[start:end] 

            
            for row in R_N:
                j,r_u_j = row[0],row[1]
                psi = self.psi_ij(r_u_i,r_u_j,i,j)
                psi_sum+=psi

            # neighbour_user = self.neighbours(u,i)
            # print('neighbour_user',self.neighbour_user)

        t2 = datetime.now()
        print('phi_sum',phi_sum,'psi_sum',psi_sum,'t2-t1',(t2-t1).total_seconds())
        
        return E
