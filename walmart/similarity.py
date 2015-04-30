import numpy as np
import pandas as pd
from feature import compute_feature2
from utils import normalize

def sim(train, valid, test, store_weather_data, is_normalize=True, hidden_feature=None):
    store_train=store_weather_data.loc[train.index]
    store_valid=store_weather_data.loc[valid.index]
    store_data=pd.concat([store_train, store_valid])
    if (test is not None):
        store_test=store_weather_data.loc[test.index]
        store_data=pd.concat([store_data, store_test])
    df=store_weather_data.loc[store_data.index]
    fmat=compute_feature2(df, store_data, hidden_feature)
    L=None
    if is_normalize:
        nm_fmat,_=normalize(fmat)
        L = sim_func(nm_fmat)
    else:
        L = sim_func(fmat)
    return L, fmat

def sim_func(fmat):
    G=np.dot(fmat, fmat.T)
    D=np.sum(G, axis=0)
    L=np.eye(G.shape[0])*D-G
    return L

def l_sim(train, valid, test, store_weather_data, is_normalize=True, hidden_feature=None):    
    store_train=store_weather_data.loc[train.index]
    store_valid=store_weather_data.loc[valid.index]
    store_data=pd.concat([store_train, store_valid])
    if (test is not None):
        store_test=store_weather_data.loc[test.index]
        store_data=pd.concat([store_data, store_test])
    df=store_weather_data.loc[store_data.index]    
    fmat=compute_feature2(df, store_data, hidden_feature)
    l=None
    if is_normalize:
        nm_fmat,_=normalize(fmat)
        l = l_sim_func(nm_fmat)
    else:
        l = l_sim_func(fmat)
    return l, fmat

def l_sim_func(fmat):
    def l(i):        
        g=np.dot(fmat, fmat[i])
        d=np.sum(g)
        g=-g
        g[i]=d+g[i]
        return g
    return l

def l_logistic_sim(theta, m):
    """
    i: index of the row
    m: feature matrix
    theta: parameter of logistic function
    """
    def l(i):
        psi=m*m[i]        
        x=np.dot(psi, theta)
        a=np.exp(x)
        g=a/(1+a)
        d=np.sum(g)
        g=-g
        g[i]=d+g[i]        
        return g
    return l

def g_logistic_sim(theta, m, Y_hat, Y):
    """
    Y_hat: current Y_hat
    m: feature matrix
    theta: current parameter of logistic function
    """
    Y_hat=Y_hat.reshape(Y.shape)
    a,b=Y_hat.shape
    g_theta=np.zeros(theta.shape)
    for i in range(a):
        # \sum_j (x_i-x_j)^T(x_i-x_j)
        D=Y_hat-Y_hat[i]
        g_sim=np.sum(D*D, axis=1) # nx1        
        # g(theta^T psi)(1-g(theta^T psi)) psi_i
        psi=m*m[i] # nxm
        x=np.dot(psi, theta) # nx1
        a=np.exp(x) # nx1       
        g_lgs=psi * (a/((1+a)**2))[:, np.newaxis] # nxm        
        g_theta+=np.sum(g_lgs * g_sim[:, np.newaxis], axis=0) # mx1
    return g_theta