import numpy as np
import pandas as pd
from feature import compute_feature2

def similarity(train, valid, test, store_weather_data):
    store_train=store_weather_data.loc[train.index]
    store_valid=store_weather_data.loc[valid.index]
    store_data=pd.concat([store_train, store_valid])
    if (test is not None):
        store_test=store_weather_data.loc[test.index]
        store_data=pd.concat([store_data, store_test])
    df=store_weather_data.loc[store_data.index]
    m=compute_feature2(df, store_data)   
    G=np.dot(m, m.T)
    D=np.sum(G, axis=0)
    L=np.eye(G.shape[0])*D-G
    return L, m

def l_similarity(train, valid, test, store_weather_data, normalize=True):    
    store_train=store_weather_data.loc[train.index]
    store_valid=store_weather_data.loc[valid.index]
    store_data=pd.concat([store_train, store_valid])
    if (test is not None):
        store_test=store_weather_data.loc[test.index]
        store_data=pd.concat([store_data, store_test])
    df=store_weather_data.loc[store_data.index]    
    m=compute_feature2(df, store_data, normalize)
    def l(i):
        g=np.dot(m, m[i])
        d=np.sum(g)
        g=-g
        g[i]=d+g[i]
        return g
    return l, m

def logistic_sim_score(theta, m):
    """
    i: index of the row
    m: feature matrix
    theta: parameter of logistic function
    """
    def l(i):
        psi=m*m[i]
        x=np.dot(psi, theta)
        a=np.exp(x)
        return a/(1+a)
    return l

def g_logistic_sim_score(theta, m, Y_hat):
    """
    Y_hat: current Y_hat
    m: feature matrix
    theta: current parameter of logistic function
    """
    a,b=Y_hat.shape
    g_theta=np.zeros(theta.shape)
    for i in range(a):
        # \sum_j (x_i-x_j)^T(x_i-x_j)
        D=Y_hat-Y_hat[i]
        g_sim=np.sum(D*D, axis=1) # nx1
        # g(theta^T psi)(1-g(theta^T psi)) f
        psi=m*m[i] # nxm
        x=np.dot(psi, theta) # nx1
        a=np.exp(x)
        g_lgs=a/((1+a)**2)[:,np.newaxis] * psi # nxm
        g_theta+=np.sum(g_sim[:, np.newaxis] * g_lgs, axis=0) # mx1
    return g_theta/2