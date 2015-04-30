import numpy as np
import pandas as pd
from feature import compute_feature2
from similarity import l_logistic_sim, sim_func, l_sim_func
from utils import normalize

def l_cost_fun(Y_hat, Y, l, train_num, alpha_train, alpha_unknown):
    Y_hat=Y_hat.reshape(Y.shape)
    return l_fun_sim(Y_hat, l) \
              + fun_sqr_error(Y_hat, Y, train_num, alpha_train, alpha_unknown)

def l_cost_fun2(theta, m, Y_hat, Y):
    Y_hat=Y_hat.reshape(Y.shape)
    l=l_logistic_sim(theta, m)
    return l_fun_sim(Y_hat, l)


def cost_fun(Y_hat, Y, L, train_num, alpha_train, alpha_unknown):
    Y_hat=Y_hat.reshape(Y.shape)
    return fun_sim(Y_hat, L) \
              + fun_sqr_error(Y_hat, Y, train_num, alpha_train, alpha_unknown)

def l_fun_sim(Y_hat, l):
    n=Y_hat.shape[0]
    sim=0
    if len(Y_hat.shape)>1:
        for i in range(n):         
            sim+=np.dot(l(i), Y_hat).dot(Y_hat[i])
    else:
        for i in range(n):         
            sim+=np.dot(l(i), Y_hat)*Y_hat[i]
    return sim

def fun_sim(Y_hat, L):
    return np.trace(Y_hat.T.dot(L).dot(Y_hat))

def fun_log_error(Y_hat, Y, train_num, alpha_train, alpha_unknown):
    log_Y_hat=np.log(Y_hat+1)
    log_Y=np.log(Y+1)
    cost_Y_hat=(log_Y_hat-log_Y)**2
    cost_Y_hat[0:train_num]=cost_Y_hat[0:train_num]*alpha_train
    cost_Y_hat[train_num:]=cost_Y_hat[train_num:]*alpha_unknown
    return np.sum(cost_Y_hat)

def fun_sqr_error(Y_hat, Y, train_num, alpha_train, alpha_unknown):
    cost_Y_hat=(Y_hat-Y)**2
    cost_Y_hat[0:train_num]=cost_Y_hat[0:train_num]*alpha_train
    cost_Y_hat[train_num:]=cost_Y_hat[train_num:]*alpha_unknown
    return np.sum(cost_Y_hat)

def l_g_cost_fun(Y_hat, Y, l, train_num, alpha_train, alpha_unknown):
    Y_hat=Y_hat.reshape(Y.shape)
    g=(l_g_fun_sim(Y_hat, l)+ \
        g_fun_sqr_error(Y_hat, Y, train_num, alpha_train, alpha_unknown))
    return g.flatten()

def g_cost_fun(Y_hat, Y, L, train_num, alpha_train, alpha_unknown):
    Y_hat=Y_hat.reshape(Y.shape)
    g=(g_fun_sim(Y_hat, L)+ \
        g_fun_sqr_error(Y_hat, Y, train_num, alpha_train, alpha_unknown))
    return g.flatten()

def l_g_fun_sim(Y_hat, l):
    n, _ =Y_hat.shape
    g=np.zeros(Y_hat.shape)
    for i in range(n):
        g[i]=np.dot(l(i), Y_hat)
    return 2*g
    
def g_fun_sim(Y_hat, L):    
    return 2*np.dot(L, Y_hat)

def g_fun_log_error(Y_hat, Y, train_num, alpha_train, alpha_unknown):
    """
    Y_hat and Y contain all values including guess for unknown days
    train_num is used to seperate Y_hat
    two alphas are used for training data and unknonw data
    """
    log_Y_hat=np.log(Y_hat+1)
    log_Y=np.log(Y+1)
    g_error=2*(log_Y_hat-log_Y)*(1.0/(Y_hat+1))
    g_error[0:train_num]=g_error[0:train_num]*alpha_train
    g_error[train_num:]=g_error[train_num:]*alpha_unknown
    return g_error

def g_fun_sqr_error(Y_hat, Y, train_num, alpha_train, alpha_unknown):
    g_error=2*(Y_hat-Y)*Y_hat
    g_error[0:train_num]=g_error[0:train_num]*alpha_train
    g_error[train_num:]=g_error[train_num:]*alpha_unknown
    return g_error

def cost_fun3(hidden, hidden_shape, fmat, Y_hat, Y_shape):
    Y_hat=Y_hat.reshape(Y_shape)
    hidden=hidden.reshape(hidden_shape)
    offset=fmat.shape[1]-hidden_shape[1]
    fmat[:,offset:]=hidden
    nm_fmat,_=normalize(fmat)
    l=l_sim_func(nm_fmat)
    return l_fun_sim(Y_hat, l)

def g_cost_fun3(hidden, hidden_shape, fmat, Y_hat, Y_shape):
    Y_hat=Y_hat.reshape(Y_shape)
    hidden=hidden.reshape(hidden_shape)
    offset=fmat.shape[1]-hidden_shape[1]
    fmat[:,offset:]=hidden
    _, nm=normalize(fmat) # shape=nx1
    n= Y_shape[0]
    g=np.zeros(hidden_shape)
    for i in range(n):
        d0=(Y_hat-Y_hat[i])**2
        if len(d0.shape)>1:
            d0=np.sum(d0, axis=1) # shape=nx1
        c0=1/((nm[i]**2)*nm) # shape=nx1      
        c1=fmat[:, offset:] * nm[i] # shape=nxm        
        c2=np.outer(np.dot(fmat,fmat[i])/nm[i], fmat[i, offset:]) # shape=nxm        
        g[i]=np.sum((c1-c2)*(d0*c0)[:, np.newaxis], axis=0)  # shape
    return g.flatten()
        