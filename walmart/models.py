import numpy as np
from scipy.optimize import fmin_l_bfgs_b, fmin_bfgs
from similarity import sim, sim_func, l_sim, l_sim_func
from cost import cost_fun, g_cost_fun, cost_fun3, g_cost_fun3, fun_log_error
from utils import normalize

def eval_model(train, valid, Y_hat):
    ntrain, m = train.values.shape
    nvalid, _ = valid.values.shape
    t_error = fun_log_error(Y_hat[:ntrain], \
                     train.values, ntrain, 1, 0)
    v_error = fun_log_error(Y_hat[ntrain:ntrain + nvalid], \
                     valid.values, nvalid, 1, 0)
    return (t_error / ntrain) ** 0.5, (v_error / nvalid) ** 0.5

def build_model3(train, valid, test, \
                 store_weather_data, \
                 valid_init=None, \
                 alpha_train=1000, alpha_unknown=0.01,
                 eps=1e-5, max_iter=100):
    """
    the model uses logistic regression to model similarities between rows.
    iteratively update Y_hat and similarities parameter theta.
    """
    # count the total number of rows
    ntrain, m = train.values.shape

    nvalid=0
    if valid is not None:
        nvalid, _ = valid.values.shape

    ntest=0
    if test is not None:
        ntest, _ = test.values.shape

    n = ntrain + nvalid + ntest

    # init hidden features
    hidden_shape=(n, 20)
    hidden=np.random.rand(n*20)

    # compute similarity matrix
    L, fmat = sim(train, valid, test, store_weather_data, \
                 is_normalize=True, hidden_feature=hidden.reshape(hidden_shape))
    offset=fmat.shape[1]-hidden_shape[1]

    # init Y and Y_hat
    Y = np.zeros((n, m))
    Y[0:ntrain] = train.values
    if (valid_init is not None):
        Y[ntrain:ntrain + nvalid] = valid_init
    Y_shape = (n, m)
    Y_hat = np.random.rand(n*m)
    
    # set up constraint on Y_hat that all are >=0
    Y_hat_bounds = [(0, None)] * len(Y_hat)

    # set up constraint on hidden that all are 1>hidden>=0    
    hidden_bounds = [(0, 1)] * len(hidden)

    err = 1000.0
    iter = 0    
    while (True): # Y_hat is a flatten format
        # compute init total cost
        fval = cost_fun(Y_hat, Y, L, ntrain, alpha_train, alpha_unknown)
        #print 'init total cost=', fval
        
        # run optimiaztion of Y_hat
        Y_hat, fval, _ = fmin_l_bfgs_b(cost_fun, Y_hat, g_cost_fun, \
                            args=(Y, L, ntrain, alpha_train, alpha_unknown), \
                            bounds=Y_hat_bounds, callback=None)
        #print 'optimized total cost=', fval

        # compute init similarity cost        
        fval = cost_fun3(hidden, hidden_shape, fmat, Y_hat, Y_shape)
        #print 'init similarity cost=', fval   

        D=compute_D(Y_hat, Y_shape)
        # run optimiaztion of hidden features
        hidden, fval, _  = fmin_l_bfgs_b(cost_fun3, hidden, g_cost_fun3, \
                            args=(hidden_shape, fmat, Y_hat, Y_shape, D), \
                            bounds=hidden_bounds, callback=None, maxiter=10)
        #print 'optimized similarity cost=', fval
        fmat[:,offset:]=hidden.reshape(hidden_shape)

        iter+=1

        # evaluate erros
        Y_hat=Y_hat.reshape(Y_shape)
        e1, e2 = eval_model(train, valid, Y_hat)
        Y_hat=Y_hat.flatten()
        #print "error at %d is: train(%f), valid(%f)" % (iter, e1, e2)
       
        # stop if reach max iteration or changes is very small
        if abs(e1 - err) / err < eps or \
            iter >= max_iter:
            break
        else:
            err=e1
        
        # compute new similarity matrix        
        L=sim_func(normalize(fmat)[0])

    return Y_hat.reshape(Y_shape)

def compute_D(Y_hat, Y_shape):
    """
    compute (x_i-x_j)^T(x_i-x_j)
    """
    Y_hat=Y_hat.reshape(Y_shape)
    n=Y_hat.shape[0]
    D=np.zeros((n,n))
    for i in range(n):
        d0=(Y_hat-Y_hat[i])**2           
        d0=np.sum(d0, axis=1) # shape=nx1
        D[i]=d0
    return D
