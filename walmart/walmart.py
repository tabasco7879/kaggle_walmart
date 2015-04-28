import pandas as pd
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import random
from utils import is_numeric
from cost import cost_fun, l_cost_fun, l_cost_fun2, l_g_cost_fun, g_cost_fun, fun_log_error
from walmart2 import load_data2, normalize_store_data, develop_valid_set2, build_target_set, denormalize_store_data
from similarity import l_similarity, logistic_sim_score, g_logistic_sim_score
from sklearn import linear_model
    
def build_model(train, valid, test, l, \
                valid_init=None, alpha_train=1000, alpha_unknown=0.01):    
    train_num, m = train.values.shape
    valid_num, _ = valid.values.shape
    n = train_num + valid_num
    if test is not None:
        test_num, _ = test.values.shape
        n+=test_num
    Y = np.zeros((n, m))
    Y[0:train_num] = train.values
    if valid_init is not None:
        Y[train_num:train_num + valid_num] = valid_init
    Y_hat = np.random.rand(n, m).flatten()
    cost = l_cost_fun(Y_hat, Y, l, train_num, alpha_train, alpha_unknown)
    print 'init cost=', cost    
    Y_hat_bounds = [(0, None)] * len(Y_hat)
    Y_hat, cost, _ = fmin_l_bfgs_b(l_cost_fun, Y_hat, l_g_cost_fun, \
                        args=(Y, l, train_num, alpha_train, alpha_unknown), \
                        bounds=Y_hat_bounds, callback=None)
    print 'optimized cost=', cost
    return Y_hat[0].reshape(Y.shape)

def build_model2(train, valid, test, \
                 store_weather_data, \
                 valid_init=None, theta_init=None, \
                 alpha_train=1000, alpha_unknown=0.01,
                 eps=1e-5, max_iter=100):
    _,fmat = l_similarity(train, valid, test, store_weather_data, normalize=False)
    if (theta_init is None):
        theta = np.random.rand(fmat.shape[1])
    else:
        theta = theta_init
    ntrain, m = train.values.shape
    nvalid, _ = valid.values.shape
    n = ntrain + nvalid
    if test is not None:
        ntest, _ = test.values.shape
        n+=ntest    
    Y = np.zeros((n, m))
    Y[0:train_num] = train.values
    if (valid_init is not None):
        Y[train_num:train_num + valid_num] = valid_init
    Y_hat = np.random.rand(n, m).flatten()
    
    err = 1000.0
    iter = 0
    while (true):        
        l = logistic_sim_score(theta, fmat)
        fval = l_cost_fun(Y_hat, Y, l, ntrain, alpha_train, alpha_unknown)
        print 'init total cost=', fval    
        Y_hat_bounds = [(0, None)] * len(Y_hat)
        Y_hat, fval, _ = fmin_l_bfgs_b(l_cost_fun, Y_hat, l_g_cost_fun, \
                            args=(Y, l, ntrain, alpha_train, alpha_unknown), \
                            bounds=Y_hat_bounds, callback=None)
        Y_hat = Y_hat.reshape(Y.shape)
        print 'total cost=' % fval

        fval = l_cost_fun2(theta, fmat, Y_hat)
        print 'init theta cost=', fval
        theta, fval, _ = fmin_l_bfgs_b(l_cost_fun2, theta, g_logistic_sim_score, \
                            args=(fmat, Y_hat))
        print 'theta cost=', fval

        e1, e2 = eval_model(train, valid, Y_hat)
        iter+=1
        if abs(e1 - err) / err < eps or iter >= max_iter:
            break

    return Y_hat, theta

def eval_model(train, valid, Y_hat):
    train_num, m = train.values.shape
    valid_num, _ = valid.values.shape
    t_error = fun_log_error(Y_hat[:train_num], \
                     train.values, train_num, 1, 0)
    v_error = fun_log_error(Y_hat[train_num:train_num + valid_num], \
                     valid.values, valid_num, 1, 0)
    return (t_error / train_num) ** 0.5, (v_error / valid_num) ** 0.5

def run_model2(store_data_file, store_weather_file, test_data_file):
    """
    use the same model as run_model2 but use the functions from walrmat2
    the model only use the training data that are relevant to reduce computation
    """
    print "start here"

    with open('test_result.csv', 'w') as f:
        f.write('id,units\n')
        f.close()

    store_data, store_weather, test = load_data2(store_data_file, \
          store_weather_file, test_data_file)
    store_data_max = store_data.groupby(level=1).max()

    train, valid = develop_valid_set2(store_data, store_weather, valid_size=0)   
    target_set = build_target_set(train, valid, test, store_weather)
    for n, trn, vld, tst in target_set:
        print "%d, train(%d), valid(%d), test(%d)" % (n, len(trn), len(vld), len(tst))
        nm_trn = normalize_store_data(trn, store_data_max)
        nm_vld = normalize_store_data(vld, store_data_max)
        nm_tst = normalize_store_data(tst, store_data_max)
        l,_ = l_similarity(nm_trn, nm_vld, nm_tst, store_weather)
        v_init = None
        Y_hat2 = None
        for i in range(1):
            Y_hat = build_model(nm_trn, nm_vld, nm_tst, l, v_init)
            #v_init=Y_hat[len(trn):len(trn)+len(vld)]
            Y_hat2 = denormalize_store_data(trn, vld, tst, Y_hat, store_data_max)
            e1, e2 = eval_model(trn, vld, Y_hat2)
            print "error at %d is: train(%f), valid(%f)" % (i, e1, e2)
        write_submission(trn, vld, tst, Y_hat2, 'test_result.csv')

def run_model3(store_data_file, store_weather_file, test_data_file):
    """
    ridge regression
    """
    print "start here"

    with open('test_result.csv', 'w') as f:
        f.write('id,units\n')
        f.close()

    store_data, store_weather, test = load_data2(store_data_file, \
          store_weather_file, test_data_file)
    store_data_max = store_data.groupby(level=1).max()
    train, valid = develop_valid_set2(store_data, store_weather, valid_size=70)    
    target_set = build_target_set(train, valid, test, store_weather)
    for n, trn, vld, tst in target_set:
        print "%d, train(%d), valid(%d), test(%d)" % (n, len(trn), len(vld), len(tst))
        nm_trn = normalize_store_data(trn, store_data_max)
        nm_vld = normalize_store_data(vld, store_data_max)
        nm_tst = normalize_store_data(tst, store_data_max)
        _,m = l_similarity(nm_trn, nm_vld, nm_tst, store_weather)
        item_count = nm_trn.values.shape[1]
        Y_hat = np.zeros((len(nm_trn) + len(nm_vld) + len(nm_tst), item_count))
        X = m[:len(nm_trn)]
        for i in range(item_count):            
            Y = nm_trn.values[:,i]
            clf = linear_model.Ridge(alpha=1.0)
            clf.fit(X, Y)
            Y_hat[:,i] = clf.predict(m)
        Y_hat2 = denormalize_store_data(trn, vld, tst, Y_hat, store_data_max)
        e1, e2 = eval_model(trn, vld, Y_hat2)
        print "error is: train(%f), valid(%f)" % (e1, e2)
        write_submission(trn, vld, tst, Y_hat2, 'test_result.csv')

def write_submission(train, valid, test, Y_hat, test_result_file):
    row = len(train) + len(valid)
    col = Y_hat.shape[1]
    with open(test_result_file, 'a') as f:
        for x in test.index:
            item = test.loc[x]
            fmt = str(x[1]) + '_%s_' + str(x[0])[:10] + ',%f\n'            
            for i in range(col):
                s = fmt % (i + 1, Y_hat[row, i])
                f.write(s)
            row+=1

run_model3('C:/Users/tao.chen/skydrive/working/kaggle_walmart/data/store_train.txt', \
          'C:/Users/tao.chen/skydrive/working/kaggle_walmart/data/store_weather.txt', \
          'C:/Users/tao.chen/skydrive/working/kaggle_walmart/data/test.csv')