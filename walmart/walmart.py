import pandas as pd
import numpy as np
from scipy.optimize import fmin_l_bfgs_b, fmin_bfgs
import random
from utils import is_numeric
from cost import cost_fun, l_cost_fun, l_cost_fun2, l_g_cost_fun, g_cost_fun, fun_log_error, l_fun_sim, fun_sim
from walmart2 import load_data2, normalize_store_data, develop_valid_set2, build_target_set, build_target_set2, denormalize_store_data
from similarity import sim, l_sim, l_logistic_sim, g_logistic_sim
from sklearn import linear_model
from models import eval_model, build_model3
    
def build_model(train, valid, test, l, \
                valid_init=None, alpha_train=1000, alpha_unknown=0.01):
    # count the total number of rows
    ntrain, m = train.values.shape

    nvalid=0
    if valid is not None:
        nvalid, _ = valid.values.shape

    ntest=0
    if test is not None:
        ntest, _ = test.values.shape

    n = ntrain + nvalid + ntest

    # init Y
    Y = np.zeros((n, m))
    Y[0:ntrain] = train.values

    if valid_init is not None:
        Y[ntrain:ntrain + nvalid] = valid_init

    # randomly init Y_hat
    Y_hat = np.random.rand(n, m).flatten()    

    # compute init cost
    cost = l_cost_fun(Y_hat, Y, l, ntrain, alpha_train, alpha_unknown)
    print 'init cost=', cost

    # set up constraint on Y_hat that all are >=0
    Y_hat_bounds = [(0, None)] * len(Y_hat)

    # run optimization
    Y_hat, cost, _ = fmin_l_bfgs_b(l_cost_fun, Y_hat, l_g_cost_fun, \
                        args=(Y, l, ntrain, alpha_train, alpha_unknown), \
                        bounds=Y_hat_bounds)
    print 'optimized cost=', cost
    
    # reformat Y_hat to matrix as optimziation works on vector format
    return Y_hat.reshape(Y.shape)

def build_model2(train, valid, test, \
                 store_weather_data, \
                 valid_init=None, theta_init=None, \
                 alpha_train=1000, alpha_unknown=0.01,
                 eps=1e-5, max_iter=100):
    """
    the model uses logistic regression to model similarities between rows.
    iteratively update Y_hat and similarities parameter theta.
    """
    # compute similarity matrix without normalization
    _,fmat = sim(train, valid, test, store_weather_data, normalize=False)
    
    # feature value ranges from 1 to -1 and reduces 0 when doing inner product
    fmat[fmat<=0]=-1

    # use similarity matrix as init
    L,_ = sim(train, valid, test, store_weather_data)

    # init theta
    if (theta_init is None):
        theta = np.random.rand(fmat.shape[1])*10
    else:
        theta = theta_init
    
    # count the total number of rows
    ntrain, m = train.values.shape

    nvalid=0
    if valid is not None:
        nvalid, _ = valid.values.shape

    ntest=0
    if test is not None:
        ntest, _ = test.values.shape

    n = ntrain + nvalid + ntest

    # init Y and Y_hat
    Y = np.zeros((n, m))
    Y[0:ntrain] = train.values
    if (valid_init is not None):
        Y[ntrain:ntrain + nvalid] = valid_init
    Y_hat = np.random.rand(n, m).flatten()
    
    # set up constraint on Y_hat that all are >=0
    Y_hat_bounds = [(0, None)] * len(Y_hat)

    err = 1000.0
    iter = 0
    first=True
    while (True): # Y_hat is a flatten format
        # compute similarity score
        if not first:
            l=l_logistic_sim(theta, fmat)
            for i in range(n):
                L[i]=l(i)
            
        first=False

        # compute init total cost
        fval = cost_fun(Y_hat, Y, L, ntrain, alpha_train, alpha_unknown)
        print 'init total cost=', fval
        
        # run optimiaztion of Y_hat
        Y_hat, fval, _ = fmin_l_bfgs_b(cost_fun, Y_hat, g_cost_fun, \
                            args=(Y, L, ntrain, alpha_train, alpha_unknown), \
                            bounds=Y_hat_bounds, callback=None)
        print 'optimized total cost=', fval

        # compute init similarity cost
        print 'Y_hat shape', Y_hat.shape
        print 'very small Y_hat', Y_hat[Y_hat<1e-4].shape
        fval = l_cost_fun2(theta, fmat, Y_hat, Y)
        print 'init similarity cost=', fval        

        # run optimiaztion of theta
        theta, fval, _  = fmin_l_bfgs_b(l_cost_fun2, theta, g_logistic_sim, \
                            args=(fmat, Y_hat, Y))
        print 'optimized similarity cost=', fval

        iter+=1

        # evaluate erros
        Y_hat=Y_hat.reshape(Y.shape)
        e1, e2 = eval_model(train, valid, Y_hat)
        Y_hat=Y_hat.flatten()
        print "error at %d is: train(%f), valid(%f)" % (iter, e1, e2)
       
        # stop if reach max iteration or changes is very small
        if abs(e1 - err) / err < eps or \
            iter >= max_iter:
            break
        else:
            err=e1

    return Y_hat.reshape(Y.shape), theta

def run_model1(store_data_file, store_weather_file, test_data_file):
    """
    the model uses the square error to measure the difference between Y_hat and
    Y, and uses similarity to regulate Y_hat, that is, if one day's sale can be
    reconstructed by similar day's sale. The performance of the model in this 
    task is not particularly good. 
    """
    print "start here"

    # write header to test result
    with open('test_result.csv', 'w') as f:
        f.write('id,units\n')
        f.close()
    
    # load data
    store_data, store_weather, test = load_data2(store_data_file, \
          store_weather_file, test_data_file)

    # compute max item sales for each store as denominator
    store_data_max = store_data.groupby(level=1).max()

    # develop training and validation set
    train, valid = develop_valid_set2(store_data, store_weather, valid_size=0)   

    # categorize testing data with a relevant but much smaller training set
    target_set = build_target_set(train, valid, test, store_weather)

    # run prediction on testing data of each category
    for n, trn, vld, tst in target_set:
        print "%d, train(%d), valid(%d), test(%d)" % (n, len(trn), len(vld), len(tst))

        # normalize training, validing and testing data set
        nm_trn = normalize_store_data(trn, store_data_max)
        nm_vld = normalize_store_data(vld, store_data_max)
        nm_tst = normalize_store_data(tst, store_data_max)

        # compute similarity matrix
        l,_ = l_sim(nm_trn, nm_vld, nm_tst, store_weather)

        v_init = None
        Y_hat2 = None

        for i in range(1):
            # run prediction on all validation and testing data set
            Y_hat = build_model(nm_trn, nm_vld, nm_tst, l, v_init)
            # save the code in case the model has stacking effect -- v_init=Y_hat[len(trn):len(trn)+len(vld)]

            # denormalize the sale
            Y_hat2 = denormalize_store_data(trn, vld, tst, Y_hat, store_data_max)

            # evaluate error in training and validation set
            e1, e2 = eval_model(trn, vld, Y_hat2)
            print "error at %d is: train(%f), valid(%f)" % (i, e1, e2)

        # write results to test result
        write_submission(trn, vld, tst, Y_hat2, 'test_result.csv')

def run_model1v1(store_data_file, store_weather_file, test_data_file):
    """
    this is an update on the model1 that each optimization only includes one
    unknown day, which ignores the similarity constraint on unknown days. On
    validation set, it doesn't seem offer any improvement over model1.
    """
    print "start here"

    # write header to test result
    with open('test_result.csv', 'w') as f:
        f.write('id,units\n')
        f.close()

    # load data
    store_data, store_weather, test = load_data2(store_data_file, \
          store_weather_file, test_data_file)

    # compute max sale for each item at each store as denominator
    store_data_max = store_data.groupby(level=1).max()

    # develop training and validation set
    train, valid = develop_valid_set2(store_data, store_weather, valid_size=70)   

    # categorize testing data with a relevant but much smaller training set
    target_set = build_target_set(train, valid, test, store_weather)

    # run prediction on testing data of each category
    for n, trn, vld, tst in target_set:
        print "%d, train(%d), valid(%d), test(%d)" % (n, len(trn), len(vld), len(tst))
        
        # normalize training, validing and testing data set
        nm_trn = normalize_store_data(trn, store_data_max)
        nm_vld = normalize_store_data(vld, store_data_max)
        nm_tst = normalize_store_data(tst, store_data_max)

        # compute feature matrix
        _, m = l_sim(nm_trn, nm_vld, nm_tst, store_weather)

        v_init = None

        # init Y_hat
        Y_hat = np.zeros((len(nm_trn)+len(nm_vld)+len(nm_tst), nm_trn.values.shape[1]))

        # predicting validation data
        helper_model2_1(nm_vld, len(nm_trn), nm_trn, m, Y_hat, store_data_max)

        # predicting testing data
        helper_model2_1(nm_tst, len(nm_trn)+len(nm_vld), nm_trn, m, Y_hat, store_data_max)

        # evaluate error in training and validation set
        e1, e2 = eval_model(trn, vld, Y_hat)
        print "error at %d is: train(%f), valid(%f)" % (n, e1, e2)

        # write results to test result
        write_submission(trn, vld, tst, Y_hat, 'test_result.csv')        

def helper_model1v1(df, offset, train, m, Y_hat, store_data_max, v_init=None):
    # construct feature matrix with one row more than training data
    m2=np.zeros((len(train)+1,m.shape[1]))

    # copy training feature matrix
    m2[:len(train)]=m[:len(train)]

    for i in range(len(df)):        
        m2[-1]=m[offset+i]
        def l(i):
            g=np.dot(m2, m2[i])
            d=np.sum(g)
            g=-g
            g[i]=d+g[i]
            return g
        df0=df.iloc[i:i+1]
        Y_hat0 = build_model(train, None, df0, l, v_init)
        Y_hat1 = denormalize_store_data(train, None, df0, Y_hat0, store_data_max)
        Y_hat[offset+i]=Y_hat1[-1]

def run_model2(store_data_file, store_weather_file, test_data_file):
    print "start here"

    # write header to test result
    with open('test_result.csv', 'w') as f:
        f.write('id,units\n')
        f.close()
    
    # load data
    store_data, store_weather, test = load_data2(store_data_file, \
          store_weather_file, test_data_file)

    # compute max item sales for each store as denominator
    store_data_max = store_data.groupby(level=1).max()

    # develop training and validation set
    train, valid = develop_valid_set2(store_data, store_weather, valid_size=0)   

    # categorize testing data with a relevant but much smaller training set
    target_set = build_target_set(train, valid, test, store_weather)

    # run prediction on testing data of each category
    for n, trn, vld, tst in target_set:
        print "%d, train(%d), valid(%d), test(%d)" % (n, len(trn), len(vld), len(tst))

        # normalize training, validing and testing data set
        nm_trn = normalize_store_data(trn, store_data_max)
        nm_vld = normalize_store_data(vld, store_data_max)
        nm_tst = normalize_store_data(tst, store_data_max)

        Y_hat, theta=build_model2(nm_trn, nm_vld, nm_tst, store_weather)

        # denormalize the sale
        Y_hat2 = denormalize_store_data(trn, vld, tst, Y_hat, store_data_max)

        # evaluate error in training and validation set
        e1, e2 = eval_model(trn, vld, Y_hat2)
        print "error is: train(%f), valid(%f)" % (e1, e2)

        # write results to test result
        write_submission(trn, vld, tst, Y_hat2, 'test_result.csv')

def run_model3(store_data_file, store_weather_file, test_data_file):
    print "start here"

    # write header to test result
    with open('test_result.csv', 'w') as f:
        f.write('id,units\n')
        f.close()
    
    # load data
    store_data, store_weather, test = load_data2(store_data_file, \
          store_weather_file, test_data_file)

    # compute max item sales for each store as denominator
    store_data_max = store_data.groupby(level=1).max()

    # develop training and validation set
    train, valid = develop_valid_set2(store_data, store_weather, valid_size=0)   

    # categorize testing data with a relevant but much smaller training set
    target_set = build_target_set2(train, valid, test, store_weather)

    # run prediction on testing data of each category
    for n, trn, vld, tst in target_set:
        print "%d, train(%d), valid(%d), test(%d)" % (n, len(trn), len(vld), len(tst))

        # normalize training, validing and testing data set
        nm_trn = normalize_store_data(trn, store_data_max)
        nm_vld = normalize_store_data(vld, store_data_max)
        nm_tst = normalize_store_data(tst, store_data_max)

        Y_hat=build_model3(nm_trn, nm_vld, nm_tst, store_weather)

        # denormalize the sale
        Y_hat2 = denormalize_store_data(trn, vld, tst, Y_hat, store_data_max)

        # evaluate error in training and validation set
        e1, e2 = eval_model(trn, vld, Y_hat2)
        print "error is: train(%f), valid(%f)" % (e1, e2)

        # write results to test result
        write_submission(trn, vld, tst, Y_hat2, 'test_result.csv')

def run_model4(store_data_file, store_weather_file, test_data_file):
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
        _,m = l_sim(nm_trn, nm_vld, nm_tst, store_weather)
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