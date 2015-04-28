###############################################################################
#   I load the training data from file and then convert it a format that each
#   row is indexed by store_nbr and date along with 111 columns for the sales
#   of each item. This should be sepreated into two dataframes: one for 
#   store_nbr and date 
"""
    store_weather_data.loc[train_m.index]
    train_m.loc[[('2012-01-01',1), ('2012-01-01',3)]]
    train_m.as_matrix
    def getweek(x):
        k=x.name
        return (k[0]-pd.to_datetime('2012-1-1')).days/7

    week=store_data.apply(lambda x: getweek(x), axis=1)
    store_data['week']=week
    store_data.set_index('week', append=True, inplace=True)
    store_data[:450].groupby(level=['week','store_nbr']).sum()
"""

import pandas as pd
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import random
from utils import is_numeric
from cost import cost_fun, l_cost_fun, l_g_cost_fun, g_cost_fun, fun_log_error
from walmart2 import load_data2, normalize_store_data, develop_valid_set2, build_target_set, denormalize_store_data
from similarity import l_similarity
from sklearn import linear_model
    
def is_bad_weather(r):
    bad=False
    if ((is_numeric(r['snowfall']) and float(r['snowfall'])>=2) or
        (is_numeric(r['preciptotal']) and float(r['preciptotal'])>=1)):
        bad=True
    return bad

def build_weather_day_range(weather_day_idx, store_data):
    r=set()
    for i in weather_day_idx:
        r.add(i)
        for d in pd.date_range(end=i[0], periods=4)[:-1]:
            if ((d, i[1]) in store_data.index):
                r.add((d, i[1]))
        for d in pd.date_range(start=i[0], periods=4)[1:]:
            if ((d, i[1]) in store_data.index):
                r.add((d, i[1]))
    return list(r)

def load_data(train_data_file, weather_file, station_store_mapping_file):
    store_data = pd.read_csv(train_data_file, header=0, \
                    delimiter=",", quoting=1, parse_dates=['date'])
    store_data = pd.pivot_table(store_data, values='units', \
                    index=['date', 'store_nbr'], columns=['item_nbr'])
    weather_data = pd.read_csv(weather_file, header=0, \
                    delimiter=",", quoting=1, parse_dates=['date'])
    station_store_mapping=pd.read_csv(station_store_mapping_file, header=0, \
                    delimiter=",", quoting=1)
    store_weather_data=pd.merge(weather_data, station_store_mapping, \
                    on='station_nbr')
    store_weather_data=store_weather_data.set_index(['date', 'store_nbr'])
    
    return (store_data, store_weather_data)

def develop_valid_set(store_data, store_weather_data):
    good_bad_weather=store_weather_data.apply(lambda x: is_bad_weather(x), axis=1)
    df=store_data[good_bad_weather==True]
    store_bad_weather_data=df[df[1].isnull()==False] # only pick up bad weather data
    store_bad_weather_data.sortlevel(inplace=True)  # required for the slice    
    weather_day_candi=store_bad_weather_data['2013-01-01':'2014-10-31']
    weather_day_idx=random.sample(weather_day_candi.index, 10)    
    valid_idx=build_weather_day_range(weather_day_idx, store_data)
    valid=store_data.loc[valid_idx]
    train=store_data.drop(valid_idx)
    return (train, valid)

def build_model(train, valid, test, l, \
                    valid_init=None, alpha_train=1000, alpha_unknown=0.01):    
    train_num, m=train.values.shape
    valid_num, _=valid.values.shape
    n=train_num+valid_num
    if test is not None:
        test_num, _ = test.values.shape
        n+=test_num
    Y=np.zeros((n, m))
    Y[0:train_num]=train.values
    if valid_init is not None:
        Y[train_num:train_num+valid_num]=valid_init
    Y_hat=np.random.rand(n, m).flatten()
    cost=l_cost_fun(Y_hat, Y, l, train_num, alpha_train, alpha_unknown)
    print 'init cost=', cost    
    Y_hat_bounds=[(0, None)]*len(Y_hat)
    Y_hat=fmin_l_bfgs_b(l_cost_fun, Y_hat, l_g_cost_fun, \
                        args=(Y, l, train_num, alpha_train, alpha_unknown), \
                        bounds=Y_hat_bounds, callback=None)
    return Y_hat[0].reshape(Y.shape)

def eval_model(train, valid, Y_hat):
    train_num, m=train.values.shape
    valid_num, _=valid.values.shape
    t_error=fun_log_error(Y_hat[:train_num], \
                     train.values, train_num, 1, 0)
    v_error=fun_log_error(Y_hat[train_num:train_num+valid_num], \
                     valid.values, valid_num, 1, 0)
    return (t_error/train_num)**0.5, (v_error/valid_num)**0.5
       
def run_model(train_data_file, weather_file, station_store_mapping_file):
    store_data, store_weather_data = \
        load_data(train_data_file, weather_file, station_store_mapping_file)
    train, valid = develop_valid_set(store_data, store_weather_data)
    # memory intensive and may not work when data size is big
    # L=similarity(train, valid, test, store_weather_data)    
    # return a function and less memory requirement
    l,_=l_similarity(train, valid, None, store_weather_data)
    valid_init=None
    for i in range(1):    
        Y_hat=build_model(train, valid, None, l, valid_init)        
        e=eval_model(train, valid, Y_hat)
        print e
        train_num, m=train.values.shape
        valid_num, _=valid.values.shape
        valid_init=Y_hat[train_num:train_num+valid_num]

def run_model2(store_data_file, store_weather_file, test_data_file):
    """
    use the same model as run_model2 but use the functions from walrmat2
    the model only use the training data that are relevant to reduce computation
    """
    print "start here"

    with open('test_result.csv', 'w') as f:
        f.write('id,units\n')
        f.close()

    store_data, store_weather, test=load_data2(store_data_file, \
          store_weather_file, test_data_file)
    store_data_max=store_data.groupby(level=1).max()

    train, valid=develop_valid_set2(store_data, store_weather, valid_size=0)   
    target_set=build_target_set(train, valid, test, store_weather)
    for n, trn, vld, tst in target_set:
        print "%d, train(%d), valid(%d), test(%d)"%(n, len(trn), len(vld), len(tst))
        nm_trn=normalize_store_data(trn, store_data_max)
        nm_vld=normalize_store_data(vld, store_data_max)
        nm_tst=normalize_store_data(tst, store_data_max)
        l,_=l_similarity(nm_trn, nm_vld, nm_tst, store_weather)
        v_init=None
        Y_hat2=None
        for i in range(1):
            Y_hat=build_model(nm_trn, nm_vld, nm_tst, l, v_init)
            #v_init=Y_hat[len(trn):len(trn)+len(vld)]
            Y_hat2=denormalize_store_data(trn, vld, tst, Y_hat, store_data_max)
            e1, e2=eval_model(trn, vld, Y_hat2)
            print "error at %d is: train(%f), valid(%f)"%(i, e1, e2)
        write_submission(trn, vld, tst, Y_hat2, 'test_result.csv')

def run_model3(store_data_file, store_weather_file, test_data_file):
    """
    ridge regression
    """
    print "start here"

    with open('test_result.csv', 'w') as f:
        f.write('id,units\n')
        f.close()

    store_data, store_weather, test=load_data2(store_data_file, \
          store_weather_file, test_data_file)
    store_data_max=store_data.groupby(level=1).max()
    train, valid=develop_valid_set2(store_data, store_weather, valid_size=70)    
    target_set=build_target_set(train, valid, test, store_weather)
    for n, trn, vld, tst in target_set:
        print "%d, train(%d), valid(%d), test(%d)"%(n, len(trn), len(vld), len(tst))
        nm_trn=normalize_store_data(trn, store_data_max)
        nm_vld=normalize_store_data(vld, store_data_max)
        nm_tst=normalize_store_data(tst, store_data_max)
        _,m=l_similarity(nm_trn, nm_vld, nm_tst, store_weather)
        item_count=nm_trn.values.shape[1]
        Y_hat=np.zeros((len(nm_trn)+len(nm_vld)+len(nm_tst), item_count))
        X=m[:len(nm_trn)]
        for i in range(item_count):            
            Y=nm_trn.values[:,i]
            clf = linear_model.Ridge(alpha=1.0)
            clf.fit(X, Y)
            Y_hat[:,i]=clf.predict(m)
        Y_hat2=denormalize_store_data(trn, vld, tst, Y_hat, store_data_max)
        e1, e2=eval_model(trn, vld, Y_hat2)
        print "error is: train(%f), valid(%f)"%(e1, e2)
        write_submission(trn, vld, tst, Y_hat2, 'test_result.csv')

def write_submission(train, valid, test, Y_hat, test_result_file):
    row=len(train)+len(valid)
    col=Y_hat.shape[1]
    with open(test_result_file, 'a') as f:
        for x in test.index:
            item=test.loc[x]
            fmt=str(x[1])+'_%s_'+str(x[0])[:10]+',%f\n'            
            for i in range(col):
                s=fmt%(i+1, Y_hat[row, i])
                f.write(s)
            row+=1

#run_model("D:/walmart/train.csv", \
#          "D:/walmart/weather.csv", \
#          "D:/walmart/key.csv")

run_model3('C:/Users/tao.chen/skydrive/working/kaggle_walmart/data/store_train.txt', \
          'C:/Users/tao.chen/skydrive/working/kaggle_walmart/data/store_weather.txt', \
          'C:/Users/tao.chen/skydrive/working/kaggle_walmart/data/test.csv')