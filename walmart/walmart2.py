import pandas as pd
import numpy as np
import random
import math
import time
from sklearn.cluster import k_means
from collections import defaultdict

def load_data2(store_data_file, store_weather_file, test_data_file):
    store_data=pd.read_csv(store_data_file, header=0, \
                delimiter=',', quoting=1, parse_dates=['date'], \
                index_col=['date', 'store_nbr'])
    store_weather=pd.read_csv(store_weather_file, header=0, \
                delimiter=',', quoting=1, parse_dates=['date'], \
                index_col=['date', 'store_nbr'])
    test_data = pd.read_csv(test_data_file, header=0, \
                    delimiter=",", quoting=1, parse_dates=['date'])
    test_data['units']=0
    test_data = pd.pivot_table(test_data, values='units', \
                    index=['date', 'store_nbr'], columns=['item_nbr'])
    store_data.sortlevel(inplace=True)
    store_weather.sortlevel(inplace=True)
    test_data.sortlevel(inplace=True)
    return (store_data, store_weather, test_data)

def normalize_store_data(store_data, store_data_max):
    def f(x):
        _, store_nbr=x.name
        norm=store_data_max.loc[store_nbr]
        n=x.values.astype('float64')/norm
        n[np.isnan(n)]=0.
        return n
    norm_store_data=store_data.apply(lambda x: f(x), axis=1)
    return norm_store_data

def denormalize_store_data(train, valid, test, Y_hat, store_data_max, column=None):
    def f(x):
        _, store_nbr=x.name
        if (column is not None):
            norm=store_data_max.loc[store_nbr][column]
        else:
            norm=store_data_max.loc[store_nbr]
        return norm
    # count the total number of rows
    ntrain, m = train.values.shape
    if column is not None:
        m=1

    nvalid=0
    if valid is not None:
        nvalid, _ = valid.values.shape

    ntest=0
    if test is not None:
        ntest, _ = test.values.shape

    # denomalize training data
    Y_hat2=np.zeros(Y_hat.shape)

    # find norm for each row of training data
    train_norm=train.apply(lambda x: f(x), axis=1).values

    if column is not None:
        Y_hat2[:ntrain]=Y_hat[:ntrain]*train_norm[:,np.newaxis]
    else:
        Y_hat2[:ntrain]=Y_hat[:ntrain]*train_norm

    # denomalize validation data
    if(nvalid>0):
        valid_norm=valid.apply(lambda x: f(x), axis=1).values
        if column is not None:
            Y_hat2[ntrain:ntrain+nvalid]=Y_hat[ntrain:ntrain+nvalid]*valid_norm[:,np.newaxis]
        else:
            Y_hat2[ntrain:ntrain+nvalid]=Y_hat[ntrain:ntrain+nvalid]*valid_norm

    # denomalize testing data
    if(ntest>0):
        test_norm=test.apply(lambda x: f(x), axis=1).values
        if column is not None:
            Y_hat2[ntrain+nvalid:]=Y_hat[ntrain+nvalid:]*test_norm[:,np.newaxis]
        else:
            Y_hat2[ntrain+nvalid:]=Y_hat[ntrain+nvalid:]*test_norm

    # update any less than 0 value to 0
    Y_hat2[Y_hat2<0]=0
    return Y_hat2

def build_day_range(store_day_idx, store_data, pre=3, aft=3):
    r=set()
    for i in store_day_idx:
        if (i in store_data.index):
            r.add(i)
        for d in pd.date_range(end=i[0], periods=pre+1)[:-1]:
            if ((d, i[1]) in store_data.index):
                r.add((d, i[1]))
        for d in pd.date_range(start=i[0], periods=aft+1)[1:]:
            if ((d, i[1]) in store_data.index):
                r.add((d, i[1]))
    return list(r)

def develop_valid_set2(store_data, store_weather, valid_size=70):
    store_weatherday=store_weather[store_weather['isweatherday']==1]
    core_cand=store_weatherday.loc['2013-01-01':]
    valid_set=set()
    while (len(valid_set)<valid_size):
        sample_day=random.sample(core_cand.index, 1)
        sample_range=build_day_range(sample_day, store_data)
        valid_set=valid_set.union(sample_range)
    valid_idx=list(valid_set)
    valid=store_data.loc[store_data.index.isin(valid_idx)]
    train=store_data.loc[~store_data.index.isin(valid_idx)]
    print 'complete develop valid set(%d)'%(len(valid))
    return (train, valid)

def develop_valid_set3(store_data, store_weather, valid_pct=0.1, pre=3, aft=3):
    def f(x):
        k=x.name
        if k[0]<pd.datetime(2013, 1, 1):
            return False        
        if (store_weather.loc[k]['isweatherday']==1):
            return True
        else:
            for d in pd.date_range(end=k[0], periods=pre+1)[:-1]:
                if ((d, k[1]) in store_weather.index and \
                        store_weather.loc[(d, k[1])]['isweatherday']==1):
                    return True                        
            for d in pd.date_range(start=k[0], periods=aft+1)[1:]:
                if ((d, k[1]) in store_weather.index and \
                        store_weather.loc[(d, k[1])]['isweatherday']==1):
                    return True
            return False            
    # build a list that in 3 day range of 
    related_store_data_idx=store_data.apply(lambda x:f(x), axis=1)
    related_store_data=store_data[related_store_data_idx]
    #print 'related_store_data', len(related_store_data)
    sample_size = int(math.ceil(len(related_store_data)*valid_pct))
    valid_idx=random.sample(related_store_data.index, sample_size)
    valid=store_data.loc[store_data.index.isin(valid_idx)]
    train=store_data.loc[~store_data.index.isin(valid_idx)]
    return train, valid
    

def build_target_set(train, valid, test, store_weather, pre=3, aft=3):
    def f(x):
        """
        1st position is 1 if the current day is a weather day
        2nd position is 1 if there is a weather day in previous 3 days
        3rd position is 1 if there is a weather day in next 3 days
        """
        k=x.name
        r=['0']*3
        if store_weather.loc[k]['isweatherday']==1:
            r[0]='1'
        for d in pd.date_range(end=k[0], periods=pre+1)[:-1]:
            if ((d, k[1]) in store_weather.index) and \
               (store_weather.loc[(d, k[1])]['isweatherday'])==1:
                r[1]='1'
                break
        for d in pd.date_range(start=k[0], periods=aft+1)[1:]:
            if ((d, k[1]) in store_weather.index) and \
               (store_weather.loc[(d, k[1])]['isweatherday'])==1:
                r[2]='1'
                break
        return int(''.join(r),2)

    valid_idx=valid.apply(lambda x: f(x), axis=1)
    train_idx=train.apply(lambda x: f(x), axis=1)
    test_idx=test.apply(lambda x: f(x), axis=1)
    target_set=[]
    for i in range(8):
        valid0=valid[valid_idx==i]
        train0=train[train_idx==i]
        test0=test[test_idx==i]
        if (len(valid0)>0 or len(test0)>0):
            target_set.append((i, train0, valid0, test0))
    return target_set

def build_target_set2(train, valid, test, store_weather, \
                      start_year=2013, start_month=4, \
                      end_year=2014, end_month=10, \
                      pre=3, aft=3):
    target_year=start_year
    target_month=start_month
    while target_year*12+target_month<=end_year*12+end_month:
        target_year0=target_year+(target_month-2)/12
        target_month0=(target_month-2)%12+1
        target_year1=target_year+target_month/12
        target_month1=(target_month)%12+1
        def f_trn(x):
            k=x.name
            if k[0].year in [target_year, target_year-1, target_year+1] and \
                k[0].month==target_month:
                return True
            if k[0].year==target_year0 and k[0].month==target_month0:
                return True
            if k[0].year==target_year1 and k[0].month==target_month1:
                return True
            if store_weather.loc[k]['isweatherday']==1:
                return True
            for d in pd.date_range(end=k[0], periods=pre+1)[:-1]:
                if ((d, k[1]) in store_weather.index) and \
                   (store_weather.loc[(d, k[1])]['isweatherday'])==1:
                    return True
            for d in pd.date_range(start=k[0], periods=aft+1)[1:]:
                if ((d, k[1]) in store_weather.index) and \
                   (store_weather.loc[(d, k[1])]['isweatherday'])==1:
                    return True
            return False

        def f_tst(x):
            k=x.name
            return k[0].year==target_year and \
                k[0].month==target_month

        train_idx=train.apply(lambda x: f_trn(x), axis=1)
        test_idx=test.apply(lambda x: f_tst(x), axis=1)

        train0=train[train_idx]
        test0=test[test_idx]

        yield (target_year*12+target_month, train0, valid, test0)

        target_year=target_year1
        target_month=target_month1

def build_target_set3(train, test, store_weather, \
                      store_data_max, \
                      columns=None, \
                      valid_pct=0.1, \
                      pre=3, aft=3):
    for col in train.columns:
        if columns is not None and col not in columns:
            continue
        def f(x):
            k=x.name
            store_nbr=k[1]
            if store_data_max.loc[store_nbr][str(col)]>0:
                return True
            return False
        def f2(x, label_dict):
            k=x.name
            return label_dict[(k[0].year, k[0].month)]
        def fg(x):
            return x[0].year, x[0].month

        train_idx=train.apply(lambda x: f(x), axis=1)
        test_idx=test.apply(lambda x: f(x), axis=1)
        train0=train[train_idx]
        test0=test[test_idx]

        grouped_train=train0[col].groupby(lambda x: fg(x))
        monthly_sales=grouped_train.mean()
        data=np.zeros((len(monthly_sales),3))
        data[:,:3]=[[i[0], i[1], monthly_sales[i]] for i in monthly_sales.index]
        label=k_means(data, 4)[1]
        label_dict = dict(zip(monthly_sales.index, label))

        train0_idx=train0.apply(lambda x: f2(x, label_dict), axis=1)
        test0_idx=test0.apply(lambda x: f2(x, label_dict), axis=1)
        for c in range(4):
            test1=test0[test0_idx==c]
            train1=train0[train0_idx==c]
            train2, valid2=develop_valid_set3(train1, store_weather, valid_pct, pre, aft)
            yield (col, train2, valid2, test1)

def build_target_set4(train, test, store_weather, \
                      store_data_max, \
                      columns=None, \
                      valid_pct=0.1, \
                      pre=3, aft=3):
    """
    if there exist month sales is 0, pick these month out and make this group 0
    otherwise split the data into seasons, 3-5, 6-8, 9-11, 12-2
    """
    for col in train.columns:
        if columns is not None and col not in columns:
            continue
        def f(x):
            k=x.name
            store_nbr=k[1]
            if store_data_max.loc[store_nbr][str(col)]>0:
                return True
            return False
        def f2(x, cat_dict):
            k=x.name
            return cat_dict[k[0].month]
        def fg(x):
            return x[0].month

        # filter out stores have no sale of the item
        train_idx=train.apply(lambda x: f(x), axis=1)
        test_idx=test.apply(lambda x: f(x), axis=1)
        train0=train[train_idx]
        test0=test[test_idx]                
        
        # check average monthly sale
        grouped_train=train0.loc['2013-01-01':][col].groupby(lambda x: fg(x))
        monthly_sales=grouped_train.mean()
        cat_dict=dict()
        for idx in monthly_sales.index:
            if monthly_sales[idx]<0.5:
                cat_dict[idx]=0
            else:
                if 5>=idx>=3:
                    cat_dict[idx]=1
                elif 8>=idx>=6:
                    cat_dict[idx]=2
                elif 11>=idx>=9:
                    cat_dict[idx]=3
                else:
                    cat_dict[idx]=4
        print 'item(%s)'%col, cat_dict

        train0_idx=train0.apply(lambda x: f2(x, cat_dict), axis=1)
        test0_idx=test0.apply(lambda x: f2(x, cat_dict), axis=1)
        for c in range(len(cat_dict)):
            test1=test0[test0_idx==c]
            train1=train0[train0_idx==c]
            if (len(test1)==0): continue
            train2, valid2=develop_valid_set3(train1, store_weather, valid_pct, pre, aft)
            yield (col, train2, valid2, test1, c)