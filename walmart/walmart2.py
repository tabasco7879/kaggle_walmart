import pandas as pd
import numpy as np
import random

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

def denormalize_store_data(train, valid, test, Y_hat, store_data_max):        
    def f(x):
        _, store_nbr=x.name
        norm=store_data_max.loc[store_nbr]
        return norm
    # count the total number of rows
    ntrain, m = train.values.shape

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
    Y_hat2[:ntrain]=Y_hat[:ntrain]*train_norm

    # denomalize validation data
    if(nvalid>0):
        valid_norm=valid.apply(lambda x: f(x), axis=1).values
        Y_hat2[ntrain:ntrain+nvalid]=Y_hat[ntrain:ntrain+nvalid]*valid_norm

    # denomalize testing data
    if(ntest>0):
        test_norm=test.apply(lambda x: f(x), axis=1).values
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
    print "complete develop valid set(%d)"%len(valid)
    return (train, valid)

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
    while target_year<=end_year and target_month<end_month:
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