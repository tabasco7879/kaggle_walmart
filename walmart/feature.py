# similarity feature functions
# is this the same store
# weather of 3 day window
# is raining
# is snowing
# rain<=0.4 or rain=T
# rain<=0.99
# rain<=1.96
# rain<=3.94
# rain>3.94
# snowfall<=2
# snowfall>2 and <=5.9
# snowfall>5.9 and <=7.9
# snowfall>7.9 and <=9.8
# snowfall>9.8

import pandas as pd
import numpy as np
from utils import is_numeric

def compute_feature(store_weather_data, store_data):
    feature_mat=[]
    for k in store_data.index:
        # store feature
        f=[0]*45
        store_id=k[1]                
        f[store_id-1]=1

        f0=compute_weather_feature(store_weather_data, k)
        f=f+f0
        for d in pd.date_range(end=k[0], periods=4)[:-1]:
            f0=compute_weather_feature(store_weather_data, (d, k[1]))
            f=f+f0
        for d in pd.date_range(start=k[0], periods=4)[1:]:
            f0=compute_weather_feature(store_weather_data, (d, k[1]))
            f=f+f0        
        feature_mat.append(f)
    m=np.array(feature_mat)*1.0
    norm=np.sum(m*m, axis=1)**0.5
    return m/norm[:,np.newaxis]

def compute_feature2(store_weather_data, store_data):    
    def f(x):
        k=x.name        
        f=compute_weather_feature(store_weather_data, k)        
        for d in pd.date_range(end=k[0], periods=4)[:-1]:
            f0=compute_weather_feature(store_weather_data, (d, k[1]))
            f=f+f0
        for d in pd.date_range(start=k[0], periods=4)[1:]:
            f0=compute_weather_feature(store_weather_data, (d, k[1]))
            f=f+f0        
        return pd.Series(f)
    store_feature_data=store_data.apply(lambda x: f(x), axis=1)
    m=store_feature_data.values*1.0
    norm=np.sum(m*m, axis=1)**0.5    
    return m/norm[:,np.newaxis]   

def is_rain0(day): return 1 if day['preciptotal']=='T' \
        or (is_numeric(day['preciptotal']) and float(day['preciptotal'])<=0.4) else 0

def is_rain1(day): return 1 if is_numeric(day['preciptotal']) \
    and 0.4<float(day['preciptotal'])<=0.99 else 0

def is_rain2(day): return 1 if is_numeric(day['preciptotal']) \
    and 0.99<float(day['preciptotal'])<=1.96 else 0

def is_rain3(day): return 1 if is_numeric(day['preciptotal']) \
    and 1.96<float(day['preciptotal'])<=3.94 else 0

def is_rain4(day): return 1 if is_numeric(day['preciptotal']) \
    and float(day['preciptotal'])>3.94 else 0

def is_snow0(day): return 1 if day['snowfall']=='T' \
        or (is_numeric(day['snowfall']) and float(day['snowfall'])<=2) else 0 

def is_snow1(day): return 1 if is_numeric(day['snowfall']) \
        and 2<float(day['snowfall'])<=5.9 else 0

def is_snow2(day): return 1 if is_numeric(day['snowfall']) \
        and 5.9<float(day['snowfall'])<=7.9 else 0

def is_snow3(day): return 1 if is_numeric(day['snowfall']) \
        and 7.9<float(day['snowfall'])<=9.8 else 0

def is_snow4(day): return 1 if is_numeric(day['snowfall']) \
    and float(day['snowfall'])>9.8 else 0
        
def compute_weather_feature(store_weather_data, k):
    funs=[is_rain0, is_rain1, is_rain2, is_rain3, is_rain4, \
        is_snow0, is_snow1, is_snow2, is_snow3, is_snow4]

    if k in store_weather_data.index:
        f=[fun(store_weather_data.loc[k]) for fun in funs]
        if sum(f)==0:
            return f+[1]
        else:
            return f+[0]
    else:
        f=[0]*len(funs)        
        return f+[1]
