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

def compute_feature2(store_weather_data, store_data, hidden_feature=None):    
    def f(x):
        k=x.name        
        fmnth=[0]*12
        fmnth[k[0].month-1]=1
        fwkdy=[0]*7
        fwkdy[k[0].weekday()]=1

        fwthr=compute_weather_feature(store_weather_data, k)        
        for d in pd.date_range(end=k[0], periods=4)[:-1]:
            f=compute_weather_feature(store_weather_data, (d, k[1]))
            fwthr=fwthr+f
        for d in pd.date_range(start=k[0], periods=4)[1:]:
            f=compute_weather_feature(store_weather_data, (d, k[1]))
            fwthr=fwthr+f        
        return pd.Series(fmnth+fwkdy+fwthr)
    store_feature_data=store_data.apply(lambda x: f(x), axis=1)
    n,m=store_feature_data.values.shape
    if hidden_feature is None:
        fmat=np.zeros((n,m))
    else:
        fmat=np.zeros((n,m+hidden_feature.shape[1]))
        fmat[:,m:]=hidden_feature
    fmat[:,:m]=store_feature_data.values*1.0    
    return fmat

def is_rain0(day): return 1 if 'RA' in day['codesum'] and \
        not is_rain1(day) and not is_rain2(day) and not is_rain3(day) and \
        not is_rain4(day) else 0

def is_rain1(day): return 1 if 'RA' in day['codesum'] and \
        is_numeric(day['preciptotal']) and 0.4<float(day['preciptotal'])<=0.99 else 0

def is_rain2(day): return 1 if 'RA' in day['codesum'] and \
        is_numeric(day['preciptotal']) and 0.99<float(day['preciptotal'])<=1.96 else 0

def is_rain3(day): return 1 if 'RA' in day['codesum'] and \
        is_numeric(day['preciptotal']) and 1.96<float(day['preciptotal'])<=3.94 else 0

def is_rain4(day): return 1 if 'RA' in day['codesum'] and \
        is_numeric(day['preciptotal']) and float(day['preciptotal'])>3.94 else 0

def is_snow0(day): return 1 if 'RA' in day['codesum'] and \
        not is_snow1(day) and not is_snow2(day) and not is_snow3(day) and \
        not is_snow4(day) else 0

def is_snow1(day): return 1 if 'SN' in day['codesum'] and \
        is_numeric(day['snowfall']) and 2<float(day['snowfall'])<=5.9 else 0

def is_snow2(day): return 1 if 'SN' in day['codesum'] and \
        is_numeric(day['snowfall']) and 5.9<float(day['snowfall'])<=7.9 else 0

def is_snow3(day): return 1 if 'SN' in day['codesum'] and \
        is_numeric(day['snowfall']) and 7.9<float(day['snowfall'])<=9.8 else 0

def is_snow4(day): return 1 if 'SN' in day['codesum'] and \
        is_numeric(day['snowfall']) and float(day['snowfall'])>9.8 else 0

def is_temp0(day): return 1 if 'M' in day['depart'] else 0

def is_temp1(day): return 1 if not is_temp0(day) and is_numeric(day['depart']) and \
        0<abs(float(day['depart']))<=10 else 0

def is_temp2(day): return 1 if not is_temp0(day) and is_numeric(day['depart']) and \
        10<abs(float(day['depart']))<=20 else 0

def is_temp3(day): return 1 if not is_temp0(day) and is_numeric(day['depart']) and \
        abs(float(day['depart']))>20 else 0

def is_TS(day): return 1 if 'TS' in day['codesum'] else 0

def is_GR(day): return 1 if 'GR' in day['codesum'] else 0

def is_DZ(day): return 1 if 'DZ' in day['codesum'] else 0

def is_PL(day): return 1 if 'PL' in day['codesum'] else 0

def is_FG2(day): return 1 if 'FG+' in day['codesum'] else 0

def is_FG(day): return 1 if not is_FG2(day) and 'FG' in day['codesum'] else 0

def is_BR(day): return 1 if 'BR' in day['codesum'] else 0

def is_UP(day): return 1 if 'UP' in day['codesum'] else 0

def is_HZ(day): return 1 if 'HZ' in day['codesum'] else 0

def is_FU(day): return 1 if 'FU' in day['codesum'] else 0

def is_DU(day): return 1 if 'DU' in day['codesum'] else 0

def is_SS(day): return 1 if 'SS' in day['codesum'] else 0

def is_SQ(day): return 1 if 'SQ' in day['codesum'] else 0

def is_FZ(day): return 1 if 'FZ' in day['codesum'] else 0

def is_MI(day): return 1 if 'MI' in day['codesum'] else 0

def is_BC(day): return 1 if 'BC' in day['codesum'] else 0

def is_BL(day): return 1 if 'BL' in day['codesum'] else 0

def is_VC(day): return 1 if 'VC' in day['codesum'] else 0
        
def compute_weather_feature(store_weather_data, k):
    funs=[is_rain0, is_rain1, is_rain2, is_rain3, is_rain4, \
          is_snow0, is_snow1, is_snow2, is_snow3, is_snow4, \
          is_temp0, is_temp1, is_temp2, is_temp3, \
          is_TS, is_GR, is_DZ, is_PL, is_FG2, is_FG, is_BR, \
          is_UP, is_HZ, is_FU, is_DU, is_SS, is_SQ, is_FZ, \
          is_MI, is_BC, is_BL, is_VC]

    if k in store_weather_data.index:
        f=[fun(store_weather_data.loc[k]) for fun in funs]        
        return f
    else:
        f=[0]*len(funs)
        return f
