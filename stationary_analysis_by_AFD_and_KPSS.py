import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
def kpss_test(df):    
    statistic, p_value, n_lags, critical_values = kpss(df.values)    
    print(f'KPSS Statistic: {statistic}')
    print(f'p-value: {p_value}')
    print(f'num lags: {n_lags}')
    print('Critial Values:')
    for key, value in critical_values.items():
        print(f'   {key} : {value}')


def adf_test(df):
    result = adfuller(df.values)
    print('ADF Statistics: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
        

variables=['CPU_usage','Canonical_memory_usage','Maximum_memory_usage']
path='Workload of Machines with High Utilization\\'
entries = os.listdir(path)
for i in entries:
    print("\nMachine id: ",i[:len(i)-4])
    data1=pd.read_csv(path+i)
    for j in variables:
        data=data1[[j]].diff().dropna()
        #print(kpss_test(data))
        print(adf_test(data))
       
