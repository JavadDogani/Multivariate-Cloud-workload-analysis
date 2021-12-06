import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import os


        
from statsmodels.tsa.stattools import grangercausalitytests
def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False, max_lag=15):    
   
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=max_lag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(max_lag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df
    
path='Data utilization of Machines with High Utilization\\'
variable1='CPU_usage'
variable2='Canonical_memory_usage'
entries = os.listdir(path)
for i in entries:
    print("\nMachine id: ",i[:len(i)-4])
    data1=pd.read_csv(path+i)
    data=data1[[variable1,variable2]].diff().dropna()
    #plt.plot(data)
    cpdatau_transform = data.diff().dropna()
    result=grangers_causation_matrix(data, variables = [variable1,variable2])
    print(result)
    
