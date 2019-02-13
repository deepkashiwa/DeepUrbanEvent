#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 12:54:44 2018

@author: huangdou
"""

import numpy as np
#import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from sklearn.decomposition import PCA
import pickle

def pca(n_components, data, filename):
    model = PCA(n_components=n_components)
    principaComponents = model.fit_transform(data)
    pickle.dump(model, open(filename, 'wb'))
    return principaComponents

def var(maxlag, data):
    model = VAR(data)
    results = model.fit(maxlags=maxlag, ic='aic',verbose=1)
    
    lag_order = maxlag
    forecast = results.forecast(data[-lag_order:], 12)
    return forecast

def rolling(temp_data):
    results_temp = np.zeros((264, 12, 64))
    for i in range(264):
        data = temp_data[i:2880+i]
#        data.extend(train[i * 6:])
#        data.extend(test[: i * 6])
        print(np.shape(data))
        results_temp[i:(i+1)] = var(12, data)
    return results_temp
        
   
def getXSYS(allData, TIMESTEP):
    XS, YS = [], []
    for i in range(allData.shape[0] - TIMESTEP - TIMESTEP):
        x = allData[i : i + TIMESTEP, :, :, :]
        y = allData[i + TIMESTEP: i + TIMESTEP + TIMESTEP, :, :, :]
        XS.append(x), YS.append(y)
    XS, YS = np.array(XS), np.array(YS)
    return XS, YS

def score(test, results):
    x_true, y_true = getXSYS(test, 12)
#    x_pred, y_pred = getXSYS(results, 6)
    y_pred = results
    
    score = (((y_pred - y_true) * 500) ** 2).mean()
    with open('../data/popScore.csv', 'w') as wf:
        wf.write(str(score))
     
if __name__ == '__main__':
    
    filename = '../data/PCA.sav'

    train = np.load('../data/popTrainValidate.npy')
    test = np.load('../data/popSpecial.npy')
    
    print('train data shape: {}\ntest data shape: {}'.format(train.shape, test.shape))
    
    train_ = train.reshape((2880, 6400))
    test_ = test.reshape((288, 6400))
    
    ## pca
    data = []
    data.extend(train_)
    data.extend(test_)
    
    temp_data = pca(64, data, filename)
    
    ## predictions
    results_temp = rolling(temp_data)
    pca_ = pickle.load(open(filename, 'rb'))
    results = pca_.inverse_transform(results_temp)
    results = results.reshape((264, 12, 80, 80, 1))
    np.save('../data/popResult.npy', results)
    score(test, results)