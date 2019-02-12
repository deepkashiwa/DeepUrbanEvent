#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 15:50:52 2018

@author: huangdou
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from statsmodels.tsa.arima_model import ARIMA
from pandas.tools.plotting import autocorrelation_plot
from multiprocessing import Pool
import os
import glob
from sklearn.decomposition import PCA
import pickle
from scoreARIMA import *

def pca(n_components, data, filename):
    model = PCA(n_components=n_components)
    principaComponents = model.fit_transform(data)
    pickle.dump(model, open(filename, 'wb'))
    return principaComponents

def arima_parallel_(filename):
    train_ = np.load('../data/temp/train/' + filename)
    test_ = np.load('../data/temp/test/' + filename)
    result = np.zeros((264, 12))
#    for i in range(1):
    start = time.time()
    history = [x for x in train_[:, 0]]
#        print(i, time.ctime())
    for t in range(264):
#        if t % 12 == 0:
#            print(i, t, time.ctime())
        model = ARIMA(history, order = (12, 1, 0))
       
        model_fit = model.fit(disp = 0)

        output = model_fit.forecast(12)[0]

        yhat = output

        obs = test_[t : t+1, 0]
        history.append(obs[0])

        result[t][:] = yhat
#        print(time.ctime())
    print(i, time.time() - start)
    np.save('../data/temp/result/' + filename, result)

def findfile(scripts_path, file_path):
    filelist = []
    os.chdir(file_path)
    for file in glob.glob('*'):
        filelist.append(file)
    os.chdir(scripts_path)
    return filelist


if __name__ == '__main__':
    filename = '../data/PCA.sav'

    test = np.load('../data/popSpecial.npy')
    train = np.load('../data/popTrainValidate.npy')
    train_ = train.reshape((2880, 6400))
    test_ = test.reshape((288, 6400))

    ## pca
    data = []
    data.extend(train_)
    data.extend(test_)
    
    temp_data = pca(64, data, filename)
    
    for i in range(64):
        filetrain = 'train/' + np.str(i)
        filetest = 'test/' + np.str(i)
        temp_train = temp_data[:2880, i : (i + 1)]
        temp_test = temp_data[2880:, i : (i + 1)]
        np.save('../data/temp/' + filetrain, temp_train)
        np.save('../data/temp/' + filetest, temp_test)
        
#    filelist = findfile('../../../scripts/', '../data/temp/train/')
    
#    pool = Pool(7)
#    pool.map(arima_parallel_, [file for file in filelist])
    
#    Parallel(n_jobs = 8)(delayed(arima_parallel)(filename) for filename in filelist)

    START, END = 0, 64
    with open('./errorList.csv', 'w') as wf:
        for i in range(START, END):
            try:
                print(i, time.ctime(), '#' * 20)
                arima_parallel_(str(i) + '.npy')
            except:
                wf.write(str(i) + '\n')
                print(i, 'ARIMA Failed', time.ctime())
    
    results_temp = np.zeros((264, 12, 64))
    for i in range(START, END):
        temp = np.load('../data/temp/result/' + str(i) + '.npy')
        results_temp[:, :, i] = temp
    pca_ = pickle.load(open(filename, 'rb'))
    results = pca_.inverse_transform(results_temp)
    results = results.reshape((264, 12, 80, 80, 1))
#    np.save('../data/popResult.npy', results)
    test = np.load('../data/popSpecial.npy')
    train, test = getXSYS(test, 12)
    score(results, test)
