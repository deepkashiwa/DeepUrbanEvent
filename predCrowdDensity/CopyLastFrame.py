import csv
import numpy as np
import os
import shutil
import sys
import time
from datetime import datetime
import matplotlib.pyplot as plt

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from common.datastructure.Mesh import Mesh
from meshdynamic import meshDynamic
### NEW1
from common.dataparam.Param import *

from keras.models import load_model, Model, Sequential
from keras.layers import Input, merge, TimeDistributed, Flatten, RepeatVector, Reshape, UpSampling2D, concatenate, add, Dropout, Embedding
from keras.optimizers import RMSprop, Adam, SGD
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv3D
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.core import Dense
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, LearningRateScheduler

def getRealTest(allData):
    XS, YS = [], []
    for i in range(allData.shape[0] - TIMESTEP - TIMESTEP):
        x = allData[i:i+TIMESTEP, :, :, :]
        y = allData[i+TIMESTEP:i+TIMESTEP+TIMESTEP, :, :, :]
        XS.append(x), YS.append(y)
    XS, YS = np.array(XS), np.array(YS)
    return XS, YS

def testModel(name, testData):
    print('Model Evaluation Started ...', time.ctime())
    f = open(PATH + '/' + name + '_prediction_scores.txt', 'a')
    print('*' * 40)
    f.write('*' * 40 + '\n')

    XS, YS_true = getRealTest(testData)
    STEP6 = [XS[:, TIMESTEP-1:TIMESTEP, :, :, :]] * TIMESTEP
    YS_pred = np.concatenate(STEP6, axis=1)

    YS_pred = MAX_VALUE * YS_pred
    YS_true = MAX_VALUE * YS_true
    for i in range(TIMESTEP):
        stepYS = YS_pred[:, i:i + 1, :, :, :]
        stepYS_true = YS_true[:, i:i + 1, :, :, :]
        MSE_cur_onestep = np.mean((stepYS - stepYS_true) ** 2)
        f.write("Model MSE for multi step 500*Array, %d, %f\n" % (i, MSE_cur_onestep))
        print("Model MSE for multi step 500*Array", i, MSE_cur_onestep)
    MSE_multistep = np.mean((YS_pred - YS_true) ** 2)
    f.write("Model MSE for multi step 500*Array, %f\n" % MSE_multistep)
    print('MSE for multi step 500*Array', MSE_multistep)

    f.close()

meshTokyo = Mesh('tokyo','500m')
HEIGHT = 80
WIDTH = 80
CHANNEL = 1
### NEW2
BATCHSIZE = 4
MODELNAME = 'CopyLastFrame'
### NEW3
KEYWORD = EVENT + 'tokyo' + meshTokyo.size + '5minpop-nonescale'
PATH = '../model' + KEYWORD
dataPATH = '../interpo_data/'
popFileName = dataPATH + '{}' + 'tokyo_' + meshTokyo.size + '_5min_pop.csv'
MAX_VALUE = 500


def main():
    ### NEW4

    if not os.path.exists(PATH):
        os.makedirs(PATH + '/log')
    currentPython = sys.argv[0]
    shutil.copy2(currentPython, PATH)

    ### NEW5
    print(KEYWORD, time.ctime())
    specialData = meshDynamic.getGridPopTimeInterval(meshTokyo, popFileName.format(specialDate))
    specialData = specialData[:-1, :, :, :]
    print(specialData.shape)

    # specialData[specialData > MAX_VALUE] = MAX_VALUE
    specialData = specialData / MAX_VALUE
    testModel(MODELNAME, specialData)


if __name__ == '__main__':
    main()