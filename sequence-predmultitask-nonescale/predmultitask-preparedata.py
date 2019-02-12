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
# NEW1
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

################# Parameter Setting #######################
meshTokyo = Mesh('tokyo', '500m')
################# Parameter Setting #######################

################### MODELNAME, DATA #######################
PATH = '../interpo_data/'
dataPATH = '../interpo_data/'
popFileName = dataPATH + '{}' + 'tokyo_' + meshTokyo.size + '_5min_pop.csv'
MAX_VALUE = 500
################### MODELNAME, DATA #######################

def main():
    # NEW2
    if not os.path.exists(PATH):
        os.makedirs(PATH + '/log')
    currentPython = sys.argv[0]
    shutil.copy2(currentPython, PATH)

    print(time.ctime())
    trainvalidateData = []
    for date in trainvalidateDates:
        data = meshDynamic.getGridPopTimeInterval(meshTokyo, popFileName.format(date))
        data = data[:-1, :, :, :]
        trainvalidateData.append(data)
    trainvalidateData = np.concatenate(trainvalidateData, axis=0)
    print(trainvalidateData.shape)

    trainvalidateData = trainvalidateData / MAX_VALUE

    print('save density trainvalidate into .npy start', time.ctime())
    np.save(PATH + '/' + 'popTrainValidate', trainvalidateData)
    print('save density trainvalidate into .npy end', time.ctime())

    specialData = meshDynamic.getGridPopTimeInterval(meshTokyo, popFileName.format(specialDate))
    specialData = specialData[:-1, :, :, :]
    print(specialData.shape)

    specialData = specialData / MAX_VALUE

    print('save density special into .npy start', time.ctime())
    np.save(PATH + '/' + 'popSpecial', specialData)
    print('save density special into .npy end', time.ctime())

if __name__ == '__main__':
    main()