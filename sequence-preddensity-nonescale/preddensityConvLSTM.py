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

from keras import backend as K
from keras.models import load_model, Model, Sequential
from keras.layers import Input, merge, TimeDistributed, Flatten, RepeatVector, Reshape, UpSampling2D, concatenate, add, Dropout, Embedding, Lambda
from keras.optimizers import RMSprop, Adam, SGD
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv3D
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.core import Dense
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, LearningRateScheduler
### NEW1
from common.dataparam.Param import *

def getXSYS(allData):
    XS, YS = [], []
    for i in range(allData.shape[0] - TIMESTEP - TIMESTEP):
        x = allData[i:i+TIMESTEP, :, :, :]
        y = allData[i+TIMESTEP:i+TIMESTEP+TIMESTEP, :, :, :]
        XS.append(x), YS.append(y)
    XS, YS = np.array(XS), np.array(YS)
    if MODELNAME == 'CNN':
        XS = XS.swapaxes(4, 1)
        XS = np.squeeze(XS)
        YS = YS.swapaxes(4, 1)
        YS = np.squeeze(YS)
    elif MODELNAME == 'ConvLSTM':
        pass
    elif MODELNAME == 'PredNet':
        pass
    else:
        pass
    return XS, YS

def getRealTest(allData):
    XS, YS = [], []
    for i in range(allData.shape[0] - TIMESTEP - TIMESTEP):
        x = allData[i:i+TIMESTEP, :, :, :]
        y = allData[i+TIMESTEP:i+TIMESTEP+TIMESTEP, :, :, :]
        XS.append(x), YS.append(y)
    XS, YS = np.array(XS), np.array(YS)
    return XS, YS

def getModel(name):
    if name == 'ConvLSTM':
        seq = Sequential()
        seq.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
                           input_shape=(None, HEIGHT, WIDTH, CHANNEL),
                           padding='same', return_sequences=True))
        seq.add(BatchNormalization())

        seq.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
                           padding='same', return_sequences=False))
        seq.add(BatchNormalization())

        seq.add(Lambda(lambda x: K.concatenate([x[:, np.newaxis, :, :, :]] * TIMESTEP, axis=1)))

        seq.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
                           padding='same', return_sequences=True))
        seq.add(BatchNormalization())

        seq.add(ConvLSTM2D(filters=CHANNEL, kernel_size=(3, 3),
                           padding='same', return_sequences=True, activation='relu'))
        return seq
    else:
        return None

def testModel(name, testData):
    assert os.path.exists(PATH + '/' + name + '.h5'), 'model is not existing'
    print('Model Evaluation Started ...', time.ctime())
    train_model = load_model(PATH + '/'+ name + '.h5', custom_objects={'TIMESTEP':TIMESTEP})
    train_model.summary()

    f = open(PATH + '/' + name + '_prediction_scores.txt', 'a')
    print('*' * 40)
    f.write('*' * 40 + '\n')

    XS, YS_true = getRealTest(testData)
    YS_pred = train_model.predict(XS)
    print(YS_true.shape, YS_pred.shape)

    assert YS_pred.shape == YS_true.shape
    for i in range(TIMESTEP):
        pred = YS_pred[:, i:i + 1, :, :, :]
        true = YS_true[:, i:i + 1, :, :, :]
        MSE_cur_onestep = np.mean((pred - true) ** 2)
        f.write("Model MSE for multi step Array, %d, %f\n" % (i, MSE_cur_onestep))
        print("Model MSE for multi step Array", i, MSE_cur_onestep)
    MSE_multistep = np.mean((YS_pred - YS_true) ** 2)
    f.write("Model MSE for multi step Array, %f\n" % MSE_multistep)
    print('MSE for multi step Array', MSE_multistep)

    YS_pred = MAX_VALUE * YS_pred
    YS_true = MAX_VALUE * YS_true
    for i in range(TIMESTEP):
        pred = YS_pred[:, i:i + 1, :, :, :]
        true = YS_true[:, i:i + 1, :, :, :]
        MSE_cur_onestep = np.mean((pred - true) ** 2)
        f.write("Model MSE for multi step 500*Array, %d, %f\n" % (i, MSE_cur_onestep))
        print("Model MSE for multi step 500*Array", i, MSE_cur_onestep)
    MSE_multistep = np.mean((YS_pred - YS_true) ** 2)
    f.write("Model MSE for multi step 500*Array, %f\n" % MSE_multistep)
    print('MSE for multi step 500*Array', MSE_multistep)

    f.close()

def trainModel(name, trainvalidateData):
    print('Model Training Started ...', time.ctime())
    trainXS, trainYS = getXSYS(trainvalidateData)
    print(trainXS.shape, trainYS.shape)

    model = getModel(name)
    model.compile(loss=LOSS, optimizer=OPTIMIZER)
    model.summary()
    csv_logger = CSVLogger(PATH + '/log' + '/' + name + '.log')
    checkpointer = ModelCheckpoint(filepath=PATH + '/' + name + '.h5', verbose=1, save_best_only=True)
    model.fit(trainXS, trainYS, batch_size=BATCHSIZE, epochs=EPOCH, shuffle=True,
              callbacks=[csv_logger, checkpointer, LR], validation_split=SPLIT)
    print('Model Training Ended ', time.ctime())

################# Parameter Setting #######################
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.45
config.gpu_options.visible_device_list = '1'
set_session(tf.Session(config=config))

meshTokyo = Mesh('tokyo', '500m')
HEIGHT = 80
WIDTH = 80
CHANNEL = 1
### NEW2
BATCHSIZE = 4
print('BATCHSIZE = 4, lr=0.0001')
SPLIT = 0.2
# LR = LearningRateScheduler(lambda epoch: 0.01 if epoch < 50 else 0.001)
LR = LearningRateScheduler(lambda epoch: 0.0001)
EPOCH = 200
LOSS = 'mse'
OPTIMIZER = 'adam'
################# Parameter Setting #######################

################### MODELNAME, DATA #######################
# Current is One Hour, step = 12
MODELNAME = 'ConvLSTM'
### NEW3
KEYWORD = EVENT + 'tokyo' + meshTokyo.size + '5minpop-nonescale'
PATH = '../model' + KEYWORD
dataPATH = '../interpo_data/'
popFileName = dataPATH + '{}' + 'tokyo_' + meshTokyo.size + '_5min_pop.csv'
MAX_VALUE = 500
################### MODELNAME, DATA #######################

def main():
    ### NEW4

    if not os.path.exists(PATH):
        os.makedirs(PATH + '/log')
    currentPython = sys.argv[0]
    shutil.copy2(currentPython, PATH)

    if not os.path.exists(PATH + '/' + MODELNAME + '.h5'):
        ### NEW5
        print(KEYWORD, time.ctime())
        trainvalidateData = []
        for date in trainvalidateDates:
            data = meshDynamic.getGridPopTimeInterval(meshTokyo, popFileName.format(date))
            data = data[:-1, :, :, :]
            trainvalidateData.append(data)
        trainvalidateData = np.concatenate(trainvalidateData, axis=0)
        print(trainvalidateData.shape)

        # trainvalidateData[trainvalidateData > MAX_VALUE] = MAX_VALUE
        trainvalidateData = trainvalidateData / MAX_VALUE
        trainModel(MODELNAME, trainvalidateData)
    else:
        ### NEW6
        print(KEYWORD, time.ctime())
        specialData = meshDynamic.getGridPopTimeInterval(meshTokyo, popFileName.format(specialDate))
        specialData = specialData[:-1, :, :, :]
        print(specialData.shape)

        # specialData[specialData > MAX_VALUE] = MAX_VALUE
        specialData = specialData / MAX_VALUE
        testModel(MODELNAME, specialData)

if __name__ == '__main__':
    main()