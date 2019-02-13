import csv
import numpy as np
import os
import shutil
import sys
import time
from datetime import datetime

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from common.datastructure.Mesh import Mesh
from meshdynamic import meshDynamic

from keras.models import load_model, Model, Sequential
from keras.layers import Input, merge, TimeDistributed, Flatten, RepeatVector, Reshape, UpSampling2D, concatenate, add, Dropout, Embedding
from keras.optimizers import RMSprop, Adam, SGD
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv3D
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.core import Dense
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, LearningRateScheduler
from common.dataparam.Param import *

def getXSYS(allData):
    XS, YS = [], []
    for i in range(allData.shape[0] - TIMESTEP - TIMESTEP):
        x = allData[i:i+TIMESTEP, :, :, :]
        y = allData[i+TIMESTEP, :, :, :]
        XS.append(x), YS.append(y)
    XS, YS = np.array(XS), np.array(YS)
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
    seq = Sequential()
    seq.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
                       input_shape=(None, HEIGHT, WIDTH, CHANNEL),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(ConvLSTM2D(filters=CHANNEL, kernel_size=(3, 3),
                       padding='same', return_sequences=False, activation='relu'))
    return seq

def testModel(name, testData, originalData):
    assert os.path.exists(PATH + '/' + name + '.h5'), 'model is not existing'
    print('Model Evaluation Started ...', time.ctime())

    train_model = load_model(PATH + '/'+ name + '.h5')
    train_model.summary()

    f = open(PATH + '/' + name + '_prediction_scores.txt', 'a')
    print('*' * 40)
    f.write('*' * 40 + '\n')

    XS, YS_true = getRealTest(testData)
    YS_pred_list = []
    for i in range(TIMESTEP):
        if i > 0:
            currentXS = np.concatenate((XS[:, i:, :, :, :], YS_pred), axis=1)
        else:
            currentXS = XS

        stepYS = train_model.predict(currentXS, BATCHSIZE)
        stepYS = stepYS[:, np.newaxis, :, :, :]
        YS_pred_list.append(stepYS)
        YS_pred = np.concatenate(YS_pred_list, axis=1)

        stepYS_true = YS_true[:, i:i + 1, :, :, :]
        assert stepYS.shape == stepYS_true.shape
        MSE_cur_onestep = np.mean((stepYS - stepYS_true) ** 2)
        f.write("Model MSE for multi step, %d, %f\n" % (i, MSE_cur_onestep))
        print("Model MSE for multi step", i, MSE_cur_onestep)

    assert YS_pred.shape == YS_true.shape
    MSE_multistep = np.mean((YS_pred - YS_true) ** 2)
    f.write("Model MSE for multi step, %f\n" % MSE_multistep)
    print('MSE for multi step', MSE_multistep)

    XS_original, YS_original = getRealTest(originalData)

    decoder = load_model(dataPATH + '/AutoencoderCNN_decoder.h5')
    YS_pred = YS_pred.reshape(-1, YS_pred.shape[2], YS_pred.shape[3], YS_pred.shape[4])
    YS_pred_decoded = decoder.predict(YS_pred) * MAX_VALUE
    print('YS_pred, YS_pred_decoded', YS_pred.shape, YS_pred_decoded.shape)
    YS_pred_decoded = YS_pred_decoded.reshape(-1, TIMESTEP, YS_pred_decoded.shape[1],
                                              YS_pred_decoded.shape[2], YS_pred_decoded.shape[3])

    assert YS_original.shape == YS_pred_decoded.shape
    for i in range(TIMESTEP):
        pred = YS_pred_decoded[:, i:i + 1, :, :, :]
        true = YS_original[:, i:i + 1, :, :, :]
        MSE_cur_onestep = np.mean((pred - true) ** 2)
        f.write("Model MSE for multi step original & decoded * MAXVALUE, %d, %f\n" % (i, MSE_cur_onestep))
        print("Model MSE for multi step original & decoded * MAXVALUE", i, MSE_cur_onestep)
    MSE_multistep = np.mean((YS_pred_decoded - YS_original) ** 2)
    f.write("Model MSE for multi step original & decoded * MAXVALUE, %f\n" % MSE_multistep)
    print('MSE for multi step original & decoded * MAXVALUE', MSE_multistep)
    ##### new added for flow ########

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
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.visible_device_list = '0'
set_session(tf.Session(config=config))

meshTokyo = Mesh('tokyo', '500m')
HEIGHT = 80
WIDTH = 80
### NEW2
CHANNEL = 4
BATCHSIZE = 4
print('BATCHSIZE = 4, lr = 0.0001')
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
KEYWORD = 'onestep' + EVENT + 'tokyo' + meshTokyo.size + '5mintransit-nonescale'
PATH = '../model' + KEYWORD
dataPATH = '../interpo_data/'
transitFileName = dataPATH + '{}' + 'tokyo_' + meshTokyo.size + '_5min_transit.csv'
MAX_VALUE = 100
################### MODELNAME, DATA #######################

def main():
    if not os.path.exists(PATH):
        os.makedirs(PATH + '/log')
    currentPython = sys.argv[0]
    shutil.copy2(currentPython, PATH)

    if not os.path.exists(PATH + '/' + MODELNAME + '.h5'):
        trainvalidateData = np.load(dataPATH + 'enTrainValidate.npy')
        print(trainvalidateData.shape)
        trainModel(MODELNAME, trainvalidateData)
    else:
        ############### Modified #####################
        specialData = np.load(dataPATH + 'enSpecial.npy')
        print(specialData.shape)

        ### NEW4
        print(KEYWORD, time.ctime())
        specialOriginalData = meshDynamic.getGridTransitTimeInterval(meshTokyo, transitFileName.format(specialDate))
        print(specialOriginalData.shape)

        testModel(MODELNAME, specialData, specialOriginalData)
        ############### Modified #####################

if __name__ == '__main__':
    main()
