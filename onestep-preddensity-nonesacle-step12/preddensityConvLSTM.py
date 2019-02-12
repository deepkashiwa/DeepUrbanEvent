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
    if MODELNAME == 'CNN':
        XS = XS.swapaxes(4, 1)
        XS = np.squeeze(XS)
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
                           padding='same', return_sequences=True))
        seq.add(BatchNormalization())

        seq.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
                           padding='same', return_sequences=True))
        seq.add(BatchNormalization())

        seq.add(ConvLSTM2D(filters=CHANNEL, kernel_size=(3, 3),
                           padding='same', return_sequences=False, activation='relu'))
        return seq
    else:
        return None

def testModel(name, testData):
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

    YS_pred = YS_pred * MAX_VALUE
    YS_true = YS_true * MAX_VALUE
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
CHANNEL = 1
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
### NEW3
KEYWORD = 'onestep' + EVENT + 'tokyo' + meshTokyo.size + '5minpop-nonescale' + 'STEP' + str(TIMESTEP)
PATH = '../model' + KEYWORD
dataPATH = '../interpo_data/'
popFileName = dataPATH + '{}' + 'tokyo_' + meshTokyo.size + '_5min_pop.csv'
MAX_VALUE = 500
################### MODELNAME, DATA #######################

def main():
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

        trainvalidateData = trainvalidateData / MAX_VALUE
        trainModel(MODELNAME, trainvalidateData)
    else:
        print(KEYWORD, time.ctime())
        specialData = meshDynamic.getGridPopTimeInterval(meshTokyo, popFileName.format(specialDate))
        specialData = specialData[:-1, :, :, :]
        print(specialData.shape)

        specialData = specialData / MAX_VALUE
        testModel(MODELNAME, specialData)

if __name__ == '__main__':
    main()
