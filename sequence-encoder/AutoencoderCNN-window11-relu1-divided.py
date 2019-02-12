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

from keras.models import load_model, Model, Sequential
from keras.layers import Input, merge, TimeDistributed, Flatten, RepeatVector, Reshape, UpSampling2D, concatenate, add, Dropout, Embedding
from keras.optimizers import RMSprop, Adam, SGD
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv3D
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.core import Dense
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, LearningRateScheduler
### NEW1
from common.dataparam.Param import *

def getXSYS(allData):
    XS, YS = allData, allData
    return XS, YS

def getRealTest(allData):
    XS, YS = allData, allData
    return XS, YS

def relu1(x):
    return K.relu(x, alpha=0.0, max_value=1.0)

def getModel(name):
    input_img = Input(shape=(HEIGHT, WIDTH, CHANNEL_BEFORE))  # adapt this if using `channels_first` image data format

    x = Conv2D(64, (1, 1), activation='relu')(input_img)
    x = Conv2D(16, (1, 1), activation='relu')(x)
    encoded = Conv2D(CHANNEL_AFTER, (1, 1), activation=relu1)(x)

    x = Conv2D(16, (1, 1), activation='relu', padding='same')(encoded)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    decoded = Conv2D(CHANNEL_BEFORE, (1, 1), activation='relu', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    return autoencoder

def saveEncoderDecoder(name):
    assert os.path.exists(PATH + '/' + name + '.h5'), 'model is not existing'
    print('Model Evaluation Started ...', time.ctime())
    train_model = load_model(PATH + '/' + name + '.h5', custom_objects={'relu1':relu1})

    encode_input_img = Input(shape=(HEIGHT, WIDTH, CHANNEL_BEFORE))
    x = train_model.layers[1](encode_input_img)
    x = train_model.layers[2](x)
    train_model.layers[3]
    encoded = train_model.layers[3](x)
    encoder = Model(encode_input_img, encoded)
    encoder.compile(loss=LOSS, optimizer=OPTIMIZER)
    encoder.save(PATH + '/' + name + '_encoder.h5')

    decode_input_img = Input(shape=(HEIGHT, WIDTH, CHANNEL_AFTER))
    y = train_model.layers[4](decode_input_img)
    y = train_model.layers[5](y)
    decoded = train_model.layers[6](y)
    decoder = Model(decode_input_img, decoded)
    decoder.compile(loss=LOSS, optimizer=OPTIMIZER)
    decoder.save(PATH + '/' + name + '_decoder.h5')

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
config.gpu_options.visible_device_list = '0'
set_session(tf.Session(config=config))

meshTokyo = Mesh('tokyo', '500m')
HEIGHT = 80
WIDTH = 80
CHANNEL_BEFORE = 225
CHANNEL_AFTER = 4
### NEW2
BATCHSIZE = 4
SPLIT = 0.2
# LR = LearningRateScheduler(lambda epoch: 0.01 if epoch < 50 else 0.001)
LR = LearningRateScheduler(lambda epoch: 0.001)
EPOCH = 200
LOSS = 'mse'
OPTIMIZER = 'adam'
MAX_VALUE = 100
################# Parameter Setting #######################

################### MODELNAME, DATA #######################
# Current is One Hour, step = 12
MODELNAME = 'AutoencoderCNN'
### NEW3
KEYWORD = EVENT + 'tokyo' + meshTokyo.size + '5minencoder-max'
PATH = '../model' + KEYWORD
dataPATH = '../interpo_data/'
transitFileName = dataPATH + '{}' + 'tokyo_' + meshTokyo.size + '_5min_transit.csv'
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
            data = meshDynamic.getGridTransitTimeInterval(meshTokyo, transitFileName.format(date))
            trainvalidateData.append(data)
        trainvalidateData = np.concatenate(trainvalidateData, axis=0)
        trainvalidateData = trainvalidateData / MAX_VALUE
        print(trainvalidateData.shape)
        trainModel(MODELNAME, trainvalidateData)
    else:
        trainvalidateData = []
        for date in trainvalidateDates:
            data = meshDynamic.getGridTransitTimeInterval(meshTokyo, transitFileName.format(date))
            trainvalidateData.append(data)
        trainvalidateData = np.concatenate(trainvalidateData, axis=0)
        trainvalidateData = trainvalidateData / MAX_VALUE
        print(trainvalidateData.shape)

        saveEncoderDecoder(MODELNAME)
        encoder = load_model(PATH + '/' + MODELNAME + '_encoder.h5', custom_objects={'relu1':relu1})

        encoded_train = encoder.predict(trainvalidateData)
        print('save encoded train into .npy start', time.ctime())
        np.save(PATH + '/' + 'enTrainValidate', encoded_train)
        print('save encoded train into .npy end', time.ctime())

        ### NEW6
        print(KEYWORD, time.ctime())
        specialData = meshDynamic.getGridTransitTimeInterval(meshTokyo, transitFileName.format(specialDate))
        specialData = specialData / MAX_VALUE
        print(specialData.shape)

        encoded_special = encoder.predict(specialData)
        print('save encoded special into .npy start', time.ctime())
        np.save(PATH + '/' + 'enSpecial', encoded_special)
        print('save encoded special into .npy end', time.ctime())

        autoencoder = load_model(PATH + '/' + MODELNAME + '.h5', custom_objects={'relu1':relu1})
        MSE_train = autoencoder.evaluate(trainvalidateData, trainvalidateData)
        print('Autoencoder Keras evaluation MSE trainvalidate', MSE_train * MAX_VALUE * MAX_VALUE)
        MSE_special = autoencoder.evaluate(specialData, specialData)
        print('Autoencoder Keras evaluation MSE test', MSE_special * MAX_VALUE * MAX_VALUE)


if __name__ == '__main__':
    main()