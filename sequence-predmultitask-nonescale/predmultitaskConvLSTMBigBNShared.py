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

import keras
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
# NEW1
from common.dataparam.Param import *

def getXSYS_density(allData):
    XS, YS = [], []
    for i in range(allData.shape[0] - TIMESTEP - TIMESTEP):
        x = allData[i:i+TIMESTEP, :, :, :]
        y = allData[i + TIMESTEP:i + TIMESTEP + TIMESTEP, :, :, :]
        XS.append(x), YS.append(y)
    XS, YS = np.array(XS), np.array(YS)
    return XS, YS

def getXSYS_flow(allData):
    XS, YS = [], []
    for i in range(allData.shape[0] - TIMESTEP - TIMESTEP):
        x = allData[i:i+TIMESTEP, :, :, :]
        y = allData[i + TIMESTEP:i + TIMESTEP + TIMESTEP, :, :, :]
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
    if name == 'ConvLSTMBigBNShared':
        input_density = Input(shape=(TIMESTEP, HEIGHT, WIDTH, CHANNEL_DENSITY))
        input_flow = Input(shape=(TIMESTEP, HEIGHT, WIDTH, CHANNEL_FLOW))
        density = ConvLSTM2D(32, (3, 3), padding='same', return_sequences=True)(input_density)
        flow = ConvLSTM2D(32, (3, 3), padding='same', return_sequences=True)(input_flow)
        density = BatchNormalization()(density)
        flow = BatchNormalization()(flow)
        merged = keras.layers.concatenate([density, flow])
        merged = ConvLSTM2D(32, (3, 3), padding='same', return_sequences=False)(merged)
        merged = BatchNormalization()(merged)
        merged = Lambda(lambda x: K.concatenate([x[:, np.newaxis, :, :, :]] * TIMESTEP, axis=1))(merged)
        merged = ConvLSTM2D(32, (3, 3), padding='same', return_sequences=True)(merged)
        merged = BatchNormalization()(merged)
        output_density = ConvLSTM2D(CHANNEL_DENSITY, (3, 3), activation='relu',
                                    padding='same', return_sequences=True, name='density')(merged)
        output_flow = ConvLSTM2D(CHANNEL_FLOW, (3, 3), activation='relu',
                                 padding='same', return_sequences=True, name='flow')(merged)

        multitask = Model(inputs=[input_density, input_flow], outputs=[output_density, output_flow])
        return multitask
    else:
        return None

def testModel(name, testData_density, testData_flow, originalData_flow):
    assert os.path.exists(PATH + '/' + name + '.h5'), 'model is not existing'
    print('Model Evaluation Started ...', time.ctime())
    train_model = load_model(PATH + '/'+ name + '.h5', custom_objects={'TIMESTEP':TIMESTEP})
    train_model.summary()

    XS_density, YS_density = getRealTest(testData_density)
    XS_flow, YS_flow = getRealTest(testData_flow)

    # for input density ConvLSTM
    # for input flow ConvLSTM

    YS_pred_density, YS_pred_flow= train_model.predict([XS_density, XS_flow])

    # for output density ConvLSTM
    # for output flow ConvLSTM

    assert YS_density.shape == YS_pred_density.shape and YS_flow.shape == YS_pred_flow.shape

    f = open(PATH + '/' + name + '_prediction_scores.txt', 'a')
    print('*' * 40)
    f.write('*' * 40 + '\n')

    YS_pred_density = MAX_VALUE_DENSITY * YS_pred_density
    YS_density = MAX_VALUE_DENSITY * YS_density
    YS_pred_flow = YS_pred_flow
    YS_flow = YS_flow

    for i in range(TIMESTEP):
        pred = YS_pred_density[:, i:i + 1, :, :, :]
        true = YS_density[:, i:i + 1, :, :, :]
        MSE_onestep_density = np.mean((pred - true) ** 2)
        f.write("Model MSE for multi step 500*Array density, %d, %f\n" % (i, MSE_onestep_density))
        print("Model MSE for multi step 500*Array density", i, MSE_onestep_density)
    MSE_multistep_density = np.mean((YS_pred_density - YS_density) ** 2)
    f.write("Model MSE for multi step 500*Array density, %f\n" % MSE_multistep_density)
    print('MSE for multi step 500*Array density', MSE_multistep_density)

    ##### new added for flow ########
    for i in range(TIMESTEP):
        pred = YS_pred_flow[:, i:i + 1, :, :, :]
        true = YS_flow[:, i:i + 1, :, :, :]
        MSE_cur_onestep = np.mean((pred - true) ** 2)
        f.write("Model MSE for multi step encodedYS & encodedYS_pred, %d, %f\n" % (i, MSE_cur_onestep))
        print("Model MSE for multi step encodedYS & encodedYS_pred", i, MSE_cur_onestep)
    MSE_multistep = np.mean((YS_pred_flow - YS_flow) ** 2)
    f.write("Model MSE for multi step encodedYS & encodedYS_pred, %f\n" % MSE_multistep)
    print('MSE for multi step encodedYS & encodedYS_pred', MSE_multistep)

    XS_flow_original, YS_flow_original = getRealTest(originalData_flow)

    decoder = load_model(dataPATH + '/AutoencoderCNN_decoder.h5')
    YS_pred_flow = YS_pred_flow.reshape(-1, YS_pred_flow.shape[2], YS_pred_flow.shape[3], YS_pred_flow.shape[4])
    YS_pred_flow_decoded = decoder.predict(YS_pred_flow) * MAX_VALUE_FLOW
    print('YS_pred_flow, YS_pred_flow_decoded', YS_pred_flow.shape, YS_pred_flow_decoded.shape)
    YS_pred_flow_decoded = YS_pred_flow_decoded.reshape(-1, TIMESTEP, YS_pred_flow_decoded.shape[1],
                                              YS_pred_flow_decoded.shape[2], YS_pred_flow_decoded.shape[3])

    assert YS_flow_original.shape == YS_pred_flow_decoded.shape
    for i in range(TIMESTEP):
        pred = YS_pred_flow_decoded[:, i:i + 1, :, :, :]
        true = YS_flow_original[:, i:i + 1, :, :, :]
        MSE_cur_onestep = np.mean((pred - true) ** 2)
        f.write("Model MSE for multi step original & decoded * MAXVALUE, %d, %f\n" % (i, MSE_cur_onestep))
        print("Model MSE for multi step original & decoded * MAXVALUE", i, MSE_cur_onestep)
    MSE_multistep = np.mean((YS_pred_flow_decoded - YS_flow_original) ** 2)
    f.write("Model MSE for multi step original & decoded * MAXVALUE, %f\n" % MSE_multistep)
    print('MSE for multi step original & decoded * MAXVALUE', MSE_multistep)
    ##### new added for flow ########

    f.close()

def trainModel(name, trainvalidateData_density, trainvalidateData_flow):
    print('Model Training Started ...', time.ctime())
    trainXS_density, trainYS_density = getXSYS_density(trainvalidateData_density)
    print(trainXS_density.shape, trainYS_density.shape)

    trainXS_flow, trainYS_flow = getXSYS_flow(trainvalidateData_flow)
    print(trainXS_flow.shape, trainYS_flow.shape)

    model = getModel(name)
    model.compile(loss=LOSS, loss_weights={'density': 0.5, 'flow': 0.5}, optimizer=OPTIMIZER)
    model.summary()
    csv_logger = CSVLogger(PATH + '/log' + '/' + name + '.log')
    checkpointer = ModelCheckpoint(filepath=PATH + '/' + name + '.h5', verbose=1, save_best_only=True)
    model.fit(x=[trainXS_density, trainXS_flow], y=[trainYS_density, trainYS_flow],
              batch_size=BATCHSIZE, epochs=EPOCH, shuffle=True,
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
CHANNEL_FLOW = 4
CHANNEL_DENSITY = 1
### NEW2
BATCHSIZE = 4
SPLIT = 0.2
# LR = LearningRateScheduler(lambda epoch: 0.01 if epoch < 50 else 0.001)
LR = LearningRateScheduler(lambda epoch: 0.0001)
print('BatchSize = 4, LR = 0.0001')
EPOCH = 200
LOSS = 'mse'
OPTIMIZER = 'adam'
################# Parameter Setting #######################

################### MODELNAME, DATA #######################
# Current is One Hour, step = 12
MODELNAME = 'ConvLSTMBigBNShared'
### NEW3
KEYWORD = EVENT + 'tokyo' + meshTokyo.size + '5minmultitask-nonescale'
PATH = '../model' + KEYWORD
dataPATH = '../interpo_data/'
# popFileName = dataPATH + '{}' + 'tokyo_' + meshTokyo.size + '_5min_pop.csv'
transitFileName = dataPATH + '{}' + 'tokyo_' + meshTokyo.size + '_5min_transit.csv'
MAX_VALUE_FLOW = 100
MAX_VALUE_DENSITY = 500
################### MODELNAME, DATA #######################

def main():
    if not os.path.exists(PATH):
        os.makedirs(PATH + '/log')
    currentPython = sys.argv[0]
    shutil.copy2(currentPython, PATH)

    print(MODELNAME)
    if not os.path.exists(PATH + '/' + MODELNAME + '.h5'):
        trainvalidateData_pop = np.load(dataPATH + 'popTrainValidate.npy')
        trainvalidateData_flow = np.load(dataPATH + 'enTrainValidate.npy')
        print(trainvalidateData_pop.shape, trainvalidateData_flow.shape)
        trainModel(MODELNAME, trainvalidateData_pop, trainvalidateData_flow)
    else:
        #########################
        specialData_pop = np.load(dataPATH + 'popSpecial.npy')
        specialData_flow = np.load(dataPATH + 'enSpecial.npy')

        ### NEW4
        print(EVENT, time.ctime())
        ### NEW5
        specialOriginalData_flow = meshDynamic.getGridTransitTimeInterval(meshTokyo, transitFileName.format(specialDate))
        print(specialData_pop.shape, specialData_flow.shape, specialOriginalData_flow.shape)
        testModel(MODELNAME, specialData_pop, specialData_flow, specialOriginalData_flow)
        #########################

if __name__ == '__main__':
    main()
