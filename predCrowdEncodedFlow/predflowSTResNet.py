import csv
import numpy as np
import os
import shutil
import sys
import time
from datetime import datetime

from keras.models import load_model, Model, Sequential
from keras.layers import Input, merge, TimeDistributed, Flatten, RepeatVector, Reshape, UpSampling2D, concatenate, add, Dropout, Embedding, Activation
from keras.optimizers import RMSprop, Adam, SGD
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv3D
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.core import Dense
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, LearningRateScheduler

def getXSYS(allData):
    XS, YS = [], []
    for i in range(allData.shape[0] - TIMESTEP):
        x = allData[i:i+TIMESTEP]
        y = allData[i+TIMESTEP]
        XS.append(x), YS.append(y)
    XS, YS = np.array(XS), np.array(YS)
    if MODELNAME == 'CNN' or MODELNAME == 'STResNet':
        XS = np.expand_dims(XS, axis=-2)    # XS = XS[:, :, :, :, np.newaxis, :]
        XS = np.squeeze(XS.swapaxes(-2, 1))
        XS = XS.reshape(XS.shape[0],XS.shape[1],XS.shape[2],-1)
    elif MODELNAME == 'ConvLSTM':
        pass
    elif MODELNAME == 'PredNet':
        pass
    else:
        print('Not a valid model...')
        pass
    return XS, YS

def getRealTest(allData):
    XS, YS = [], []
    for i in range(allData.shape[0] - TIMESTEP*2):
        x = allData[i:i+TIMESTEP]
        y = allData[i+TIMESTEP:i+TIMESTEP*2]    # seq to seq prediction
        XS.append(x), YS.append(y)
    XS, YS = np.array(XS), np.array(YS)
    return XS, YS

def _residual_unit():
    def f(input):
        residual = BatchNormalization()(input)
        residual = Activation('relu')(residual)
        residual = Conv2D(32, (3, 3), padding='same')(residual)
        residual = BatchNormalization()(residual)
        residual = Activation('relu')(residual)
        residual = Conv2D(32, (3, 3), padding='same')(residual)
        return add([input, residual])
    return f

def res_units(repetations):
    def f(input):
        for i in range(repetations):
            input = _residual_unit()(input)
        return input
    return f

def getModel(name):
    if name == 'STResNet':
        input = Input(shape=(HEIGHT, WIDTH, TIMESTEP*CHANNEL))
        x = Conv2D(32, (3, 3), padding='same', input_shape=(HEIGHT, WIDTH, TIMESTEP*CHANNEL))(input)
        x = res_units(nb_res_units)(x)
        x = Activation('relu')(x)
        output = Conv2D(CHANNEL, (3, 3), activation='relu', padding='same')(x)
        model = Model(inputs=input, outputs=output)
        return model
    else:
        return None

def trainModel(name, trainData):
    print('Model Training Started ...', time.ctime())
    trainXS, trainYS = getXSYS(trainData)
    print('TrainX dim:', trainXS.shape, 'TrainY dim:', trainYS.shape)

    model = getModel(name)
    model.compile(loss=LOSS, optimizer=OPTIMIZER)
    model.summary()
    csv_logger = CSVLogger(PATH + '/' + name + '.log')
    checkpointer = ModelCheckpoint(filepath=PATH + '/' + name + '.h5', verbose=1, save_best_only=True)
    model.fit(trainXS, trainYS, batch_size=BATCHSIZE, epochs=EPOCH, shuffle=True,
              callbacks=[csv_logger, checkpointer, LR], validation_split=SPLIT)
    print('Model Training Ended ', time.ctime())
    
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

        currentXS = np.expand_dims(currentXS, axis = -2)
        currentXS = np.squeeze(currentXS.swapaxes(-2, 1))
        inputXS = currentXS.reshape(currentXS.shape[0], currentXS.shape[1], currentXS.shape[2], -1)
        
        stepYS = train_model.predict(inputXS)
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

    f.close()

################### Hyper Parameters #####################
TIMESTEP = 12
HEIGHT = 80
WIDTH = 80
CHANNEL = 4

MODELNAME = 'STResNet'
nb_res_units = 1

BATCHSIZE = 4
SPLIT = 0.2
LR = LearningRateScheduler(lambda epoch: 0.0001)
EPOCH = 200
LOSS = 'mse'
OPTIMIZER = 'adam'

PATH = './flow_log5'
dataPATH = './E3_NewYearDay'
MAX_VALUE = 100

##########################################################

def main():
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.gpu_options.visible_device_list = '5'
    set_session(tf.Session(config=config))
    
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    
    if not os.path.exists(PATH + '/' + MODELNAME + '.h5'):
        print(time.ctime())
        trainvalidateData = np.load(dataPATH + '/enTrainValidate.npy')
        print(trainvalidateData.shape)
        
        trainModel(MODELNAME, trainvalidateData)
    else:
        print(time.ctime())
        specialData = np.load(dataPATH + '/enSpecial.npy')
        specialOriginalData = np.load(dataPATH + '/originalFlowSpecial.npy')
        print(specialData.shape, specialOriginalData.shape)
        
        testModel(MODELNAME, specialData, specialOriginalData)

###########################################################

if __name__ == '__main__':
    main()

