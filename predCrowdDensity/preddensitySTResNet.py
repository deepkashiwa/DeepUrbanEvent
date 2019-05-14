import csv
import numpy as np
import os
import shutil
import sys
import time
import pandas as pd
from datetime import datetime

from STResNet import stresnet

from keras.metrics import mse
from keras.optimizers import Adam
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, LearningRateScheduler


def getXSYS(mode, dayInfo=False):
    '''
       Generate features and labels
       mode: {'train', 'test'}
       dayInfo: True/False
    '''
    '''
    dates = pd.date_range(START, END).strftime('%Y%m%d').tolist()
    startIndex, endIndex = dates.index(trainStartDate), dates.index(trainEndDate)
    trainData = allData[startIndex * DAYTIMESTEP : (endIndex + 1) * DAYTIMESTEP]
    '''
    interval_p, interval_t = 1, 7    # day interval for period/trend
    depends = [range(1, len_c + 1),
               [interval_p * DAYTIMESTEP * i for i in range(1, len_p + 1)],
               [interval_t * DAYTIMESTEP * i for i in range(1, len_t + 1)]]
    
    # mode
    if mode == 'train':
        data = np.load(dataPath + '/popTrainValidate.npy')
        start = max(len_c, interval_p * DAYTIMESTEP * len_p, interval_t * DAYTIMESTEP * len_t)
        end = data.shape[0]
        if dayInfo:
            day_info = np.genfromtxt(dataPath + '/dayInfo_onehot_%s_%s.csv' % (trainStartDate, trainEndDate), 
                                     delimiter = ',', skip_header = 1)
            print('Read-in train dayInfo dim:', day_info.shape)
            day_info_inter = []
            for row in day_info:
                for i in range(30//5):    # dayInfo within every 30 min stays same
                    day_info_inter.append(row)
            #day_info_inter = np.array(day_info_inter)
            dayInfo_dim = day_info.shape[1]
            day_feature = np.array(day_info_inter)[start:end]
        else:
            dayInfo_dim = 0
            day_feature = None
    '''
    elif mode=='test':
        data = np.load(dataPath + '/popSpecial.npy')
        start = len_c
        end = data.shape[0]
        if dayInfo:
            day_info = np.genfromtxt(dataPath + '/dayInfo_onehot_%s_%s.csv' % (testStartDate, testEndDate), 
                                     delimiter = ',', skip_header = 1)
            print('Read-in test dayInfo dim:', day_info.shape)
            day_info_intep = []
            for row in day_info:
                for i in range(30//5):    # dayInfo within every 30 min stays same
                    day_info_intep.append(row)
            day_info_intep = np.array(day_info_intep)
            dayInfo_dim = day_info.shape[1]
            day_feature = day_info_intep[start:end]
        else:
            dayInfo_dim = 0
            day_feature = None
    else:
        assert False, 'Invalid mode...'
    '''
    XC, XP, XT = [], [], []
    for i in range(start, end):
        x_c = [data[i - j] for j in depends[0]]
        x_p = [data[i - j] for j in depends[1]]
        x_t = [data[i - j] for j in depends[2]]
        if len_c > 0:
            XC.append(np.dstack(x_c))
        if len_p > 0:
            XP.append(np.dstack(x_p))
        if len_t > 0:
            XT.append(np.dstack(x_t))
        
    #XC, XP, XT = np.array(XC), np.array(XP), np.array(XT)
    #print(mode, 'X', XC.shape, XP.shape, XT.shape, day_feature.shape)
    #XS = [XC, XP, XT, day_feature] if day_feature is not None else [XC, XP, XT]
    XS = [x for x in [XC, XP, XT, day_feature] if x != []]
    print('train X')
    for item in XS:
        print(np.array(item).shape)
    YS = data[start:end]
    print(mode, 'Y', YS.shape)
    
    return XS, YS, dayInfo_dim

def getRealTest(data, dayInfo):
    if dayInfo:
        day_info = np.genfromtxt(dataPath + '/dayInfo_onehot_%s_%s.csv' % (testStartDate, testEndDate), 
                                 delimiter = ',', skip_header = 1)
        day_info_inter = []
        for row in day_info:
            for i in range(30//5):
                day_info_inter.append(row)
        dayInfo_dim = day_info.shape[1]
        day_feature = np.array(day_info_inter)
        print('Test dayInfo dim', dayInfo_dim, day_feature.shape)
    XC, YS, DAY = [], [], []
    for i in range(data.shape[0] - TIMESTEP*2):
        x = data[i:i+TIMESTEP]
        y = data[i+TIMESTEP:i+TIMESTEP*2]
        day = day_feature[i+TIMESTEP:i+TIMESTEP*2]
        XC.append(x), YS.append(y), DAY.append(day)
    #XC = np.array(XC).swapaxes(4, 1)
    #XC = np.squeeze(XC)
    XS = [np.array(XC), np.array(DAY)]
    return XS, np.array(YS)


def getModel(modelName, residual_units, dayInfo_dim):
    if modelName == 'STResNet':
        # format input
        c_dim = (len_c, map_height, map_width, nb_channel) if len_c > 0 else None
        p_dim = (len_c, map_height, map_width, nb_channel) if len_p > 0 else None
        t_dim = (len_c, map_height, map_width, nb_channel) if len_t > 0 else None
        print('Input dimensions:', c_dim, p_dim, t_dim, dayInfo_dim)
        
        # call the model
        model = stresnet(c_dim = c_dim, p_dim = p_dim, t_dim = t_dim,
                         residual_units = residual_units, dayInfo_dim = dayInfo_dim)
        model.summary()
        # from keras.utils import plot_model
        # plot_model(model, to_file = modelPath+'/model.png', show_shapes = True)
        return model
    else:
        print("Model name can't be recognized...")
        return None


def trainModel(model, trainX, trainY):
    print('*' * 10, 'model training started ...', time.ctime(), '*' * 10)    
    csv_logger = CSVLogger(logPath + '/epochLog.log')
    #early_stopper = EarlyStopping(monitor='val_loss', patience=10, verbose = 1, mode='auto')
    model_checkpointer = ModelCheckpoint(modelPath + '/temp' + '/modelBest.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    
    #print('trainX dim:', trainX.shape, 'trainY dim:', trainY.shape)
    # train
    history = model.fit(trainX, trainY,
                        batch_size = BATCHSIZE,
                        epochs = EPOCHS,
                        validation_split = SPLIT,
                        callbacks = [LR, csv_logger, model_checkpointer],
                        # shuffle = True,
                        verbose = 1)
    model.save_weights(modelPath + '/temp' + '/trained_weights.h5', overwrite = True)
    print('*' * 10, 'model training ended ...', time.ctime(), '*' * 10)

        
def testModel(model, testX, testY):
    with open(logPath + '/prediction_scores.txt', 'a') as wf:
        wf.write(','.join(['*' * 10, 'model testing started ...', time.ctime(), '*' * 10]) + '\n')
        print('*' * 10, 'model testing started ...', time.ctime(), '*' * 10)
    
    assert os.path.exists(modelPath + '/temp' + '/modelBest.h5'), 'Model is not existing...'
    model.load_weights(modelPath + '/temp' + '/modelBest.h5')
    
    f = open(logPath + '/prediction_scores.txt', 'a')
    print('*' * 40)
    f.write('*'*20 + 'Hyper parameter' + '*'*20 + '\n')
    
    # sequence to sequence prediction
    YS_pred_list = []
    for i in range(TIMESTEP):
        # preprocess XS
        XC, DAY = testX[0], testX[1]
        if i > 0:
            XC = np.concatenate((XC[:, i:, :, :, :], YS_pred), axis = 1)
        else:
            pass
        XC = XC.swapaxes(1, 4)
        XC = np.squeeze(XC)
        DAY = np.squeeze(DAY[:, i:i+1, :])
        curXS = [XC, DAY]
        
        stepYS = model.predict(curXS, BATCHSIZE)
        stepYS = stepYS[:, np.newaxis, :, :, :]
        YS_pred_list.append(stepYS)
        YS_pred = np.concatenate(YS_pred_list, axis = 1)    # generate YS_pred
        
        stepYS_true = testY[:, i:i+1, :, :, :]
        print(stepYS.shape, stepYS_true.shape)
        assert stepYS.shape == stepYS_true.shape    # (N, 1, 80, 80, 1)
        MSE_cur_step = np.mean((stepYS - stepYS_true)**2)
        f.write("Model MSE for step %d, %f\n" % (i, MSE_cur_step))
        print("Model MSE for step", i, MSE_cur_step)
        
    # compute MSE for whole test set
    print(YS_pred.shape, testY.shape)
    assert YS_pred.shape == testY.shape    # (N, 6, 80, 80, 1)
    MSE_multisteps = np.mean((YS_pred - testY)**2)
    f.write("Model MSE for multisteps, %f\n" % MSE_multisteps)
    print('MSE for multisteps', MSE_multisteps)
    
    # compute rescaled MSE
    YS_pred = YS_pred * 500
    YS_true = testY * 500
    for i in range(TIMESTEP):
        stepYS_pred = YS_pred[:, i:i+1, :, :, :]
        stepYS_true = YS_true[:, i:i+1, :, :, :]
        MSE_cur_step = np.mean((stepYS_pred - stepYS_true)**2)
        f.write("Model rescaled(*500) MSE for step, %d, %f\n" % (i, MSE_cur_step))
        print("Model rescaled(*500) MSE for step", i, MSE_cur_step)
    MSE_multisteps = np.mean((YS_pred - YS_true) ** 2)
    f.write("Model rescaled(*500) MSE for multisteps, %f\n" % MSE_multisteps)
    print('MSE rescaled(*500) for multisteps', MSE_multisteps)
    
    f.close()
    
    '''
    # evaluate on test set
    scores = model.evaluate(testX, testY, BATCHSIZE *2, verbose = 1)
    predY = model.predict(testX, BATCHSIZE *2, verbose = 1)
    testY, predY = normal.denormalize(testY), normal.denormalize(predY)
    print('testY range:', testY.min(), testY.max())
    print('predY range:', predY.min(), predY.max())
    
    # Metrics
    MSE = scores[3]*((normal.dmax - normal.dmin)/2)**2
    
    model.save_weights(modelPath + '/temp' + '/tested_weights.h5', overwrite = True)
    
    print("Model scaled MSE", scores[0])
    print("Model rescaled MSE", MSE)
    with open(logPath + '/prediction_scores.txt', 'a') as wf:
        wf.write("Model scaled MSE, %.6f\n" % scores[0])
        wf.write("Model rescaled MSE, %.6f\n" % MSE)
        
    with open(logPath + '/prediction_scores.txt', 'a') as wf:
        wf.write(','.join(['*' * 10, 'model testing ended ...', time.ctime(), '*' * 10]) + '\n')
        print('*' * 10, 'model testing ended ...', time.ctime(), '*' * 10)
    '''

        
################# Parameter Settings #######################

dataPath = './E1_Earthquake'
START, END = '20110301', '20110311'
trainStartDate, trainEndDate = '20110301', '20110310'
testStartDate, testEndDate = '20110311', '20110311'

modelPath = './model'
logPath = './log'
modelName = 'STResNet'
dayInfo = True
DAYTIMESTEP = 24 * 12    # 5 min time interval
map_height, map_width = 80, 80
nb_channel = 1    # population density
len_c, len_p, len_t = 6, 0, 0    # seq of 6 closeness
nb_residual_units = 1

TIMESTEP = 6    # test sequence length
LOSS = 'mse'
OPTIMIZER = 'adam'
METRICS = [mse]
LR = LearningRateScheduler(lambda epoch: 0.0001)
BATCHSIZE = 4
EPOCHS = 200
SPLIT = 0.2


################## Parameter Settings #######################


def main():
    # reproducibility
    np.random.seed(100)
    import random
    random.seed(100)
    from tensorflow import set_random_seed
    set_random_seed(100)
    os.environ['PYTHONHASHSEED'] = '0'
    
    # set GPU session
    import tensorflow as tf
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    # session_conf.gpu_options.per_process_gpu_memory_fraction = 0.5
    session_conf.gpu_options.allow_growth = True
    session_conf.gpu_options.visible_device_list = '7'
    tf.set_random_seed(100)
    session = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    from keras import backend as K
    K.set_session(session)
    
    # set log path
    if not os.path.exists(logPath):
        print('Making a folder for log output...')
        os.makedirs(os.path.dirname(__file__) + logPath)
    
    # get features
    trainX, trainY, dayInfo_dim = getXSYS('train', dayInfo)
    #print('trainXY shapes:', trainX.shape, trainY.shape)
    testData = np.load(dataPath + '/popSpecial.npy')
    XS, YS = getRealTest(testData, dayInfo)
    print('Test data dims:', XS[0].shape, XS[1].shape, YS.shape)
    
    # get model
    model = getModel(modelName, nb_residual_units, dayInfo_dim)
    model.compile(loss = LOSS, optimizer = OPTIMIZER, metrics = METRICS)
    print('Model compiled...')
    
    # train model
    trainModel(model, trainX, trainY)
    
    # test model
    testModel(model, XS, YS)
    

if __name__ == '__main__':
    main()


