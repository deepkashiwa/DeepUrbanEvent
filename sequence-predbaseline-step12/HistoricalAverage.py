import numpy as np
import os
import sys
import shutil
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from common.dataparam.Param import *
from common.datastructure.Mesh import Mesh
from meshdynamic import meshDynamic

################# Parameter Setting #######################
dataPATH = '../interpo_data/'
meshTokyo = Mesh('tokyo', '500m')
popFileName = dataPATH + '{}' + 'tokyo_' + meshTokyo.size + '_5min_pop.csv'
transitFileName = dataPATH + '{}' + 'tokyo_' + meshTokyo.size + '_5min_transit.csv'
MODELNAME = 'HA'
KEYWORD = EVENT + 'tokyo' + meshTokyo.size + '5minbaseline-nonescale' + 'STEP' + str(TIMESTEP)
PATH = '../model' + KEYWORD
################# Parameter Setting #######################

if not os.path.exists(PATH):
    os.makedirs(PATH)
currentPython = sys.argv[0]
shutil.copy2(currentPython, PATH)

def getXSYS(allData):
    XS, YS = [], []
    for i in range(allData.shape[0] - TIMESTEP - TIMESTEP):
        x = allData[i:i+TIMESTEP, :, :, :]
        y = allData[i + TIMESTEP:i + TIMESTEP + TIMESTEP, :, :, :]
        XS.append(x), YS.append(y)
    XS, YS = np.array(XS), np.array(YS)
    return XS, YS

# def HA_pop(name):
#     print(EVENT, specialDate)
#
#     trainvalidateData = []
#     for date in trainvalidateDates:
#         data = meshDynamic.getGridPopTimeInterval(meshTokyo, popFileName.format(date))
#         data = data[:-1, :, :, :]
#         trainvalidateData.append(data)
#     ha = np.mean(trainvalidateData, axis=0)
#     print(ha.shape)
#
#     specialData = meshDynamic.getGridPopTimeInterval(meshTokyo, popFileName.format(specialDate))
#     specialData = specialData[:-1, :, :, :]
#     print(specialData.shape)
#
#     XS_ha, YS_ha = getXSYS(ha)
#     XS, YS = getXSYS(specialData)
#
#     assert YS_ha.shape == YS.shape
#
#     f = open(PATH + '/' + name + '_prediction_scores_pop.txt', 'a')
#     print('*' * 40)
#     f.write('*' * 40 + '\n')
#     for i in range(TIMESTEP):
#         pred = YS_ha[:, i:i + 1, :, :, :]
#         true = YS[:, i:i + 1, :, :, :]
#         MSE_cur_onestep = np.mean((pred - true) ** 2)
#         f.write("Model MSE for multi Density Array, %d, %f\n" % (i, MSE_cur_onestep))
#         print("Model MSE for multi Density Array", i, MSE_cur_onestep)
#     MSE_multistep = np.mean((YS_ha - YS) ** 2)
#     f.write("Model MSE for multi Density Array, %f\n" % MSE_multistep)
#     print('MSE for multi step Density Array', MSE_multistep)
#
# def HA_flow(name):
#     print(EVENT, specialDate)
#
#     trainvalidateData = []
#     for date in trainvalidateDates:
#         data = meshDynamic.getGridTransitTimeInterval(meshTokyo, transitFileName.format(date))
#         trainvalidateData.append(data)
#     ha = np.mean(trainvalidateData, axis=0)
#     print(ha.shape)
#
#     specialData = meshDynamic.getGridTransitTimeInterval(meshTokyo, transitFileName.format(specialDate))
#     print(specialData.shape)
#
#     XS_ha, YS_ha = getXSYS(ha)
#     XS, YS = getXSYS(specialData)
#
#     assert YS_ha.shape == YS.shape
#
#     f = open(PATH + '/' + name + '_prediction_scores_transit.txt', 'a')
#     print('*' * 40)
#     f.write('*' * 40 + '\n')
#     for i in range(TIMESTEP):
#         pred = YS_ha[:, i:i + 1, :, :, :]
#         true = YS[:, i:i + 1, :, :, :]
#         MSE_cur_onestep = np.mean((pred - true) ** 2)
#         f.write("Model MSE for multi Flow Array, %d, %f\n" % (i, MSE_cur_onestep))
#         print("Model MSE for multi Flow Array", i, MSE_cur_onestep)
#     MSE_multistep = np.mean((YS_ha - YS) ** 2)
#     f.write("Model MSE for multi Flow Array, %f\n" % MSE_multistep)
#     print('MSE for multi step Flow Array', MSE_multistep)

def HA_pop(name):
    print(EVENT, specialDate)

    selectedDates = ['20110219', '20110220', '20110226']

    trainvalidateData = []
    for date in selectedDates:
        data = meshDynamic.getGridPopTimeInterval(meshTokyo, popFileName.format(date))
        data = data[:-1, :, :, :]
        trainvalidateData.append(data)
    ha = np.mean(trainvalidateData, axis=0)
    print(ha.shape)

    specialData = meshDynamic.getGridPopTimeInterval(meshTokyo, popFileName.format(specialDate))
    specialData = specialData[:-1, :, :, :]
    print(specialData.shape)

    XS_ha, YS_ha = getXSYS(ha)
    XS, YS = getXSYS(specialData)

    assert YS_ha.shape == YS.shape

    f = open(PATH + '/' + name + '_prediction_scores_pop.txt', 'a')
    print('*' * 40 + 'Weekend')
    f.write('*' * 40 + 'Weekend' + '\n')
    for i in range(TIMESTEP):
        pred = YS_ha[:, i:i + 1, :, :, :]
        true = YS[:, i:i + 1, :, :, :]
        MSE_cur_onestep = np.mean((pred - true) ** 2)
        f.write("Model MSE for multi Density Array, %d, %f\n" % (i, MSE_cur_onestep))
        print("Model MSE for multi Density Array", i, MSE_cur_onestep)
    MSE_multistep = np.mean((YS_ha - YS) ** 2)
    f.write("Model MSE for multi Density Array, %f\n" % MSE_multistep)
    print('MSE for multi step Density Array', MSE_multistep)

def HA_flow(name):
    print(EVENT, specialDate)

    selectedDates = ['20110219', '20110220', '20110226']

    trainvalidateData = []
    for date in selectedDates:
        data = meshDynamic.getGridTransitTimeInterval(meshTokyo, transitFileName.format(date))
        trainvalidateData.append(data)
    ha = np.mean(trainvalidateData, axis=0)
    print(ha.shape)

    specialData = meshDynamic.getGridTransitTimeInterval(meshTokyo, transitFileName.format(specialDate))
    print(specialData.shape)

    XS_ha, YS_ha = getXSYS(ha)
    XS, YS = getXSYS(specialData)

    assert YS_ha.shape == YS.shape

    f = open(PATH + '/' + name + '_prediction_scores_transit.txt', 'a')
    print('*' * 40 + 'Weekend')
    f.write('*' * 40 + 'Weekend' + '\n')
    for i in range(TIMESTEP):
        pred = YS_ha[:, i:i + 1, :, :, :]
        true = YS[:, i:i + 1, :, :, :]
        MSE_cur_onestep = np.mean((pred - true) ** 2)
        f.write("Model MSE for multi Flow Array, %d, %f\n" % (i, MSE_cur_onestep))
        print("Model MSE for multi Flow Array", i, MSE_cur_onestep)
    MSE_multistep = np.mean((YS_ha - YS) ** 2)
    f.write("Model MSE for multi Flow Array, %f\n" % MSE_multistep)
    print('MSE for multi step Flow Array', MSE_multistep)


def HA_pop_lastday(name):
    print(EVENT, specialDate)

    date = '20110226'

    ha = meshDynamic.getGridPopTimeInterval(meshTokyo, popFileName.format(date))
    ha = ha[:-1, :, :, :]
    print(ha.shape)

    specialData = meshDynamic.getGridPopTimeInterval(meshTokyo, popFileName.format(specialDate))
    specialData = specialData[:-1, :, :, :]
    print(specialData.shape)

    XS_ha, YS_ha = getXSYS(ha)
    XS, YS = getXSYS(specialData)

    assert YS_ha.shape == YS.shape

    f = open(PATH + '/' + name + '_prediction_scores_pop.txt', 'a')
    print('*' * 40 + 'LastDay')
    f.write('*' * 40 + 'LastDay' + '\n')
    for i in range(TIMESTEP):
        pred = YS_ha[:, i:i + 1, :, :, :]
        true = YS[:, i:i + 1, :, :, :]
        MSE_cur_onestep = np.mean((pred - true) ** 2)
        f.write("Model MSE for multi Density Array, %d, %f\n" % (i, MSE_cur_onestep))
        print("Model MSE for multi Density Array", i, MSE_cur_onestep)
    MSE_multistep = np.mean((YS_ha - YS) ** 2)
    f.write("Model MSE for multi Density Array, %f\n" % MSE_multistep)
    print('MSE for multi step Density Array', MSE_multistep)


def HA_flow_lastday(name):
    print(EVENT, specialDate)

    date = '20110226'
    ha = meshDynamic.getGridTransitTimeInterval(meshTokyo, transitFileName.format(date))
    print(ha.shape)

    specialData = meshDynamic.getGridTransitTimeInterval(meshTokyo, transitFileName.format(specialDate))
    print(specialData.shape)

    XS_ha, YS_ha = getXSYS(ha)
    XS, YS = getXSYS(specialData)

    assert YS_ha.shape == YS.shape

    f = open(PATH + '/' + name + '_prediction_scores_transit.txt', 'a')
    print('*' * 40 + 'LastDay')
    f.write('*' * 40 + 'LastDay' + '\n')
    for i in range(TIMESTEP):
        pred = YS_ha[:, i:i + 1, :, :, :]
        true = YS[:, i:i + 1, :, :, :]
        MSE_cur_onestep = np.mean((pred - true) ** 2)
        f.write("Model MSE for multi Flow Array, %d, %f\n" % (i, MSE_cur_onestep))
        print("Model MSE for multi Flow Array", i, MSE_cur_onestep)
    MSE_multistep = np.mean((YS_ha - YS) ** 2)
    f.write("Model MSE for multi Flow Array, %f\n" % MSE_multistep)
    print('MSE for multi step Flow Array', MSE_multistep)

# def HA_pop_test(name):
#     print(EVENT, specialDate)
#
#     date = '20110227'
#
#     ha = meshDynamic.getGridPopTimeInterval(meshTokyo, popFileName.format(date))
#     ha = ha[:-1, :, :, :]
#     print(ha.shape)
#
#     specialData = meshDynamic.getGridPopTimeInterval(meshTokyo, popFileName.format(specialDate))
#     specialData = specialData[:-1, :, :, :]
#     print(specialData.shape)
#
#     XS_ha, YS_ha = getXSYS(ha)
#     XS, YS = getXSYS(specialData)
#
#     assert YS_ha.shape == YS.shape
#
#     f = open(PATH + '/' + name + '_prediction_scores_pop.txt', 'a')
#     print('*' * 40 + 'LastDay')
#     f.write('*' * 40 + 'LastDay' + '\n')
#     for i in range(TIMESTEP):
#         pred = YS_ha[:, i:i + 1, :, :, :]
#         true = YS[:, i:i + 1, :, :, :]
#         MSE_cur_onestep = np.mean((pred - true) ** 2)
#         f.write("Model MSE for multi Density Array, %d, %f\n" % (i, MSE_cur_onestep))
#         print("Model MSE for multi Density Array", i, MSE_cur_onestep)
#     MSE_multistep = np.mean((YS_ha - YS) ** 2)
#     f.write("Model MSE for multi Density Array, %f\n" % MSE_multistep)
#     print('MSE for multi step Density Array', MSE_multistep)

def main():
    HA_pop(MODELNAME)
    HA_flow(MODELNAME)
    HA_pop_lastday(MODELNAME)
    HA_flow_lastday(MODELNAME)
    # HA_pop_test(MODELNAME)

if __name__ == '__main__':
    main()