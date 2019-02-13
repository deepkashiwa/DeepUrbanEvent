import csv
import datetime
import numpy as np
import os
import pandas as pd
import time
from math import radians, cos, sin, asin, sqrt
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from common.datastructure.Point import Point
from common.datastructure.old.Mesh import Mesh

STEP = 5
mesh = Mesh('tokyo', '500m')
inputDimension_poi = 40

mesh100 = Mesh('tokyo', '100m')

def loadTrajectory(filePath, mesh):
    LENGTHTRESH = STEP + 1
    df = pd.read_csv(filePath, header=None, dtype=str)
    TDB = {}
    total = len(df)
    for i, row in df.iterrows():
        print(i, total)
        oid = row[0]
        date = row[1]
        lon = float(row[2])
        lat = float(row[3])
        # datatime = time.strptime(date, "%Y-%m-%d %H:%M:%S")
        point = Point(lon, lat)
        if mesh.inMesh(point):
            grid = mesh.inWhichGrid(point)
            if oid in TDB:
                TDB[oid].append(grid)
            else:
                TDB[oid] = [grid]
        else:
            pass
    R = {}
    for key in TDB:
        if len(TDB[key]) < LENGTHTRESH:
            pass
        else:
            R[key] = TDB[key]
    return R


def loadLabelTrajectoryGrid(filePath, mesh):
    LENGTHTRESH = STEP + 1
    df = pd.read_csv(filePath, header=None, dtype=str)
    TDB = {}

    ids = set(df[0])

    print('loadLabelTrajectoryGrid total ids', len(ids))
    total = len(df)
    for i, row in df.iterrows():
        if i % 100000 == 0:
            print(i, total)
        oid = row[0]
        status = row[3]
        date = row[4] + ' ' + row[5]
        lon = float(row[8])
        lat = float(row[9])
        # datatime = time.strptime(date, "%Y-%m-%d %H:%M:%S")
        point = Point(lon, lat)

        key = oid + '_' + row[4]
        if mesh.inMesh(point):
            grid = mesh.inWhichGrid(point)
            if key in TDB:
                TDB[key].append(grid)
            else:
                TDB[key] = [grid]
        else:
            pass
    R = {}
    for key in TDB:
        if len(TDB[key]) < LENGTHTRESH:
            pass
        else:
            R[key] = TDB[key]
    return R


def loadLabelTrajectoryLonLat(filePath, mesh):
    LENGTHTRESH = STEP + 1
    df = pd.read_csv(filePath, header=None, dtype=str)
    TDB = {}

    ids = set(df[0])

    print('loadLabelTrajectoryGrid total ids', len(ids))
    total = len(df)
    for i, row in df.iterrows():
        if i % 100000 == 0:
            print(i, total)
        oid = row[0]
        status = row[3]
        date = row[4] + ' ' + row[5]
        lon = float(row[8])
        lat = float(row[9])
        # datatime = time.strptime(date, "%Y-%m-%d %H:%M:%S")
        point = Point(lon, lat)

        key = oid + '_' + row[4]
        if mesh.inMeshPoint(point):
            item = (row[8], row[9])
            if key in TDB:
                TDB[key].append(item)
            else:
                TDB[key] = [item]
        else:
            pass
    R = {}
    for key in TDB:
        if len(TDB[key]) < LENGTHTRESH:
            pass
        else:
            R[key] = TDB[key]
    return R


def getStepLabelTrajectoryGrid(filePath='./common.generator.instances/tokyo_output.csv'):
    print('started', time.ctime())
    TDB = loadLabelTrajectoryGrid(filePath, mesh)
    print('ended', time.ctime())

    wf = open('./common.generator.instances/tokyo_label_grid_' + str(STEP) + '.csv', 'w')
    for oid in TDB:
        sequence = TDB[oid]
        for i in range(STEP, len(sequence)):
            line = [oid]
            x = sequence[i-STEP:i]
            y = sequence[i]
            line.extend(x)
            line.append(y)
            line = [str(l) for l in line]
            wf.write(','.join(line) + '\n')
    wf.close()


def overDistance(p1, p2):
    x1, y1, x2, y2 = float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1])
    d = haversine(x1, y1, x2, y2)

    distanceUnit = 100.0
    # maxDistanceClass = 20
    maxDistanceClass = 50

    distanceClass = int(d / distanceUnit)
    if distanceClass > maxDistanceClass:
        return True
    else:
        return False


def getStepLabelTrajectoryLonLat(filePath='./common.generator.instances/tokyo_output.csv'):
    print('started', time.ctime())
    TDB = loadLabelTrajectoryLonLat(filePath, mesh)
    print('ended', time.ctime())

    wf = open('./common.generator.instances/tokyo_label_lonlat_' + str(STEP) + '.csv', 'w')
    wf_grid = open('./common.generator.instances/tokyo_label_grid_' + str(STEP) + '.csv', 'w')
    cnt = 0
    N_TDB = len(TDB)
    for oid in TDB:
        sequence = TDB[oid]
        for i in range(STEP, len(sequence)):
            # line = [oid]
            line = []
            x = sequence[i-STEP:i]
            y = sequence[i]

            y_minus = sequence[i-1]
            if overDistance(y_minus, y):
                continue
            else:
                line.extend(x)
                line.append(y)

                newline = [oid]
                newline_grid = [oid]
                for l in line:
                    newline.append(l[0])
                    newline.append(l[1])
                    # Keey the grid file same with lonlat file.
                    point = Point(float(l[0]), float(l[1]))
                    grid = mesh.inWhichGrid(point)
                    newline_grid.append(str(grid))

                wf.write(','.join(newline) + '\n')
                wf_grid.write(','.join(newline_grid) + '\n')
                if cnt % 100000 == 0:
                    print(cnt, '/', N_TDB)
                cnt += 1
    wf.close()
    wf_grid.close()


def shuffleGridTrainTest(rate=0.3):
    filePath = './common.generator.instances/tokyo_label_grid_' + str(STEP) + '.csv'
    df = pd.read_csv(filePath, header=None, dtype=str)
    INDEX = list(range(len(df)))
    np.random.shuffle(INDEX)
    np.random.shuffle(INDEX)
    np.random.shuffle(INDEX)
    train = INDEX[:int(len(INDEX) * (1 - rate))]
    test = INDEX[int(len(INDEX) * (1 - rate)):]
    trainData = df.loc[train]
    testData = df.loc[test]
    trainData.to_csv('./common.generator.instances/tokyo_label_grid_' + str(STEP) + '_train.csv', header=0, index=0)
    testData.to_csv('./common.generator.instances/tokyo_label_grid_' + str(STEP) + '_test.csv', header=0, index=0)


def shuffleGridTrainValidateTest(validateRate = 0.2, testRate = 0.2):
    print(validateRate, testRate)
    filePath = './common.generator.instances/tokyo_label_grid_' + str(STEP) + '.csv'
    df = pd.read_csv(filePath, header=None, dtype=str)
    INDEX = list(range(len(df)))
    np.random.shuffle(INDEX)
    np.random.shuffle(INDEX)
    np.random.shuffle(INDEX)
    train = INDEX[:int(len(INDEX) * (1 - testRate - validateRate))]
    validate = INDEX[int(len(INDEX) * (1 - testRate - validateRate)):int(len(INDEX) * (1 - testRate))]
    test = INDEX[int(len(INDEX) * (1 - testRate)):]
    trainData = df.loc[train]
    validateData = df.loc[validate]
    testData = df.loc[test]
    trainData.to_csv('./common.generator.instances/tokyo_label_grid_' + str(STEP) + '_train.csv', header=0, index=0)
    validateData.to_csv('./common.generator.instances/tokyo_label_grid_' + str(STEP) + '_validate.csv', header=0, index=0)
    testData.to_csv('./common.generator.instances/tokyo_label_grid_' + str(STEP) + '_test.csv', header=0, index=0)


def shuffleBothTrainValidateTest(validateRate = 0.2, testRate = 0.2):
    print(validateRate, testRate)
    gridFilePath = './common.generator.instances/tokyo_label_grid_' + str(STEP) + '.csv'
    df_grid = pd.read_csv(gridFilePath, header=None, dtype=str)

    lonlatFilePath = './common.generator.instances/tokyo_label_lonlat_' + str(STEP) + '.csv'
    df_lonlat = pd.read_csv(lonlatFilePath, header=None, dtype=str)

    assert len(df_grid) == len(df_lonlat)

    INDEX = list(range(len(df_grid)))
    np.random.shuffle(INDEX)
    np.random.shuffle(INDEX)
    np.random.shuffle(INDEX)
    train = INDEX[:int(len(INDEX) * (1 - testRate - validateRate))]
    validate = INDEX[int(len(INDEX) * (1 - testRate - validateRate)):int(len(INDEX) * (1 - testRate))]
    test = INDEX[int(len(INDEX) * (1 - testRate)):]
    
    trainData_grid = df_grid.loc[train]
    validateData_grid = df_grid.loc[validate]
    testData_grid = df_grid.loc[test]
    trainData_grid.to_csv('./common.generator.instances/tokyo_label_grid_' + str(STEP) + '_train.csv', header=0, index=0)
    validateData_grid.to_csv('./common.generator.instances/tokyo_label_grid_' + str(STEP) + '_validate.csv', header=0, index=0)
    testData_grid.to_csv('./common.generator.instances/tokyo_label_grid_' + str(STEP) + '_test.csv', header=0, index=0)

    trainData_lonlat = df_lonlat.loc[train]
    validateData_lonlat = df_lonlat.loc[validate]
    testData_lonlat = df_lonlat.loc[test]
    trainData_lonlat.to_csv('./common.generator.instances/tokyo_label_lonlat_' + str(STEP) + '_train.csv', header=0, index=0)
    validateData_lonlat.to_csv('./common.generator.instances/tokyo_label_lonlat_' + str(STEP) + '_validate.csv', header=0, index=0)
    testData_lonlat.to_csv('./common.generator.instances/tokyo_label_lonlat_' + str(STEP) + '_test.csv', header=0, index=0)


def shuffleGridPart(Unit=100000):
    filePath = './common.generator.instances/tokyo_label_grid_' + str(STEP) + '.csv'
    df = pd.read_csv(filePath, header=None, dtype=str)
    N = len(df)
    INDEX = list(range(N))
    np.random.shuffle(INDEX)
    np.random.shuffle(INDEX)
    np.random.shuffle(INDEX)

    wfs = [open('./common.generator.instances/tokyolabelgrid_step_' + str(STEP) +'_part_' + str(i) + '.csv', 'w') for i in range(N // Unit + 1)]
    for i in range(N):
        part = i // Unit
        wf = wfs[part]
        lineNumber = INDEX[i]
        line = ','.join(list(df.iloc[lineNumber])) + '\n'
        wf.write(line)

    for wf in wfs:
        wf.close()


def getStepXSYS(trainOrtest='train'):
    filePath = './common.generator.instances/tokyo_label_grid_' + str(STEP) + '_' + trainOrtest + '.csv'
    df = pd.read_csv(filePath, header=None)
    XS = df.iloc[:, 1:-1]
    YS = df.iloc[:, -1:]
    XS = XS.values
    YS = YS.values.reshape(-1)
    return XS, YS

def getGridStepXSYSTVT(trainOrvalidateOrTest='train'):
    filePath = './common.generator.instances/tokyo_label_grid_' + str(STEP) + '_' + trainOrvalidateOrTest + '.csv'
    df = pd.read_csv(filePath, header=None)
    XS = df.iloc[:, 1:-1]
    YS = df.iloc[:, -1:]
    XS = XS.values
    YS = YS.values.reshape(-1)
    return XS, YS


def getLonLatStepXSYSTVT(trainOrvalidateOrTest='train'):
    filePath = './common.generator.instances/tokyo_label_lonlat_' + str(STEP) + '_' + trainOrvalidateOrTest + '.csv'
    df = pd.read_csv(filePath, header=None)
    XS = df.iloc[:, 1:-2]
    YS = df.iloc[:, -2:]
    XS = XS.values.reshape(-1, STEP, 2)
    YS = YS.values.reshape(-1, 2)

    deltaLon = mesh.maxLon - mesh.minLon
    deltaLat = mesh.maxLat - mesh.minLat
    XS[:, :, 0] = (XS[:, :, 0] - mesh.minLon) / deltaLon
    XS[:, :, 1] = (XS[:, :, 1] - mesh.minLat) / deltaLat
    YS[:, 0] = (YS[:, 0] - mesh.minLon) / deltaLon
    YS[:, 1] = (YS[:, 1] - mesh.minLat) / deltaLat

    # XS[:, :, 0] = (XS[:, :, 0] - mesh.minLon)
    # XS[:, :, 1] = (XS[:, :, 1] - mesh.minLat)
    # YS[:, 0] = (YS[:, 0] - mesh.minLon)
    # YS[:, 1] = (YS[:, 1] - mesh.minLat)

    return XS, YS


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r * 1000 # return meters


def setLonLatDisDegXSYSTVT(trainOrvalidateOrTest='train'):
    lonlatfilePath = './common.generator.instances/tokyo_label_lonlat_' + str(STEP) + '_' + trainOrvalidateOrTest + '.csv'
    df_lonlat = pd.read_csv(lonlatfilePath, header=None)

    # gridfilePath = './common.generator.instances/tokyo_label_grid_' + str(STEP) + '_' + trainOrvalidateOrTest + '.csv'
    # df_grid = pd.read_csv(gridfilePath, header=None)

    XS_lonlat = df_lonlat.iloc[:, 1:-2]
    XS_lonlat = XS_lonlat.values
    print(XS_lonlat[0])

    YS_lonlat = df_lonlat.iloc[:, -4:]
    YS_lonlat = YS_lonlat.values
    print(YS_lonlat[0])

    x1, y1, x2, y2 = YS_lonlat[0][0], YS_lonlat[0][1], YS_lonlat[0][2], YS_lonlat[0][3]
    x, y = x2 - x1, y2 - y1
    print(x, y)
    print(139.76053251-139.76615675, 35.75053483-35.74980643)
    distanceUnit = 100.0
    degreeUnit = 45.0
    maxDistanceClass = 20

    filterXS = []
    Distance = []
    Degree = []
    for i in range(len(YS_lonlat)):
        x1, y1, x2, y2 = YS_lonlat[i][0], YS_lonlat[i][1], YS_lonlat[i][2], YS_lonlat[i][3]
        x, y = x2 - x1, y2 - y1
        d = haversine(x1, y1, x2, y2)

        distanceClass = int(d / distanceUnit)
        if distanceClass > maxDistanceClass:
            continue
        else:
            Distance.append(distanceClass)
            degree = np.arctan2(y, x) / np.pi * 180
            degree = (degree + 360.0) % 360.0
            degree = int(degree / degreeUnit)
            degreeClass = 0 if int(degree) == int(360.0 / degreeUnit) else int(degree)
            Degree.append(degreeClass)
            filterXS.append(XS_lonlat[i, :])
    print('array', filterXS[0])
    XS_filter = np.array(filterXS)
    print('dtype', XS_filter.dtype)
    YS_Dis = np.array(Distance, dtype='int32')
    YS_Deg = np.array(Degree, dtype='int32')

    return XS_filter, YS_Dis, YS_Deg


def setGridDisDegXSYSTVT(city, trainOrvalidateOrTest):
    lonlatfilePath = './common.generator.instances/' + city + '_label_lonlat_' + str(STEP) + '_' + trainOrvalidateOrTest + '.csv'
    df_lonlat = pd.read_csv(lonlatfilePath, header=None)

    gridfilePath = './common.generator.instances/' + city + '_label_grid_' + str(STEP) + '_' + trainOrvalidateOrTest + '.csv'
    df_grid = pd.read_csv(gridfilePath, header=None)

    XS_grid = df_grid.iloc[:, 1:-1]
    XS_grid = XS_grid.values

    YS_lonlat = df_lonlat.iloc[:, -4:]
    YS_lonlat = YS_lonlat.values

    distanceUnit = 200.0
    degreeUnit = 45.0
    maxDistanceClass = 10

    filterXS = []
    Distance = []
    Degree = []
    for i in range(len(YS_lonlat)):
        x1, y1, x2, y2 = YS_lonlat[i][0], YS_lonlat[i][1], YS_lonlat[i][2], YS_lonlat[i][3]
        x, y = x2 - x1, y2 - y1
        d = haversine(x1, y1, x2, y2)

        distanceClass = int(d / distanceUnit)
        if distanceClass > maxDistanceClass:
            continue
        else:
            Distance.append(distanceClass)
            degree = np.arctan2(y, x) / np.pi * 180
            degree = (degree + 360.0) % 360.0
            degree = int(degree / degreeUnit)
            degreeClass = 0 if int(degree) == int(360.0 / degreeUnit) else int(degree)
            Degree.append(degreeClass)
            filterXS.append(XS_grid[i, :])

    XS_filter = np.array(filterXS, dtype='int32')
    YS_Dis = np.array(Distance, dtype='int32')
    YS_Deg = np.array(Degree, dtype='int32')

    return XS_filter, YS_Dis, YS_Deg


def setGridDDXSYSTVT(city, trainOrvalidateOrTest):
    lonlatfilePath = './common.generator.instances/' + city + '_label_lonlat_' + str(STEP) + '_' + trainOrvalidateOrTest + '.csv'
    df_lonlat = pd.read_csv(lonlatfilePath, header=None)

    gridfilePath = './common.generator.instances/' + city + '_label_grid_' + str(STEP) + '_' + trainOrvalidateOrTest + '.csv'
    df_grid = pd.read_csv(gridfilePath, header=None)

    XS_grid = df_grid.iloc[:, 1:-1]
    XS_grid = XS_grid.values

    YS_lonlat = df_lonlat.iloc[:, -4:]
    YS_lonlat = YS_lonlat.values

    distanceUnit = 500.0
    degreeUnit = 45.0
    maxDistanceClass = 10
    maxDegreeClass = int(360.0 / degreeUnit)
    dict = {}

    ID = 0
    for i in range(maxDistanceClass+1):
        for j in range(maxDegreeClass):
            dict[(i,j)] = ID
            ID += 1
    filterXS = []
    DD = []
    for i in range(len(YS_lonlat)):
        x1, y1, x2, y2 = YS_lonlat[i][0], YS_lonlat[i][1], YS_lonlat[i][2], YS_lonlat[i][3]
        x, y = x2 - x1, y2 - y1
        d = haversine(x1, y1, x2, y2)

        distanceClass = int(d / distanceUnit)
        if distanceClass > maxDistanceClass:
            continue
        else:
            degree = np.arctan2(y, x) / np.pi * 180
            degree = (degree + 360.0) % 360.0
            degree = int(degree / degreeUnit)
            degreeClass = 0 if int(degree) == int(360.0 / degreeUnit) else int(degree)
            index = (distanceClass, degreeClass)
            DD.append(dict[index])
            filterXS.append(XS_grid[i, :])

    XS_filter = np.array(filterXS, dtype='int32')
    YS = np.array(DD, dtype='int32')

    return XS_filter, YS



def setGridAroundXSYSTVT(city, trainOrvalidateOrTest):
    outputWindow = 2
    dict = {}
    ID = 0
    for i in range(0 - outputWindow, outputWindow + 1):
        for j in range(0 - outputWindow, outputWindow + 1):
            dict[(i, j)] = ID
            ID += 1

    gridfilePath = './common.generator.instances/' + city + '_label_grid_' + str(STEP) + '_' + trainOrvalidateOrTest + '.csv'
    df_grid = pd.read_csv(gridfilePath, header=None)

    XS_grid = df_grid.iloc[:, 1:-1]
    XS_grid = XS_grid.values

    YS_grid = df_grid.iloc[:, -2:]
    YS_grid = YS_grid.values

    maxDistance = 2 * (2 ** 0.5)

    XS = []
    YS = []
    for i in range(len(YS_grid)):
        p1, p2 = YS_grid[i][0], YS_grid[i][1]
        x1, y1 = mesh.Index[p1]
        x2, y2 = mesh.Index[p2]
        x, y = x2 - x1, y2 - y1

        distance = (x ** 2 + y ** 2) ** 0.5
        if distance > maxDistance:
            continue
        else:
            aroundID = dict[(x, y)]
            YS.append(aroundID)
            XS.append(XS_grid[i, :])

    XS = np.array(XS, dtype='int32')
    YS = np.array(YS, dtype='int32')

    return XS, YS


def verify(YS_Dis, YS_Deg):
    min_dis, max_dis = np.min(YS_Dis), np.max(YS_Dis)
    min_deg, max_deg = np.min(YS_Deg), np.max(YS_Deg)
    print(min_dis, max_dis, min_deg, max_deg)

    n, bins, patches = plt.hist(YS_Dis, bins=list(range(min_dis, max_dis + 2)))
    print(n, bins, patches)
    plt.savefig('./YS_Distance_' + str(min_dis) + '_' + str(max_dis) + '.png')
    # plt.savefig('./YS_Distance_NoZero_' + str(min_dis) + '_' + str(max_dis) + '.png')
    plt.close()

    n, bins, patches = plt.hist(YS_Deg, bins=list(range(min_deg, max_deg + 2)))
    print(n, bins, patches)
    plt.savefig('./YS_Degree_' + str(min_deg) + '_' + str(max_deg) + '.png')
    # plt.savefig('./YS_Degree_NoZero_' + str(min_deg) + '_' + str(max_deg) + '.png')
    plt.close()


def getStepXSYSPart(part):
    filePath = './common.generator.instances/tokyolabelgrid_step_' + str(STEP) +'_part_' + str(part) + '.csv'
    df = pd.read_csv(filePath, header=None)
    XS = df.iloc[:, 1:-1]
    YS = df.iloc[:, -1:]
    XS = XS.values
    YS = YS.values.reshape(-1)
    return XS, YS


def getPOI(city, meshSize):
    inputDimension_poi = 40
    POI = []
    poiFileName = './common.generator.instances/GridPoI_' + city + '_' + meshSize + '.csv'
    poiReadFile = open(poiFileName, 'r')
    poiReader = csv.reader(poiReadFile)
    for line in poiReader:
        # gid = int(line[0])
        poi = []
        # for i in range(inputDimension_location):
        #     poi.append(float(line[1 + i]))
        for i in range(inputDimension_poi):
            poi.append(int(line[5 + i]))
        POI.append(poi)
    poiReadFile.close()
    return POI


def genScaledPOI(city, meshSize):
    POI = getPOI(city, meshSize)
    POI = np.array(POI)
    scaledPOI = MinMaxScaler().fit_transform(POI)
    scaledPoiFileName = './common.generator.instances/ScaledGridPoI_' + city + '_' + meshSize + '.csv'
    np.savetxt(scaledPoiFileName, scaledPOI, fmt='%.8f', delimiter=',')
    return POI


def getScaledPOI(city, meshSize):
    scaledPoiFileName = './common.generator.instances/ScaledGridPoI_' + city + '_' + meshSize + '.csv'
    POI = np.loadtxt(scaledPoiFileName, delimiter=',')
    return POI


def getStepXSPOI(XS, POI):
    assert len(XS.shape) == 2
    XSPOI = np.zeros((XS.shape[0], XS.shape[1], inputDimension_poi), dtype='int32')
    for i in range(XS.shape[0]):
        for j in range(XS.shape[1]):
            grid = XS[i][j]
            poi = POI[grid]
            XSPOI[i][j] = poi
    return XSPOI


def getStepXSScaledPOI(XS, scaledPOI):
    assert len(XS.shape) == 2
    XSPOI = np.zeros((XS.shape[0], XS.shape[1], inputDimension_poi))
    for i in range(XS.shape[0]):
        for j in range(XS.shape[1]):
            grid = XS[i][j]
            poi = scaledPOI[grid]
            XSPOI[i][j] = poi
    return XSPOI


def getStepXSPOIVideo(Points, POI):
    def convertToXS(Point, POI):
        R = np.zeros((10, 10, inputDimension_poi))
        X = Point
        current_x, current_y = mesh.Index[X]

        # l = []
        # l.extend(coordinate_to_one_zero(current_x))
        # l.extend(coordinate_to_one_zero(current_y))
        # L = np.array(l)

        for i, dx in enumerate(list(range(-4, 6))):
            for j, dy in enumerate(list(range(-4, 6))):
                x = current_x + dx
                y = current_y + dy
                if mesh.inMesh(x, y):
                    grid = mesh.ReverseIndex[(x, y)]
                    R[j][i] = POI[grid]
                else:
                    poi = [0] * inputDimension_poi
                    R[j][i] = poi
        R = R[::-1, :, :]
        return R

    # POI = getPOI()
    N = len(Points)
    R = np.zeros((N, STEP, 10, 10, inputDimension_poi))
    # L = np.zeros((N, STEP, inputDimension_location * 2))

    for i in range(N):
        points = Points[i]
        for j in range(STEP):
            # r, l = convertToXS(points[j], POI)
            r = convertToXS(points[j], POI)
            R[i][j] = r
            # L[i][j] = l
    # return R, L
    return R


def getWindow(mesh, id, window):
    R = np.zeros((window, window), dtype='int32')
    X = id
    current_x, current_y = mesh.Index[X]
    start = 0 - window // 2
    end = window + start
    for i, dx in enumerate(list(range(start, end))):
        for j, dy in enumerate(list(range(start, end))):
            x = current_x + dx
            y = current_y + dy
            if mesh.inMesh(x, y):
                grid = mesh.ReverseIndex[(x, y)]
                R[j][i] = grid
            else:
                R[j][i] = -1
    R = R[::-1, :]
    return R


def getStepXSPOIVideoOutside(Points, POI1000, mode):
    def convertToXS(Point, POI):
        R = np.zeros((3, 3, inputDimension_poi))
        X = Point
        current_x, current_y = mesh.Index[X]
        for i, dx in enumerate(list(range(-1, 2))):
            for j, dy in enumerate(list(range(-1, 2))):
                x = current_x + dx
                y = current_y + dy
                if mesh.inMesh(x, y):
                    grid = mesh.ReverseIndex[(x, y)]
                    R[j][i] = POI[grid]
                else:
                    poi = [0] * inputDimension_poi
                    R[j][i] = poi
        R = R[::-1, :, :]
        return R

    if mode == 'video':
        N = len(Points)
        R = np.zeros((N, STEP, 3, 3, inputDimension_poi))
        for i in range(N):
            points = Points[i]
            for j in range(STEP):
                r = convertToXS(points[j], POI1000)
                R[i][j] = r
        return R
    elif mode == 'vector':
        N = len(Points)
        R = np.zeros((N, STEP, inputDimension_poi))
        for i in range(N):
            points = Points[i]
            for j in range(STEP):
                r = convertToXS(points[j], POI1000)
                r = r.reshape(3 * 3, inputDimension_poi)
                r = r.sum(axis=0)
                R[i][j] = r
        return R
    else:
        assert False


def treeINDEX(index, LON_SMALL, LAT_SMALL, LON_BIG, LAT_BIG):
    assert index < LON_SMALL * LAT_SMALL
    X_LEN = LON_BIG // LON_SMALL
    Y_LEN = LAT_BIG // LAT_SMALL
    tree = np.zeros((X_LEN, Y_LEN), dtype='int32')
    for i in range(X_LEN):
        seed = (index // LAT_SMALL) * (X_LEN * LON_BIG) \
               + (index % LAT_SMALL) * Y_LEN + i * LON_BIG
        for j in range(Y_LEN):
            tree[i][j] = seed
            seed += 1
    tree = np.swapaxes(tree, 0, 1)
    tree = tree[::-1, :]
    return tree


def treePOI(POI100, index, LON_SMALL, LAT_SMALL, LON_BIG, LAT_BIG):
    assert index < LON_SMALL * LAT_SMALL
    X_LEN = LON_BIG // LON_SMALL
    Y_LEN = LAT_BIG // LAT_SMALL
    tree = np.zeros((X_LEN, Y_LEN), dtype='int32')
    for i in range(X_LEN):
        seed = (index // LAT_SMALL) * (X_LEN * LON_BIG) \
               + (index % LAT_SMALL) * Y_LEN + i * LON_BIG
        for j in range(Y_LEN):
            tree[i][j] = seed
            seed += 1
    tree = np.swapaxes(tree, 0, 1)
    tree = tree[::-1, :]

    treepoi = np.zeros((X_LEN, Y_LEN, inputDimension_poi), dtype='int32')
    for i in range(X_LEN):
        for j in range(Y_LEN):
            treepoi[i][j] = POI100[tree[i][j]]
    return treepoi


def outsidePOI(POI1000, index, window):
    R = np.zeros((window, window, inputDimension_poi), dtype='int32')
    X = index
    current_x, current_y = mesh.Index[X]
    start = 0 - window // 2
    end = window + start
    for i, dx in enumerate(list(range(start, end))):
        for j, dy in enumerate(list(range(start, end))):
            x = current_x + dx
            y = current_y + dy
            if mesh.inMesh(x, y):
                grid = mesh.ReverseIndex[(x, y)]
                R[j][i] = POI1000[grid]
            else:
                poi = [0] * inputDimension_poi
                R[j][i] = poi
    R = R[::-1, :, :]
    return R


def scaledOutsidePOI(scaledPOI1000, index, window):
    R = np.zeros((window, window, inputDimension_poi))
    X = index
    current_x, current_y = mesh.Index[X]
    start = 0 - window // 2
    end = window + start
    for i, dx in enumerate(list(range(start, end))):
        for j, dy in enumerate(list(range(start, end))):
            x = current_x + dx
            y = current_y + dy
            if mesh.inMesh(x, y):
                grid = mesh.ReverseIndex[(x, y)]
                R[j][i] = scaledPOI1000[grid]
            else:
                poi = [0] * inputDimension_poi
                R[j][i] = poi
    R = R[::-1, :, :]
    return R


def genOutsidePOI(city, meshSize, window=3):
    POI1000 = getPOI(city, meshSize)
    wfFileName = './common.generator.instances/GridPoIOutside_' + city + '_' + meshSize + '_' + str(window) + '.csv'
    wf = open(wfFileName, 'w')
    for id in range(mesh.lonNum * mesh.latNum):
        outsidepoi = outsidePOI(POI1000, id, window)
        outsidepoi = outsidepoi.ravel().tolist()
        line = [id]
        line.extend(outsidepoi)
        line = [str(item) for item in line]
        line = ','.join(line) + '\n'
        wf.write(line)
        print(id)
    wf.close()


def genScaledOutsidePOI(city, meshSize, window):
    scaledPOI1000 = getScaledPOI(city, meshSize)
    wfFileName = './common.generator.instances/ScaledGridPoIOutside_' + city + '_' + meshSize + '_' + str(window) + '.csv'
    R = []
    for id in range(mesh.lonNum * mesh.latNum):
        outsidepoi = scaledOutsidePOI(scaledPOI1000, id, window)
        outsidepoi = outsidepoi.ravel()
        R.append(outsidepoi)
        if id % 10000 == 0:
            print(id)
    X = np.array(R)
    print(X.shape)
    np.savetxt(wfFileName, X, fmt='%.8f', delimiter=',')


def genScaledOutsidePOIMovie(city, meshSize, window):
    scaledPOI1000 = getScaledPOI(city, meshSize)
    PATH = './common.movie.' + city + '.' + meshSize + '.' + str(window)

    if os.path.exists(PATH):
        pass
    else:
        os.makedirs(PATH)

    wfFileName = PATH + '/ScaledGridPoIOutside_' + city + '_' + meshSize + '_' + str(window)
    # R = []
    for id in range(mesh.lonNum * mesh.latNum):
        outsidepoi = scaledOutsidePOI(scaledPOI1000, id, window)
        outsidepoi = outsidepoi.ravel()
        # R.append(outsidepoi)
        if id % 10000 == 0:
            print(id)
        X = outsidepoi
        tmp = wfFileName + '_id' + str(id) + '.csv'
        np.savetxt(tmp, X, fmt='%.4f', delimiter=',')


def gentreePOI():
    POI100 = getPOI('tokyo', '100m')
    wfFileName = './common.generator.instances/GridPoIVideo_' + 'tokyo' + '_' + '1000m' + '.csv'
    wf = open(wfFileName, 'w')

    for id in range(mesh.lonNum * mesh.latNum):
        treepoi = treePOI(POI100, id,  mesh.lonNum, mesh.latNum, mesh100.lonNum, mesh100.latNum)
        treepoi = treepoi.ravel().tolist()
        line = [id]
        line.extend(treepoi)
        line = [str(item) for item in line]
        line = ','.join(line) + '\n'
        wf.write(line)
        print(id)
    wf.close()


def gettreePOI():
    treefilePath = './common.generator.instances/GridPoIVideo_' + 'tokyo' + '_' + '1000m' + '.csv'
    poi = pd.read_csv(treefilePath, header=None)
    poi = poi.iloc[:, 1:]
    return poi.values


def getOutsidePOI(window=3):
    wfFileName = './common.generator.instances/GridPoIOutside_' + 'tokyo' + '_' + '1000m' + '_' + str(window) + '.csv'
    poi = pd.read_csv(wfFileName, header=None)
    poi = poi.iloc[:, 1:]
    return poi.values


def getScaledOutsidePOI(city, meshSize, window):
    wfFileName = './common.generator.instances/ScaledGridPoIOutside_' + city + '_' + meshSize + '_' + str(window) + '.csv'
    poi = pd.read_csv(wfFileName, header=None)
    return poi.values


def getStepXSTreePOI(XS, treePOI):
    assert len(XS.shape) == 2
    XSPOI = np.zeros((XS.shape[0], XS.shape[1], 10, 10, inputDimension_poi), dtype='int32')
    for i in range(XS.shape[0]):
        for j in range(XS.shape[1]):
            grid = XS[i][j]
            poi = treePOI[grid]
            XSPOI[i][j] = poi.reshape(10,10,inputDimension_poi)
    return XSPOI


def getStepXSOutsidePOI(XS, outsidePOI, window):
    assert len(XS.shape) == 2
    XSPOI = np.zeros((XS.shape[0], XS.shape[1], window, window, inputDimension_poi), dtype='int32')
    for i in range(XS.shape[0]):
        for j in range(XS.shape[1]):
            grid = XS[i][j]
            poi = outsidePOI[grid]
            XSPOI[i][j] = poi.reshape(window, window, inputDimension_poi)
    return XSPOI


def getStepXSScaledOutsidePOI(XS, scaledOutsidePOI, window):
    assert len(XS.shape) == 2
    XSPOI = np.zeros((XS.shape[0], XS.shape[1], window, window, inputDimension_poi))
    for i in range(XS.shape[0]):
        for j in range(XS.shape[1]):
            grid = XS[i][j]
            poi = scaledOutsidePOI[grid]
            XSPOI[i][j] = poi.reshape(window, window, inputDimension_poi)
    return XSPOI

def getGridMovie(grid, city, meshSize, window):
    PATH = './common.movie.' + city + '.' + meshSize + '.' + str(window)
    fileName = PATH + '/ScaledGridPoIOutside_' + city + '_' + meshSize + '_' + str(window)
    rf = fileName + '_id' + str(grid) + '.csv'
    return np.loadtxt(rf, dtype='float32')

def getStepXSScaledOutsidePOIMovie(XS, city, meshSize, window):
    print('video start', time.ctime())
    assert len(XS.shape) == 2
    XSPOI = np.zeros((XS.shape[0], XS.shape[1], window, window, inputDimension_poi))
    ID = 0
    for i in range(XS.shape[0]):
        for j in range(XS.shape[1]):
            grid = XS[i][j]
            poi = getGridMovie(grid, city, meshSize, window)
            print(ID)
            ID += 1
            XSPOI[i][j] = poi.reshape(window, window, inputDimension_poi)
    print('video end', time.ctime())
    return XSPOI

def getStepXSOutsidePOIVector(XS, outsidePOI, window):
    assert len(XS.shape) == 2
    XSPOI = np.zeros((XS.shape[0], XS.shape[1], inputDimension_poi), dtype='int32')
    for i in range(XS.shape[0]):
        for j in range(XS.shape[1]):
            grid = XS[i][j]
            poi = outsidePOI[grid]
            poi = poi.reshape(window * window, inputDimension_poi)
            poi = poi.sum(axis=0)
            XSPOI[i][j] = poi
    return XSPOI


def getStepXSPOIVideoInside(Points, POI):
    xlen = mesh100.lonNum // mesh.lonNum
    ylen = mesh100.latNum // mesh.latNum

    def convertToXS(Point, POI):
        R = np.zeros((xlen, ylen, inputDimension_poi))
        X = Point
        tree = treeINDEX(X, mesh.lonNum, mesh.latNum, mesh100.lonNum, mesh100.latNum)
        for i in range(xlen):
            for j in range(ylen):
                smallindex = tree[i][j]
                R[i][j] = POI[smallindex]
        return R

    N = len(Points)
    R = np.zeros((N, STEP, xlen, ylen, inputDimension_poi))

    for i in range(N):
        points = Points[i]
        for j in range(STEP):
            r = convertToXS(points[j], POI)
            R[i][j] = r
    return R


def getDisDeg(XS, YS, mesh=Mesh('tokyo', '500m')):
    assert len(XS) == len(YS)
    P1, P2 = XS[:, -1], YS

    newXS = []
    Distance = []
    Degree = []
    # distanceThresh = 22.0  # about 11km
    distanceUnit = 1.0
    degreeUnit = 45.0

    maxDistance = 4
    for i in range(len(P1)):
        p1, p2 = P1[i], P2[i]
        x1, y1 = mesh.Index[p1]
        x2, y2 = mesh.Index[p2]
        x, y = x2 - x1, y2 - y1
        d = (x ** 2 + y ** 2) ** 0.5

        distanceClass = int(d / distanceUnit)
        if distanceClass > maxDistance:
            Distance.append(maxDistance)
        else:
            Distance.append(distanceClass)

        degree = np.arctan2(y, x) / np.pi * 180
        degree = (degree + 360.0) % 360.0
        degree = int(degree / degreeUnit)
        degreeClass = 0 if int(degree) == int(360.0 / degreeUnit) else int(degree)
        Degree.append(degreeClass)
        newXS.append(XS[i, :])

    # return np.array(Distance, dtype='int32'), np.array(Degree, dtype='int32')
    return np.array(newXS, dtype='int32'), \
           np.array(Distance, dtype='int32'), \
           np.array(Degree, dtype='int32')


def generateGridTVT():
    XS, YS = getGridStepXSYSTVT('train')
    print(XS.shape, YS.shape)
    XS, YS = getGridStepXSYSTVT('validate')
    print(XS.shape, YS.shape)
    XS, YS = getGridStepXSYSTVT('test')
    print(XS.shape, YS.shape)


def main():
    pass
    # shufflePart()
    # XS, YS = getStepXSYSPart(0)
    # print(XS.shape, YS.shape)
    # generateGridTVT()


if __name__ == '__main__':
    main()