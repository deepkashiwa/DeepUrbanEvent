import csv
import numpy as np
import os
import sys
import time
import jismesh.utils as ju
import pandas as pd

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from common.datastructure.Point import Point
from common.datastructure.Mesh import Mesh

# meshTokyo = Mesh('tokyo','500m')
# GRIDNUMBER = meshTokyo.lonNum * meshTokyo.latNum
# print(meshTokyo.size, GRIDNUMBER)
# InterpolatedStep = 12


def getTimestamps(fileName):
    last_tid = ''
    D = []
    with open(fileName, "r") as rf:
        reader = csv.reader(rf)
        for line in reader:
            tid = line[0]
            if last_tid != '' and last_tid != tid:
                break
            timestamp = line[1]
            D.append(timestamp)
            last_tid = tid
    return D

def getMesh(mesh, readFileName, writeFileName):
    cnt = 0
    wf = open(writeFileName, 'w')
    with open(readFileName, 'r') as rf:
        for line in csv.reader(rf):
            if cnt % 1000000 == 0:
                print(cnt)
            tid = line[0]
            timestamp = line[1]
            p = Point(float(line[2]), float(line[3]))
            meshid = mesh.inWhichGrid(p)
            wf.write(','.join([tid, timestamp, str(meshid)])+'\n')
            cnt += 1
    wf.close()

def genMeshDynamic(mesh, fileName, meshFileName):
    MD = {}
    with open(fileName, "r") as rf:
        reader = csv.reader(rf)
        for line in reader:
            tid = line[0]
            timestamp = line[1]
            meshid = line[2]
            key = (timestamp, meshid)
            if key in MD:
                MD[key].add(tid)
            else:
                MD[key] = set(tid)

    wf = open(meshFileName, 'w')
    Timestamps = getTimestamps(fileName)
    for ts in Timestamps:
        for meshid in range(mesh.lonNum * mesh.latNum):
            key = (ts, str(meshid))
            if key in MD:
                value = len(MD[key])
            else:
                value = 0
            wf.write(','.join([key[0], key[1], str(value)]) + '\n')
    wf.close()

def getGrids(fileName):
    last_tid = ''
    G = []
    with open(fileName, "r") as rf:
        reader = csv.reader(rf)
        for line in reader:
            tid = line[0]
            if last_tid != '' and last_tid != tid:
                break
            grid = line[1]
            G.append(grid)
            last_tid = tid
    return G

def getDynamicMesh_mobmap(trajFileName, dynamicFileName, meshcode_level):
    Timestamps = getTimestamps(trajFileName)
    TIMENUMBER = len(Timestamps)
    TS = {}
    for i in range(TIMENUMBER):
        TS[Timestamps[i]] = i
    print('getDynamicMesh Started : ', time.ctime())
    R = []
    for i in range(TIMENUMBER):
        R.append({})
    with open(trajFileName, 'r') as rf:
        reader = csv.reader(rf)
        for line in reader:
            # tid = line[0]
            timestamp = line[1]
            lon = float(line[2])
            lat = float(line[3])
            meshcode = ju.to_meshcode(lat, lon, meshcode_level)
            if meshcode in R[TS[timestamp]]:
                R[TS[timestamp]][meshcode] += 1
            else:
                R[TS[timestamp]][meshcode] = 1

    print('getDynamicMesh Count Ended : ', time.ctime())
    with open(dynamicFileName, 'w') as wf:
        wf.write("@dynamic-mesh\n")
        wf.write("@use-mesh-code," + str(meshcode_level))
        for i in range(len(R)):
            timestamp = Timestamps[i]
            for key in R[i]:
                meshcode = key
                meshpop = R[i][meshcode]
                wf.write(','.join([timestamp, meshcode, str(meshpop)]) + '\n')

    print('getDynamicMesh Ended : ', time.ctime())

def getDynamicMeshMobmap(trajFileName, dynamicFileName, meshcode_level):
    Timestamps = getTimestamps(trajFileName)
    TIMENUMBER = len(Timestamps)
    TS = {}
    for i in range(TIMENUMBER):
        TS[Timestamps[i]] = i
    print('getDynamicMesh Started : ', time.ctime())
    R = []
    for i in range(TIMENUMBER):
        R.append({})
    with open(trajFileName, 'r') as rf:
        reader = csv.reader(rf)
        for line in reader:
            # tid = line[0]
            timestamp = line[1]
            lon = float(line[2])
            lat = float(line[3])
            meshcode = ju.to_meshcode(lat, lon, meshcode_level)
            if meshcode in R[TS[timestamp]]:
                R[TS[timestamp]][meshcode] += 1
            else:
                R[TS[timestamp]][meshcode] = 1

    with open(dynamicFileName, 'w') as wf:
        wf.write("@dynamic-mesh\n")
        wf.write("@use-mesh-code," + str(meshcode_level))
        for i in range(len(R)):
            timestamp = Timestamps[i]
            for key in R[i]:
                meshcode = key
                meshpop = R[i][meshcode]
                wf.write(','.join([timestamp, meshcode, str(meshpop)]) + '\n')

    print('getDynamicMesh Ended : ', time.ctime())


def getRfromDynamicMeshMobmap(meshcode_level, dynamicFileName, dynamicFileName1, dynamicFileName2):
    df1 = pd.read_csv(dynamicFileName, header=None, skiprows=2)
    df1.iloc[:,2] = np.log10(df1.iloc[:,2]+1) * 100

    df2 = pd.read_csv(dynamicFileName, header=None, skiprows=2)
    df2.iloc[:, 2] = np.log(df2.iloc[:,2]+1) * 100

    with open(dynamicFileName1, 'w') as wf:
        wf.write("@dynamic-mesh\n")
        wf.write("@use-mesh-code," + str(meshcode_level) + '\n')

    with open(dynamicFileName2, 'w') as wf:
        wf.write("@dynamic-mesh\n")
        wf.write("@use-mesh-code," + str(meshcode_level) + '\n')

    df1.to_csv(dynamicFileName1, header=False, index=False, mode='a')
    df2.to_csv(dynamicFileName2, header=False, index=False, mode='a')

def getDynamicMeshMobmapR(R, trajFileName, dynamicFileName, meshcode_level):
    Timestamps = getTimestamps(trajFileName)
    print('getDynamicMesh Count Ended : ', time.ctime())
    with open(dynamicFileName, 'w') as wf:
        wf.write("@dynamic-mesh\n")
        wf.write("@use-mesh-code," + str(meshcode_level))
        for i in range(len(R)):
            timestamp = Timestamps[i]
            for key in R[i]:
                meshcode = key
                meshpop = R[i][meshcode]
                wf.write(','.join([timestamp, meshcode, str(meshpop)]) + '\n')

    print('getDynamicMesh Ended : ', time.ctime())


def genMeshDynamicTimeInterval(fileName, meshFileName, startTimestamp, endTimestamp):
    Timestamps = getTimestamps(fileName)
    startIndex = Timestamps.index(startTimestamp)
    endIndex = Timestamps.index(endTimestamp)
    Interval = [Timestamps[t] for t in range(startIndex, endIndex)]

    def strHH(timestamp):
        return timestamp[11:13] + timestamp[14:16]

    wf = open(meshFileName[:-4] + '_' + strHH(startTimestamp) + '_' + strHH(endTimestamp) + '.csv', 'w')
    with open(meshFileName, 'r') as rf:
        for line in csv.reader(rf):
            if line[0] in Interval:
                wf.write(','.join(line) + '\n')
            else:
                pass
    wf.close()

def genMeshDynamicTimeInterval_Mobmap(fileName, meshFileName, startTimestamp, endTimestamp):
    Timestamps = getTimestamps(fileName)
    startIndex = Timestamps.index(startTimestamp)
    endIndex = Timestamps.index(endTimestamp)
    Interval = [Timestamps[t] for t in range(startIndex, endIndex)]

    def strHH(timestamp):
        return timestamp[11:13] + timestamp[14:16]

    wf = open(meshFileName[:-4] + '_' + strHH(startTimestamp) + '_' + strHH(endTimestamp) + '.csv', 'w')
    with open(meshFileName, 'r') as rf:
        for line in csv.reader(rf):
            if line[0] == '@dynamic-mesh' or '"@use-mesh-code':
                wf.write(line + '\n')
            if line[0] in Interval:
                wf.write(','.join(line) + '\n')
            else:
                pass
    wf.close()

def genMeshDynamicMobmap(mesh, meshFileName, mobmapFile, timestamp):
    wf = open(mobmapFile, 'w')
    wf.write('@static-mesh' + '\n')
    wf.write(','.join([str(x) for x in
            [mesh.minLat, mesh.minLon, mesh.dLat, mesh.dLon]]) + '\n')
    with open(meshFileName, 'r') as rf:
        for line in csv.reader(rf):
            if timestamp != line[0]:
                continue
            else:
                meshid = line[1]
                number = line[2]
                xi, yi = mesh.Index[int(meshid)]
                wf.write(','.join([str(item) for item in [yi, xi, number]]) + '\n')
    wf.close()

def loadGTrajectory(fileName):
    print('loadTrajectory Started : ', time.ctime())
    TDB = {}
    with open(fileName, 'r') as rf:
        reader = csv.reader(rf)
        for line in reader:
            tid = line[0]
            # timestamp = line[1]
            meshid = line[2]
            if tid in TDB:
                TDB[tid].append(meshid)
            else:
                TDB[tid] = [meshid]
    print('loadTrajectory Ended : ', time.ctime())
    return TDB

def getINDEX(mesh, gTrajFileName):
    GRIDNUMBER = mesh.lonNum * mesh.latNum
    print('getTrajectoryINDEX Started : ', time.ctime())
    Timestamps = getTimestamps(gTrajFileName)
    print('timestamps...', len(Timestamps))
    TDB = loadGTrajectory(gTrajFileName)
    INDEX = []
    for i in range(len(Timestamps)):
        INDEX.append([])
        for G in range(GRIDNUMBER):
            INDEX[i].append(set())  # set().add

    # print(np.array(INDEX).shape)
    for tid in TDB:
        traj = TDB[tid]
        for i in range(len(traj)):
            HH = i
            if traj[i] == 'None':
                pass
            else:
                gid = int(traj[i])
                INDEX[HH][gid].add(tid)  # set().add
    return INDEX

def getGridImageIndex(mesh, window=15):
    GRIDNUMBER = mesh.lonNum * mesh.latNum
    IMG = []
    for g in range(GRIDNUMBER):
        R = np.zeros((window, window), dtype='int32')
        current_x, current_y = mesh.Index[g]
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
        IMG.append(R)
    return IMG

def genGridTransit(mesh, gTrajFileName, transitFileName):
    GRIDNUMBER = mesh.lonNum * mesh.latNum
    print('genGridTransit Started : ', time.ctime())
    transitWriteFile = open(transitFileName, 'w')
    INDEX = getINDEX(mesh, gTrajFileName)
    Timestamps = getTimestamps(gTrajFileName)
    GridImageIndex = getGridImageIndex(mesh)
    print('INDEX, Timestamps, GridImageIndex have been prepared.', time.ctime())

    for i in range(len(Timestamps) - 1):
        for j in range(GRIDNUMBER):
            cur_time = i
            next_time = i + 1
            cur_grid = j
            transitgrids = GridImageIndex[cur_grid]
            Transit = np.zeros(transitgrids.shape, dtype='int32')
            for ii in range(transitgrids.shape[0]):
                for jj in range(transitgrids.shape[1]):
                    next_grid = transitgrids[ii][jj]
                    if next_grid != -1:
                        trajfirst = INDEX[cur_time][cur_grid]
                        trajsecond = INDEX[next_time][next_grid]
                        transit_num = len(trajfirst & trajsecond)
                        Transit[ii][jj] = transit_num
                    else:
                        pass
            FlattedTransit = Transit.reshape(-1).tolist()
            lineitem = [str(i), str(j)]
            lineitem.extend([str(t) for t in FlattedTransit])
            line = ','.join(lineitem) + '\n'
            transitWriteFile.write(line)
        print('genGridTransit timestamp: ', i)
    transitWriteFile.close()
    print('genGridTransit Ended: ', time.ctime())

# This grid transit version is for 1minutes trajectory, more accurate, not for 5minutes.
# !!!!!!!!!!!!!!!!!!!! 1 minute trajectory data.
# TT is supposed to be 288 not 289 because it is interval.
def genGridTransit_5minutes_from_1minute(mesh, gTrajFileName, transitFileName):
    GRIDNUMBER = mesh.lonNum * mesh.latNum
    print('genGridTransit Started : ', time.ctime())
    transitWriteFile = open(transitFileName, 'w')
    INDEX = getINDEX(mesh, gTrajFileName)
    # Timestamps = getTimestamps(gTrajFileName)
    GridImageIndex = getGridImageIndex(mesh)
    print('INDEX, Timestamps, GridImageIndex have been prepared.', time.ctime())

    TT, SPAN = 24 * 12, 5
    for i in range(TT):
        for j in range(GRIDNUMBER):
            cur_time = i
            cur_grid = j
            transitgrids = GridImageIndex[cur_grid]
            Transit = np.zeros(transitgrids.shape, dtype='int32')
            for ii in range(transitgrids.shape[0]):
                for jj in range(transitgrids.shape[1]):
                    next_grid = transitgrids[ii][jj]
                    if next_grid != -1:
                        cur_time_start = cur_time * SPAN
                        cur_time_end = (cur_time + 1) * SPAN + 1
                        SS = set()
                        for pp in range(cur_time_start, cur_time_end):
                            trajfirst = INDEX[pp][cur_grid]
                            for qq in range(pp, cur_time_end):
                                trajsecond = INDEX[qq][next_grid]
                                SS.update(trajfirst & trajsecond)
                        transit_num = len(SS)
                        Transit[ii][jj] = transit_num
                    else:
                        pass
            FlattedTransit = Transit.reshape(-1).tolist()
            lineitem = [str(i), str(j)]
            lineitem.extend([str(t) for t in FlattedTransit])
            line = ','.join(lineitem) + '\n'
            transitWriteFile.write(line)
        print('genGridTransit timestamp: ', i)
    transitWriteFile.close()
    print('genGridTransit Ended: ', time.ctime())

def getGridTransit(mesh, gTrajFileName, transitFileName):
    GRIDNUMBER = mesh.lonNum * mesh.latNum
    Timestamps = getTimestamps(gTrajFileName)
    TIMENUMBER = len(Timestamps) - 1 # -1 is because of transit
    print('getGridTransit Started : ', time.ctime())
    R = []
    for i in range(TIMENUMBER):
        R.append([])
        for j in range(GRIDNUMBER):
            R[i].append([])
    with open(transitFileName, 'r') as rf:
        tansistReader = csv.reader(rf)
        for line in tansistReader:
            timestamp = int(line[0])
            grid = int(line[1])
            R[timestamp][grid] = line[2:]
    R = np.array(R, dtype='int32') # 144, 6000, 225
    R = R.reshape(R.shape[0], mesh.lonNum, mesh.latNum, R.shape[2])
    R = np.swapaxes(R, 2, 1)
    R = R[:, ::-1, :, :] # 144, 75, 80, 225
    return R

def getGridPop(mesh, gTrajFileName, popFileName):
    GRIDNUMBER = mesh.lonNum * mesh.latNum
    Timestamps = getTimestamps(gTrajFileName)
    TIMENUMBER = len(Timestamps)
    TS = {}
    for i in range(TIMENUMBER):
        TS[Timestamps[i]] = i
    print('getGridPop Started : ', time.ctime())
    R = []
    for i in range(TIMENUMBER):
        R.append([])
        for j in range(GRIDNUMBER):
            R[i].append([])
    with open(popFileName, 'r') as rf:
        tansistReader = csv.reader(rf)
        for line in tansistReader:
            timestamp = TS[line[0]]
            grid = int(line[1])
            R[timestamp][grid] = int(line[2])
    R = np.array(R, dtype='int32') # shape 145, 6000
    R = R.reshape(R.shape[0], int(R.shape[1] ** 0.5), int(R.shape[1] ** 0.5), 1)
    R = np.swapaxes(R, 2, 1)
    R = R[:, ::-1, :, :]  # shape 145, 80, 80, 1
    return R

def getGridPopPartition(R, M, K):
    # Original 8*8 matrix N = 8 = M*K
    # M = 4 # M*M sub matrix
    # K = 2 # each sub matrix has the size of K * K
    P = []
    for i in range(M):
        for j in range(M):
            P.append(R[:, i*K:i*K+K, j*K:j*K+K, :])
    return np.array(P)

def getGridPop2DNumpy(mesh, gTrajFileName, popFileName):
    GRIDNUMBER = mesh.lonNum * mesh.latNum
    Timestamps = getTimestamps(gTrajFileName)
    TIMENUMBER = len(Timestamps)
    TS = {}
    for i in range(TIMENUMBER):
        TS[Timestamps[i]] = i
    print('getGridPop Started : ', time.ctime())
    R = []
    for i in range(TIMENUMBER):
        R.append([])
        for j in range(GRIDNUMBER):
            R[i].append([])
    with open(popFileName, 'r') as rf:
        tansistReader = csv.reader(rf)
        for line in tansistReader:
            timestamp = TS[line[0]]
            grid = int(line[1])
            R[timestamp][grid] = int(line[2])
    R = np.array(R, dtype='int32')  # shape 145, 6000
    return R

def getGridPopTimeInterval(mesh, popFileName):
    print('getGridPop', popFileName, time.ctime())

    GRIDNUMBER = mesh.lonNum * mesh.latNum
    Timestamps = []
    lastTimestamp = ''
    with open(popFileName, 'r') as rf:
        tansistReader = csv.reader(rf)
        for line in tansistReader:
            timestamp = line[0]
            if timestamp != lastTimestamp:
                Timestamps.append(timestamp)
            lastTimestamp = timestamp

    TIMENUMBER = len(Timestamps)
    TS = {}
    for i in range(TIMENUMBER):
        TS[Timestamps[i]] = i

    R = []
    for i in range(TIMENUMBER):
        R.append([])
        for j in range(GRIDNUMBER):
            R[i].append([])
    with open(popFileName, 'r') as rf:
        tansistReader = csv.reader(rf)
        for line in tansistReader:
            timestamp = TS[line[0]]
            grid = int(line[1])
            R[timestamp][grid] = int(line[2])
    R = np.array(R, dtype='int32') # shape 145, 6000
    R = R.reshape(R.shape[0], int(R.shape[1] ** 0.5), int(R.shape[1] ** 0.5), 1)
    R = np.swapaxes(R, 2, 1)
    R = R[:, ::-1, :, :]  # shape 145, 75, 80, 1

    return R

def getGridTransitTimeInterval(mesh, transitFileName):
    print('getGridTransit Started : ', transitFileName, time.ctime())
    GRIDNUMBER = mesh.lonNum * mesh.latNum

    # Timestamps = []
    # lastTimestamp = ''
    # with open(transitFileName, 'r') as rf:
    #     tansistReader = csv.reader(rf)
    #     for line in tansistReader:
    #         timestamp = line[0]
    #         if timestamp != lastTimestamp:
    #             Timestamps.append(timestamp)
    #         lastTimestamp = timestamp
    # TIMENUMBER = len(Timestamps)

    TIMENUMBER = 24 * 12

    R = []
    for i in range(TIMENUMBER):
        R.append([])
        for j in range(GRIDNUMBER):
            R[i].append([])
    with open(transitFileName, 'r') as rf:
        tansistReader = csv.reader(rf)
        for line in tansistReader:
            timestamp = int(line[0])
            grid = int(line[1])
            R[timestamp][grid] = line[2:]
    R = np.array(R, dtype='int32') # 144, 6000, 225
    R = R.reshape(R.shape[0], mesh.lonNum, mesh.latNum, R.shape[2])
    R = np.swapaxes(R, 2, 1)
    R = R[:, ::-1, :, :] # 144, 75, 80, 225
    return R

def shuffleTrainValidateTest(InterpolatedStep, path, fileName, R, testRate=0.2):
    TIMESTEP  = InterpolatedStep * 2
    Sequence = []
    for i in range(R.shape[0] - TIMESTEP):
        Sequence.append(R[i:i+TIMESTEP, :, :, :])
    Sequence = np.array(Sequence, dtype='int32')
    INDEX = list(range(len(Sequence)))
    np.random.shuffle(INDEX)
    np.random.shuffle(INDEX)
    np.random.shuffle(INDEX)
    trainINDEX = INDEX[:int(len(INDEX) * (1 - testRate))]
    testINDEX = INDEX[int(len(INDEX) * (1 - testRate)):]
    train = Sequence[trainINDEX]
    test = Sequence[testINDEX]
    np.save(path + 'train_' + fileName, train)
    np.save(path + 'test_' + fileName, test)
    print(train.shape, test.shape)

    # trainINDEX = INDEX[:int(len(INDEX) * (1 - testRate - validateRate))]
    # validateINDEX = INDEX[int(len(INDEX) * (1 - testRate - validateRate)):int(len(INDEX) * (1 - testRate))]
    # testINDEX = INDEX[int(len(INDEX) * (1 - testRate)):]
    # train = Sequence[trainINDEX]
    # validate = Sequence[validateINDEX]
    # test = Sequence[testINDEX]
    # np.save(path + 'train_' + fileName, train)
    # np.save(path + 'validate_' + fileName, validate)
    # np.save(path + 'test_' + fileName, test)

    # print(train.shape, validate.shape, test.shape)
    # or directly return not save to file because just too big.
    # return train, validate, test

def getShuffledTrainTest(path, fileName, TrainTest):
      return np.load(path + TrainTest + '_' + fileName + '.npy')

def testcode(mesh):
    GRIDNUMBER = mesh.lonNum * mesh.latNum
    window = 5
    R = np.zeros((window, window), dtype='int32')
    center = mesh.ReverseIndex[(2,2)]
    current_x, current_y = mesh.Index[center]
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
    print(R)

    for i in range(len(R)):
        print(R[i])
    for i in range(len(R)):
        print(R[i][0], R[i][1], R[i][2], R[i][3], R[i][4])

    T = R.reshape(-1)
    print(T.tolist())

    P = T.reshape(window, window)
    print(P)

    print(R.shape)
    print(R[54][4178])
    print(np.max(R) == 3369)
    print(mesh.Index[3369])
    x, y = mesh.Index[3369]
    lon, lat = mesh.minLon + (x + 0.5) * mesh.dLon, \
               mesh.minLat + (y + 0.5) * mesh.dLat
    print(lon, lat)

    print(mesh.lonNum, mesh.latNum)
    T = np.array(range(GRIDNUMBER))
    T = T.reshape(mesh.lonNum, mesh.latNum)
    T = np.swapaxes(T, 1, 0)
    T = T[::-1, :]
    print(T)
    print(T.shape)


def run5min201802(mesh, dataPATH, dates):
    print('Now is getting trainig XS and YS...', dates)

    # timestamp = '2011-10-20 09:00:00'
    # filenameTime = timestamp[0:4] + timestamp[5:7] + timestamp[8:10] \
    #                + timestamp[11:13] + timestamp[14:16] + timestamp[17:19]
    # print(filenameTime)

    for date in dates:
        # first step: from trajectory point to mesh
        getMesh(dataPATH + date + 'tokyo_interpo5min.csv',
                dataPATH + date + 'tokyo_' + mesh.size + '_5min.csv')

        # second step: calculate mesh population at each timestamp
        genMeshDynamic(dataPATH + date + 'tokyo_' + mesh.size + '_5min.csv',
                       dataPATH + date + 'tokyo_' + mesh.size + '_5min_pop.csv')

        # fourth step: mesh transit between two consecutive timestamps
        genGridTransit(dataPATH + date + 'tokyo_' + mesh.size + '_5min.csv',
                       dataPATH + date + 'tokyo_' + mesh.size + '_5min_transit.csv')


def getHHTransit(HH):
    assert HH <= 22, 'Hour should not be over 22.'
    dataPATH = '../interpo_data/'
    date = '20111020'
    R = getGridTransit(dataPATH + date + 'tokyo_meshtransit10min_1min_15.csv')
    # (144, 72, 80, 225)
    R = R[HH*6:HH*6+6, :, :, :]
    # (6, 72, 80, 225)
    R = R.reshape(R.shape[0], -1, R.shape[-1])
    # (6, 5760, 225)
    R = R.transpose(1, 0, 2)
    # (5760, 6, 225)
    R = R.reshape(R.shape[0], R.shape[1], int(R.shape[2]**0.5), int(R.shape[2]**0.5), 1)
    return R

def runCrowdDensity():
    dataPATH = '../interpo_data/'
    meshTokyo = Mesh('tokyo', '500m')
    #meshcode_level = 4
    alldates = ["20110217","20110218","20110219","20110220", "20110221",
				"20110222","20110223", "20110224", "20110225", "20110226", "20110227"]
    for date in alldates:
        print('this is date', date)

        getMesh(meshTokyo, dataPATH + date + 'tokyo_interpo5min.csv',
                dataPATH + date + 'tokyo_' + meshTokyo.size + '_5min.csv')

        genMeshDynamic(meshTokyo, dataPATH + date + 'tokyo_' + meshTokyo.size + '_5min.csv',
                       dataPATH + date + 'tokyo_' + meshTokyo.size + '_5min_pop.csv')


# def runCrowdFlow_from5min():
#     from common.dataparam.Param import alldates
#     dataPATH = '../interpo_data/'
#     meshTokyo = Mesh('tokyo', '500m')
#     #meshcode_level = 4
#
#     for date in alldates:
#         print('this is date', date)
#         genGridTransit(meshTokyo,
#                        dataPATH + date + 'tokyo_' + meshTokyo.size + '_5min.csv',
#                        dataPATH + date + 'tokyo_' + meshTokyo.size + '_5min_transit_from5min.csv')

# paper crowd flow is from 1min.!!!!!!!!!!!!
def runCrowdFlow():
    dataPATH = '../interpo_data/'
    meshTokyo = Mesh('tokyo', '500m')
    #meshcode_level = 4

    alldates = ["20110217", "20110218", "20110219", "20110220", "20110221",
                "20110222", "20110223", "20110224", "20110225", "20110226", "20110227"]
    
    for date in alldates:
        print('this is date', date)
        getMesh(meshTokyo, dataPATH + date + 'tokyo_interpo1min.csv',
                dataPATH + date + 'tokyo_' + meshTokyo.size + '_1min.csv')

        genGridTransit_5minutes_from_1minute(meshTokyo,
                       dataPATH + date + 'tokyo_' + meshTokyo.size + '_1min.csv',
                       dataPATH + date + 'tokyo_' + meshTokyo.size + '_5min_transit.csv')

def main():
    runCrowdDensity()

if __name__ == '__main__':
    main()