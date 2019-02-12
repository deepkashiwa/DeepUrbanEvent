import jismesh.utils as ju
import numpy as np
import time
import csv

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
    return R

def testJISMesh():
    meshcode = ju.to_meshcode(35.658581, 139.745433, 3)
    print(meshcode)

    minLat, minLon = ju.to_meshpoint('53393599', 3, 0, 0)
    maxLat, maxLon = ju.to_meshpoint('53393599', 3, 1, 1)

    print(minLat, minLon)
    print(maxLat, maxLon)

    meshlevel = ju.to_meshlevel('53393599')
    print(meshlevel)

def main():
    dataPATH = '../interpo_data/'
    date = '20111020'
    meshcode_level = 4
    trajFileName = dataPATH + date + 'tokyo_interpo5min.csv'
    dynamicFileName = dataPATH + date + 'tokyo_5min_dynamic' + str(meshcode_level) + '.csv'
    getDynamicMesh_mobmap(trajFileName, dynamicFileName, meshcode_level)

if __name__ == '__main__':
    main()