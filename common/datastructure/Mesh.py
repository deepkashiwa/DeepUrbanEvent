from decimal import Decimal, getcontext
import jismesh
import jismesh.utils as ju

'''
Tokyo 0.4 * 0.32, 80 * 80
Osaka 0.4 * 0.32 80 * 80
'''
class Mesh(object):
    def __init__(self, city, size):
        getcontext().prec = 8

        if size == '4000m':
            self.dLon = 0.04
            self.dLat = 0.03
        elif size == '1000m':
            self.dLon = 0.01
            self.dLat = 0.008
        elif size == '500m':
            self.dLon = 0.005
            self.dLat = 0.004
        elif size == '100m':
            self.dLon = 0.001
            self.dLat = 0.0008
        else:
            print('invalid mesh size...')
            assert False
        if city == 'tokyo':
            self.minLon = 139.5
            self.maxLon = 139.9
            self.minLat = 35.5
            # self.maxLat = 35.8
            self.maxLat = 35.82
        elif city == 'osaka':
            self.minLon = 135.3
            self.maxLon = 135.7
            self.minLat = 34.5
            self.maxLat = 34.82
        elif city == 'nagoya':
            self.minLon = 136.7
            self.maxLon = 137.1
            self.minLat = 35.0
            self.maxLat = 35.3
        else:
            print('invalid city name...')
            assert False

        self.lonNum = int((Decimal(self.maxLon) - Decimal(self.minLon)) / Decimal(self.dLon))
        self.latNum = int((Decimal(self.maxLat) - Decimal(self.minLat)) / Decimal(self.dLat))
        self.size = size

        ID = 0
        self.Index = {}
        self.ReverseIndex = {}
        for i in range(self.lonNum):
            for j in range(self.latNum):
                self.Index[ID] = (i,j)
                self.ReverseIndex[(i,j)] = ID
                ID += 1

    def inMesh(self, x, y):
        if x >= 0 and x < self.lonNum and y >= 0 and y < self.latNum:
            return True
        else:
            return False

    def inMeshPoint(self, point):
        if point.lon >= self.minLon and point.lon <= self.maxLon \
                and point.lat >= self.minLat and point.lat <= self.maxLat:
            return True
        else:
            return False

    def inWhichGrid(self, point):
        if self.inMeshPoint(point):
            x = int((Decimal(point.lon) - Decimal(self.minLon)) / Decimal(self.dLon))
            y = int((Decimal(point.lat) - Decimal(self.minLat)) / Decimal(self.dLat))
            if x == self.lonNum:
                x -= 1
            if y == self.latNum:
                y -= 1
            return self.ReverseIndex[(x,y)]
        else:
            return None

    def toJISMesh(self):
        JISMesh = []
        if self.size == '1000m':
            level = 3
        elif self.size == '500m':
            level = 4
        elif self.size == '250m':
            level = 5
        elif self.size == '100m':
            level = 6
        else:
            assert 'Invalid mesh size'

        for id in range(self.lonNum * self.latNum):
            x, y = self.Index[id]
            center_lon, center_lat = self.minLon + x * self.dLon + 0.5 * self.dLon, \
                                     self.minLat + y * self.dLat + 0.5 * self.dLat
            meshcode = jismesh.utils.to_meshcode(center_lat, center_lon, level)
            JISMesh.append(meshcode)

        return JISMesh

if __name__ == '__main__':
    mesh = Mesh('tokyo', '500m')

    from common.datastructure.Point import Point
    tokyoStation = Point(139.767030, 35.681193)
    print(mesh.inWhichGrid(tokyoStation))
    print(mesh.maxLat - mesh.minLat)
    print(mesh.dLon, mesh.dLat, mesh.minLon, mesh.maxLon, mesh.minLat, mesh.maxLat)
    print(mesh.lonNum, mesh.latNum, mesh.lonNum * mesh.latNum)
    print(mesh.ReverseIndex[(1,3)])
    print(mesh.Index[78][0])
    print(mesh.inMesh(79,79))
    print(mesh.Index[2])

