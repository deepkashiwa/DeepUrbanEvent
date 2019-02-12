

class Point(object):
    def __init__(self, lon, lat):
        self.lon = lon
        self.lat = lat

    def toString(self):
        return '(' + self.lon + ',' + self.lat + ')'
