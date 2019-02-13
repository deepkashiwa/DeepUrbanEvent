from decimal import Decimal, getcontext

class Trajectory(object):
    def __init__(self):
        self.oid = ''
        self.sequence = {}

    def __init__(self, oid):
        self.oid = oid
        self.sequence = {}


