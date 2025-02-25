import numpy

def calcAxisAngle(axis_position):
    if axis_position[0] is 0.0:
        return 0.0
    else:
        return numpy.atan(axis_position[1]/axis_position[0])