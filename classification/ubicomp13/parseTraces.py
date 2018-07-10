import os
import struct
import math

def parseBinary(filename):
   timestamps = []
   sensorTypes = []
   readings = list()
   with open(filename, 'rb') as f:
        while True:
           ts_f = f.read(8)
           st_f = f.read(1)
           x_f = f.read(4) 
           y_f = f.read(4)
           z_f = f.read(4)
           if ts_f and st_f and x_f and y_f and z_f:
             ts=struct.unpack('>q',ts_f)[0] #long long
             timestamps.append(ts)
             st=ord(struct.unpack('>c',st_f)[0]) #char to int
             sensorTypes.append(st)
             x = struct.unpack('>f',x_f)[0]
             y = struct.unpack('>f',y_f)[0]
             z = struct.unpack('>f',z_f)[0]
             readings.append([x, y, z])
           else:
               break
        timestamps =  [x/1000000000. for x in timestamps] #convert to seconds
   return (timestamps, sensorTypes, readings)

def parseTxt(filename):
   timestamps = []
   sensorTypes= []
   readings= []
   f = open(filename,"r")

   for l in f.readlines():
       d = l.split()
       timestamps.append(long(d[0])/1000000000.)
       sensorTypes.append(int(d[1]))
       readings.append([float(d[2]), float(d[3]), float(d[4])])
   return (timestamps, sensorTypes, readings)


"""
Extracts all sensor readings pertaining to the particular sensor (identified by its senType).
"""
def extract_type(timestamps, sensorTypes, readings, senType):
   ts = [i[1] for i in enumerate(timestamps) if sensorTypes[i[0]]==senType]
   val =[i[1] for i in enumerate(readings) if sensorTypes[i[0]]==senType]
   return (ts, val)

def parseTrace(filename):

    if os.path.islink(filename): #handle symlinks
        filename=os.readlink(filename)

    if filename.rpartition('.')[2]=='txt' or filename.rpartition('.')[2]=='dat':
        (timestamps, sensorTypes, readings) = parseTxt(filename)
    elif filename.rpartition('.')[2]=='out':
        (timestamps, sensorTypes, readings) = parseBinary(filename)
    else:
        raise ValueError('Unsupported file type')

    (accTs, accData) = extract_type(timestamps, sensorTypes, readings, 1)
    (gyroTs, gyroData) = extract_type(timestamps, sensorTypes, readings, 2)
    (magnTs, magnData) = extract_type(timestamps, sensorTypes, readings, 3)

    #all returned timestamps start from 0
    accTs = [x - accTs[0] for x in accTs]
    gyroTs = [x - gyroTs[0] for x in gyroTs] 
    magnTs = [x - magnTs[0] for x in magnTs]

    return (accTs, accData, gyroTs, gyroData, magnTs, magnData)

"""
Returns the magnitude of acceleration of the signal.
"""
def getAccMagn(filename):
    ats, accData, _, _, _, _ = parseTrace(filename)
    a=[math.sqrt(x*x+y*y+z*z) for x,y,z in accData]
    return ats, a

