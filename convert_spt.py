import matplotlib.pyplot as plt
import numpy as np
from numpy.lib import recfunctions as rfn

dat = np.genfromtxt('astrid_deb_full_16_02_02.csv',\
                    unpack=True,delimiter=',',names=True, dtype=None)

sptToInt = {'B5':1,'B7':2,'B8':3,'B9':4,'A0':5,'A1':6,'A2':7,'A3':8,'A4':9,\
            'A5':10,'A6':11,'A7':12,'A8':13,'A9':14,'F0':14.5,'F1':15,'F2':16,'F3':17,\
            'F4':18,'F5':19,'F6':20,'F7':21,'F8':22,'F9':23,'F10':24,\
            'G0':25,'G1':26,'G2':27,'G3':28,'G4':29,'G5':30,'G6':31,'G7':32,\
            'G8':33,'G9':34,'K0':35,'K1':36,'K2':37,'K3':38,'K4':39,'K5':40}
spT = dat['spT']
num_spT = []
for spec in spT:
#    print str(spec[0:2])=='A2',spec,sptToInt[spec[0:2]]
    num_spT.append(str(sptToInt[spec[0:2]]))
print num_spT
dat2 = rfn.append_fields(dat,names='num_spT',data=num_spT,usemask=False)

np.savetxt('astrid_deb_full_numSpT.csv',dat2,delimiter=',',fmt='%s')
