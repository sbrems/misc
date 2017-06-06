from __future__ import print_function
from __future__ import division
import numpy as np


def hmsToDeg(hms,sep=":"):
    hmssplit = np.array(hms.split(sep),dtype=np.float64)
    if len(hmssplit) != 3:
        raise ValueError('Unknown format of RA. Maybe wrong seperator?,', ra)
    sign = np.sign(hmssplit[0])
    deg = hmssplit[0]*15. + sign* hmssplit[1]/4. + sign*hmssplit[2] / 240.
    
    return deg


def dmsToDeg(hms,sep=":"):
    hmssplit = np.array(hms.split(sep),dtype=np.float64)
    if len(hmssplit) != 3:
        raise ValueError('Unknown format of RA. Maybe wrong seperator?,', ra)
    sign =  np.sign(hmssplit[0])
    deg = hmssplit[0] + sign * hmssplit[1]/60. + sign * hmssplit[2] / 3600.
    return deg
    

