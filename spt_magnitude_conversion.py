import numpy as np
import re #to split strings
import pickle
import os

#convert SpT to magnitude.
#MS V-mag: taken from Allens Astrophysical quantities
def rounddown10(x):
    return int(np.floor(x/10.0))*10
def roundhalf(x):
    return 0.5*round(float(x)/.5)

spt_allen_ms = [15,19,20,22,25,28,#O,B
                30,32,35,40,42,45,48,#A,F
                50,52,55,58,60,62,65,#G,K
                70,72,75]#M
spt_allen_I = [19,22,25,28,#O,B
               30,32,35,40,42,45,48,#A,F
               50,52,55,58,60,62,65,#G,K
               70,72,75]#M
spc2num = {'O':10,'B':20,'A':30,'F':40,'G':50,'K':60,'M':70,'NAN':np.nan}
lumc2num= {'0':0,0:0,'I':1,'II':2,'III':3,'IV':4,'V':5,'NAN':np.nan}
num2spc = {v: k for k,v in spc2num.items()}


def spt2vmag_ms(spc,subc):
    '''Provide Spectral class and Subclass separated in letters (O,B,A,...) 
    and subclass in numerical values. Returns absolute V magnitude. Based on 
    Allens Astrophysical quantities. From O5-M5. Other return NaN.'''
    try:
        spc_num = spc2num[spc.upper()]
    except:
        spc_num = np.nan
    if np.isfinite(subc):
        num_value = spc_num+subc
    else:
        num_value = spc_num+5.#use center of SpT given
    #now give the values for the main sequence V-magnitudes based on numerical Spectral type

    lumv= [-5.7,-4.5,-4.0,-2.45,-1.2,-0.25,#O,B
           0.65,1.3,1.95,2.7,3.6,3.5,4.0,#A,F
           4.4,4.7,5.1,5.5,5.9,6.4,7.35,#G,K
           8.8,9.9,12.3]#M

    vmag = np.interp(num_value,spt_allen_ms,lumv,left=np.nan,right=np.nan)
    if (not np.isfinite(vmag)) and (spc.upper() != 'NAN'):
        print('Warning!Interpolation only between O5 and M5. You gave', spc,subc)
    return vmag

def spt2vmag_I(spc,subc):
    '''Provide Spectral class and Subclass separated in letters (O,B,A,...) 
    and subclass in numerical values. Returns absolute V magnitude. Based on 
    Allens Astrophysical quantities. From O5-M5. Other return NaN.'''
    try:
        spc_num = spc2num[spc.upper()]
    except:
        spc_num = np.nan
    if np.isfinite(subc):
        num_value = spc_num+subc
    else:
        num_value = spc_num+5.#use center of SpT given
    #now give the values for the main sequence V-magnitudes based on numerical Spectral type
    
    lumv= [-6.5,-6.4,-6.2,-6.2,#O,B
           -6.3,-6.5,-6.6,-6.6,-6.6,-6.6,-6.5,#A,F
           -6.4,-6.3,-6.2,-6.1,-6.0,-5.9,-5.8,#G,K
           -5.6,-5.6,-5.6]#M
    
    vmag = np.interp(num_value,spt_allen_I,lumv,left=np.nan,right=np.nan)
    if (not np.isfinite(vmag)) and (spc.upper() != 'NAN'):
        print('Warning!Interpolation only between O9 and M5. You gave', spc,subc)
    return vmag

def bmv2spt_ms(bmv):
    '''Give B-V magnitude. Converting to SpT via Allens Astrophysical Quantities
    for Main Sequence stars.
    Returning spectral class and supclass.'''
    bmvx= [-0.33,-0.31,-0.3,-0.24,-0.17,-0.11,#O,B
           -0.02,0.05,0.15,0.3,0.35,0.44,0.52,#A,F
           0.58,0.63,0.68,0.74,0.81,0.91,1.15,#G,K
           1.40,1.49,1.64]#M
    spt_num = np.interp(bmv,bmvx,spt_allen_ms)
    spc = num2spc[rounddown10(spt_num)]
    subc= roundhalf(np.float(spt_num-rounddown10(spt_num)))
    
    return spc,subc


def bmv2spt_I(bmv):
    '''Give B-V magnitude. Converting to SpT via Allens Astrophysical Quantities
    for Main Sequence stars.
    Returning spectral class and subclass.'''
    bmvx= [-0.27,-0.17,-0.1,-0.03,#O,B
           -0.01,0.03,0.09,0.17,0.23,0.32,0.56,#A,F
           0.76,0.87,1.02,1.14,1.25,1.36,1.6,#G,K
           1.67,1.71,1.8]#M
    spt_num = np.interp(bmv,bmvx,spt_allen_I)
    spc = num2spc[rounddown10(spt_num)]
    subc= roundhalf(np.float(spt_num-rounddown10(spt_num)))
    
    return spc, subc


def manual_split(spt):
    '''Look at a list with manually entered sptypes'''
    spt = spt.strip()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    man_converter = pickle.load(open(os.path.join(dir_path,
                                                  "manual_conversions.p"),
                                                  "rb"))
    try:
        res = man_converter[spt]
        print('Already known exception. Converted {} to {}'.format(spt, res))
    except:
        print('Unknown spt '+spt+'. Please enter the value manually')
        while True:
            try:
                res = eval(input('Enter in the form "[10,5.,3]" for O5III. use [np.nan, np.nan, np.nan] for invalid values for {}\n'.format(spt)))
                if len(res) != 3:
                    raise ValueError('Sorry, that didnt seem a right input')
                else:
                    print('New entry: {} : {}'.format(spt, res))
                    break
            except ValueError:
                print('Try again')
                continue
        man_converter[spt] = res
        pickle.dump(man_converter, open(os.path.join(dir_path,
                                                     "manual_conversions.p"),
                                        "wb"))
    return res

def split_spt(spt, enter_manually=True):
    '''Converting type O3IV to [10,3,4] or [10,5,nan] if only O
    O is given (5 picked as center)'''
    spt_orig = spt
    # remove small characters since simbad only uses small ones
    spt = ''.join([x for x in spt if not spt.islower()])
    repl_cars = [" ","C","E","(",")","/",":","*","CN","_"]
    for car in repl_cars:
        spt = str(spt.replace(car, ""))
    split1 = [x for x in re.split('([OBAFGKMLTY])', spt.strip()) if x]  # split first letter
    if len(split1) == 1:
        res = [spc2num[split1[0]],5.,np.nan]
    elif len(split1) == 2:
        split2 = [x for x in re.split('([0-9]+\.?[0-9]?)', split1[1]) if x]
        if len(split2) == 2:
            try:
                res = [spc2num[split1[0]], np.float(split2[0]), lumc2num[split2[1]]]
            except KeyError:
                res = manual_split(spt_orig)
        elif len(split2) == 1:
            try:
                res = [spc2num[split1[0]], 5., lumc2num[split2[0]]]
            except:
                try:
                    res = [spc2num[split1[0]], np.float(split2[0]), np.nan]
                except:
                    res = manual_split(spt_orig)
        else:
            if enter_manually:
                res = manual_split(spt_orig)
            else:
                res = [np.nan]*3
                raise ValueError('Error.Cannot convert spectral type: {}'.format(spt_orig))
    else:
        if enter_manually:
            res = manual_split(spt_orig)
        else:
            res = [np.nan]*3
            raise ValueError('Error.Cannot convert spectral type: {}'.format(spt_orig))
    return res
