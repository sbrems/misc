#! /usr/bin/env python

"""Module with a dictionary and variables for storing constant parameters.

Usage
-----
from param import VLT_NACO
VLT_NACO['diam']

"""

VLT_SPHERE = {
    'latitude' : -24.627,
    'longitude' : -70.404,
    'plsc' : 0.01225,                       # plate scale [arcsec]/px
    'diam': 8.2,                            # telescope diameter [m]
              }

VLT_NACO = {
    'latitude' : -24.627,
    'longitude' : -70.404,
    'plsc': 0.027190,                       # plate scale [arcsec]/px
    'diam': 8.2,                            # telescope diameter [m]
    'lambdal' : 3.8e-6,                     # filter central wavelength [m] L band
    'camera_filter' : 'Ll',
    # header keywords
    'kw_categ' : 'HIERARCH ESO DPR CATG',   # header keyword for calibration frames: 'CALIB' / 'SCIENCE'
    'kw_type' : 'HIERARCH ESO DPR TYPE'     # header keyword for frame type: 'FLAT,SKY' / 'DARK' / 'OBJECT' / 'SKY'
            }

VLT_SINFONI = {
    'latitude' : -24.627,
    'longitude' : -70.404,
    # plsc: depending on the chosen mode, can also be: 0.125'' or 0.05''
    'plsc': 0.0125,                 # plate scale [arcsec]/px
    'diam': 8.2,                    # telescope diameter [m]
    'camera_filter' : 'H+K',
    'lambdahk': 1.95e-6,            # wavelength of the middle of H+K
    'lambdah': 1.65e-6,             # wavelength of the middle of H band
    'lambdak': 2.166e-6,            # wavelength of the Brackett gamma line (~middle of the K band)
    'spec_res':5e-4,                # spectral resolution in um (for H+K)
    # header keywords
    'kw_categ' : 'HIERARCH ESO DPR CATG',   # header keyword for calibration frames: 'CALIB' / 'SCIENCE'
    'kw_type' : 'HIERARCH ESO DPR TYPE'     # header keyword for frame type: 'FLAT,SKY' / 'DARK' / 'OBJECT' / 'SKY'
            }

LBT = {
    'latitude' : 32.70131,                  # LBT's latitude in degrees
    'longitude' : -109.889064,              # LBT's longitude in degrees
    'lambdal' : 3.47e-6,                    # central wavelenght L cont2' band [m]
    'plsc' : 0.0106,                        # plate scale [arcsec]/px
    'diam' : 8.4,                           # telescope diameter [m]
    # header keywords
    'lst': 'LBT_LST',                       # Local sidereal Time header keyword
    'ra' : 'LBT_RA',                        # right ascension header keyword
    'dec' : 'LBT_DEC',                      # declination header keyword
    'altitude' : 'LBT_ALT',                 # altitude header keyword
    'azimuth' : 'LBT_AZ',                   # azimuth header keyword
    'exptime' : 'EXPTIME',                  # nominal total integration time per pixel header keyword
    'acqtime' : 'ACQTIME',                  # total controller acquisition time header keyword    
    'filter' : 'LMIR_FW2'                   # filter
            }

KECK_NIRC2 = {
    'plsc_narrow' : 0.009942,               # plate scale [arcsec]/px, narrow camera
    'plsc_medium' : 0.019829,               # plate scale [arcsec]/px, medium camera
    'plsc_wide' : 0.039686,                 # plate scale [arcsec]/px, wide camera   
    'latitude' : 19.82636,                  # Keck's latitude in degrees
    # header keywords
    'camera_name' : 'CAMNAME'               # camera name bwt narrow, medium and wide
            }


