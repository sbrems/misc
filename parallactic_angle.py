import math
import astropy.units as u

def get_pa(header,verbose=True):
    CD11 = float(header['CD1_1'])
    CD12 = float(header['CD1_2'])
    CD21 = float(header['CD2_1'])
    CD22 = float(header['CD2_2'])
    
    ## This is my code to interpet the CD matrix in the WCS and determine the
    ## image orientation (position angle) and flip status.  I've used it and it
    ## seems to work, but there are some edge cases which are untested, so it
    ## might fail in those cases.
    ## Note: I'm using astropy units below, you can strip those out if you keep
    ## track of degrees and radians manually.
    if (abs(CD21) > abs(CD22)) and (CD21 >= 0): 
        North = "Right"
        positionAngle = 270.*u.deg + math.degrees(math.atan(CD22/CD21))*u.deg
    elif (abs(CD21) > abs(CD22)) and (CD21 < 0):
        North = "Left"
        positionAngle = 90.*u.deg + math.degrees(math.atan(CD22/CD21))*u.deg
    elif (abs(CD21) < abs(CD22)) and (CD22 >= 0):
        North = "Up"
        positionAngle = 0.*u.deg + math.degrees(math.atan(CD21/CD22))*u.deg
    elif (abs(CD21) < abs(CD22)) and (CD22 < 0):
        North = "Down"
        positionAngle = 180.*u.deg + math.degrees(math.atan(CD21/CD22))*u.deg
    if (abs(CD11) > abs(CD12)) and (CD11 > 0): East = "Right"
    if (abs(CD11) > abs(CD12)) and (CD11 < 0): East = "Left"
    if (abs(CD11) < abs(CD12)) and (CD12 > 0): East = "Up"
    if (abs(CD11) < abs(CD12)) and (CD12 < 0): East = "Down"
    if North == "Up" and East == "Left": imageFlipped = False
    if North == "Up" and East == "Right": imageFlipped = True
    if North == "Down" and East == "Left": imageFlipped = True
    if North == "Down" and East == "Right": imageFlipped = False
    if North == "Right" and East == "Up": imageFlipped = False
    if North == "Right" and East == "Down": imageFlipped = True
    if North == "Left" and East == "Up": imageFlipped = True
    if North == "Left" and East == "Down": imageFlipped = False
    if verbose: print("Position angle of WCS is {0:.1f} degrees.".format(positionAngle.to(u.deg).value))
    if verbose: print("Image orientation is North {0}, East {1}.".format(North, East))
    if imageFlipped and verbose:
        print("Image is mirrored.")
    
    ## Now you have position angle and flip status and can mark up your image
    return positionAngle.to(u.deg).value
