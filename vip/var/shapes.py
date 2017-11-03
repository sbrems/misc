#! /usr/bin/env python

"""
Module with various functions.
"""

from __future__ import division

__author__ = 'C. Gomez @ ULg'
__all__ = ['dist',
           'frame_center',
           'get_square',
           'get_square_robust',
           'get_circle',
           'get_ellipse',
           'get_annulus',
           'get_annulus_quad',
           'get_annulus_cube',
           'get_ell_annulus',
           'mask_circle',
           'create_ringed_spider_mask']

import numpy as np
from skimage.draw import polygon, circle


def create_ringed_spider_mask(im_shape, ann_out, ann_in=0, sp_width=10, sp_angle=0):
    """
    Mask out information is outside the annulus and inside the spiders (zeros).

    Parameters
    ----------
    im_shape : tuple of int
        Tuple of length two with 2d array shape (Y,X).
    ann_out : int
        Outer radius of the annulus.
    ann_in : int
        Inner radius of the annulus.
    sp_width : int
        Width of the spider arms (3 branches).
    sp_angle : int
        angle of the first spider arm (on the positive horizontal axis) in
        counter-clockwise sense.

    Returns
    -------
    mask : array_like
        2d array of zeros and ones.

    """
    mask = np.zeros(im_shape)

    s = im_shape[0]
    r = s/2.
    theta = np.arctan2(sp_width/2., r)

    t0 = np.array([theta,np.pi-theta,np.pi+theta,np.pi*2.-theta])
    t1 = t0 + sp_angle/180. * np.pi
    t2 = t1 + np.pi/3.
    t3 = t2 + np.pi/3.

    x1 = r * np.cos(t1) + s/2.
    y1 = r * np.sin(t1) + s/2.
    x2 = r * np.cos(t2) + s/2.
    y2 = r * np.sin(t2) + s/2.
    x3 = r * np.cos(t3) + s/2.
    y3 = r * np.sin(t3) + s/2.

    rr1, cc1 = polygon(y1, x1)
    rr2, cc2 = polygon(y2, x2)
    rr3, cc3 = polygon(y3, x3)

    cy, cx = frame_center(mask)
    rr0, cc0 = circle(cy, cx, min(ann_out, cy))
    rr4, cc4 = circle(cy, cx, ann_in)

    mask[rr0,cc0] = 1
    mask[rr1,cc1] = 0
    mask[rr2,cc2] = 0
    mask[rr3,cc3] = 0
    mask[rr4,cc4] = 0
    return mask


def dist(yc,xc,y1,x1):
    """ Returns the Euclidean distance between two points.
    """
    return np.sqrt((yc-y1)**2+(xc-x1)**2)


def frame_center(array, verbose=False):
    """ Returns the coordinates y,x of a frame central pixel if the sides are 
    odd numbers. Python uses 0-based indexing, so the coordinates of the central
    pixel of a 5x5 pixels frame are (2,2). Those are as well the coordinates of
    the center of that pixel (sub-pixel center of the frame).
    """   
    y = array.shape[0]/2.       
    x = array.shape[1]/2.
    
    # If frame size is even
    if array.shape[0]%2==0:
        cy = np.ceil(y)
    else:
        cy = np.ceil(y) - 1    # side length/2 - 1, python has 0-based indexing
    if array.shape[1]%2==0:
        cx = np.ceil(x)
    else:
        cx = np.ceil(x) - 1

    cy = int(cy); cx = int(cx)
    if verbose:
        print 'Center px coordinates at x,y = ({:},{:})'.format(cy, cx)
    return cy, cx

    
def get_square(array, size, y, x, position=False):                 
    """ Returns an square subframe. 
    
    Parameters
    ----------
    array : array_like
        Input frame.
    size : int, odd
        Size of the subframe.
    y, x : int
        Coordinates of the center of the subframe.
    position : {False, True}, optional
        If set to True return also the coordinates of the bottom-left vertex.
        
    Returns
    -------
    array_view : array_like
        Sub array.
        
    """
    if not array.ndim==2:
        raise TypeError('Input array is not a frame or 2d array.')
    
    if size%2!=0:  size -= 1 # making it odd to get the wing
    wing = int(size/2)
    # wing is added to the sides of the subframe center. Note the +1 when 
    # closing the interval (python doesn't include the endpoint)
    array_view = array[int(y-wing):int(y+wing+1),
                       int(x-wing):int(x+wing+1)].copy()
    
    if position:
        return array_view, y-wing, x-wing
    else:
        return array_view


def get_square_robust(array, size, y, x, position=False, 
                      out_borders='reduced_square', return_wings=False,
                      strict=False):                 
    """ 
    Returns a square subframe from a larger array robustly (different options in
    case the requested subframe outpasses the borders of the larger array.
    
    Parameters
    ----------
    array : array_like
        Input frame.
    size : int, ideally odd
        Size of the subframe to be returned.
    y, x : int
        Coordinates of the center of the subframe.
    position : bool, {False, True}, optional
        If set to True, returns also the coordinates of the left bottom vertex.
    out_borders: string {'reduced_square','rectangular', 'whatever'}, optional
        Option that set what to do if the provided size is such that the 
        sub-array exceeds the borders of the array:
            - 'reduced_square' (default) -> returns a smaller square sub-array: 
            the biggest that fits within the borders of the array (warning msg)
            - 'rectangular' -> returns a cropped sub-array with only the part 
            that fits within the borders of the array; thus a rectangle (warning
            msg)
            - 'whatever' -> returns a square sub-array of the requested size, 
            but filled with zeros where it outpasses the borders of the array 
            (warning msg)
    return_wings: bool, {False,True}, optional
        If True, the function only returns the size of the sub square
        (this can be used to test that there will not be any size reduction of 
        the requested square beforehand)
    strict: bool, {False, True}, optional
        Set to True when you want an error to be raised if the size is not an 
        odd number. Else, the subsquare will be computed even if size is an even
        number. In the later case, the center is placed in such a way that 
        frame_center function of the sub_array would give the input center 
        (at pixel = half dimension minus 1).
        
    Returns
    -------
    default:
    array_view : array_like
        Sub array of the requested dimensions (or smaller depending on its 
        location in the original array and the selected out_borders option)

    if position is set to True and return_wing to False: 
    array_view, y_coord, x_coord: array_like, int, int 
        y_coord and x_coord are the indices of the left bottom vertex

    if return_wing is set to True: 
    wing: int
        the semi-size of the square in agreement with the out_borders option
    """
    if not array.ndim == 2:
        raise TypeError('Input array is not a frame or 2d array.')
    
    n_y = array.shape[0]
    n_x = array.shape[1]

    if strict:
        if size%2==0: 
            raise ValueError('The given size of the sub-square should be odd.')
        wing_bef = int((size-1)/2)
        wing_aft = wing_bef
    else:
        if size%2==0:
            wing_bef = (size/2)-1
            wing_aft = size/2
        else:
            wing_bef = int((size-1)/2)
            wing_aft = wing_bef

    #Consider the case of the sub-array exceeding the array
    if (y-wing_bef < 0 or y+wing_aft+1 >= n_y or x-wing_bef < 0 or 
        x+wing_aft+1 >= n_x):
        if out_borders=='reduced_square':
            wing_bef = min(y,x,n_y-1-y,n_x-1-x)
            wing_aft = wing_bef
            msg = "!!! WARNING: The size of the square sub-array was reduced"+\
                  " to fit within the borders of the array. Now, wings = "+\
                  str(wing_bef)+"px x "+str(wing_aft)+ "px !!!"
            print msg
        elif out_borders=='rectangular':
            wing_y = min(y,n_y-1-y)
            wing_x = min(x,n_x-1-x)
            y_init = y-wing_y
            y_fin = y+wing_y+1
            x_init = x-wing_x
            x_fin = x+wing_x+1
            array_view = array[int(y_init):int(y_fin), int(x_init):int(x_fin)]
            msg = "!!! WARNING: The square sub-array was changed to a "+\
                  "rectangular sub-array to fit within the borders of the "+\
                  "array. Now, [y_init,yfin]= ["+ str(y_init)+", "+ str(y_fin)+\
                  "] and [x_init,x_fin] = ["+ str(x_init)+", "+ str(x_fin)+ "]."
            print msg
            if position:
                return array_view, y_init, x_init
            else:
                return array_view
        else:
            msg = "!!! WARNING: The square sub-array was not changed but it"+\
                  " exceeds the borders of the array."
            print msg

    if return_wings: return wing_bef,wing_aft
    
    else:
        # wing is added to the sides of the subframe center. Note the +1 when 
        # closing the interval (python doesn't include the endpoint)
        array_view = array[int(y-wing_bef):int(y+wing_aft+1),
                           int(x-wing_bef):int(x+wing_aft+1)].copy()
        if position: return array_view, y-wing_bef, x-wing_bef
        else: return array_view


def get_circle(array, radius, output_values=False, cy=None, cx=None):           
    """Returns a centered circular region from a 2d ndarray. All the rest 
    pixels are set to zeros. 
    
    Parameters
    ----------
    array : array_like
        Input 2d array or image. 
    radius : int
        The radius of the circular region.
    output_values : {False, True}
        Sets the type of output.
    cy, cx : int
        Coordinates of the circle center.
        
    Returns
    -------
    values : array_like
        1d array with the values of the pixels in the circular region.
    array_masked : array_like
        Input array with the circular mask applied.
    """
    if not array.ndim == 2:
        raise TypeError('Input array is not a frame or 2d array.')
    sy, sx = array.shape
    if not cy and not cx:
        cy, cx = frame_center(array, verbose=False)
         
    yy, xx = np.ogrid[:sy, :sx]                                                 # ogrid is a multidim mesh creator (faster than mgrid)
    circle = (yy - cy)**2 + (xx - cx)**2                                        # eq of circle. squared distance to the center                                        
    circle_mask = circle < radius**2                                            # mask of 1's and 0's                                       
    if output_values:
        values = array[circle_mask]
        return values
    else:
        array_masked = array*circle_mask
        return array_masked


def get_ellipse(array, a, b, PA, output_values=False, cy=None, cx=None,
                output_indices=False):
    """ Returns a centered elliptical region from a 2d ndarray. All the rest 
    pixels are set to zeros.
    
    Parameters
    ----------
    array : array_like
        Input 2d array or image. 
    a : float or int
        Semi-major axis.
    b : float or int
        Semi-minor axis.
    PA : deg, float
        The PA of the semi-major axis.
    output_values : {False, True}, optional
        If True returns the values of the pixels in the annulus.
    cy, cx : int
        Coordinates of the circle center.
    output_indices : {False, True}, optional
        If True returns the indices inside the annulus.
    
    Returns
    -------
    Depending on output_values, output_indices:
    values : array_like
        1d array with the values of the pixels in the circular region.
    array_masked : array_like
        Input array with the circular mask applied.
    y, x : array_like
        Coordinates of pixels in circle.
    """

    if not array.ndim == 2:
        raise TypeError('Input array is not a frame or 2d array.')
    sy, sx = array.shape
    if not cy and not cx:
        cy, cx = frame_center(array, verbose=False)

    # Definition of other parameters of the ellipse
    f = np.sqrt(a ** 2 - b ** 2)  # distance between center and foci of the ellipse
    PA_rad = np.deg2rad(PA)
    pos_f1 = (cy + f * np.cos(PA_rad), cx + f * np.sin(PA_rad))  # coords of first focus
    pos_f2 = (cy - f * np.cos(PA_rad), cx - f * np.sin(PA_rad))  # coords of second focus

    yy, xx = np.ogrid[:sy, :sx]
    # ogrid is a multidim mesh creator (faster than mgrid)
    ellipse = dist(yy, xx, pos_f1[0], pos_f1[1]) + dist(yy, xx, pos_f2[0],
                                                        pos_f2[1])
    ellipse_mask = ellipse < 2 * a  # mask of 1's and 0's

    if output_values and not output_indices:
        values = array[ellipse_mask]
        return values
    elif output_indices and not output_values:
        indices = np.array(np.where(ellipse_mask))
        y = indices[0]
        x = indices[1]
        return y, x
    elif output_indices and output_values:
        msg = 'output_values and output_indices cannot be both True.'
        raise ValueError(msg)
    else:
        array_masked = array * ellipse_mask
        return array_masked


def get_annulus(array, inner_radius, width, output_values=False, 
                output_indices=False):                                          
    """Returns a centerered annulus from a 2d ndarray. All the rest pixels are 
    set to zeros. 
    
    Parameters
    ----------
    array : array_like
        Input 2d array or image. 
    inner_radius : float
        The inner radius of the donut region.
    width : int
        The size of the annulus.
    output_values : {False, True}, optional
        If True returns the values of the pixels in the annulus.
    output_indices : {False, True}, optional
        If True returns the indices inside the annulus.
    
    Returns
    -------
    Depending on output_values, output_indices:
    values : array_like
        1d array with the values of the pixels in the annulus.
    array_masked : array_like
        Input array with the annular mask applied.
    y, x : array_like
        Coordinates of pixels in annulus.
    """
    if not array.ndim == 2:
        raise TypeError('Input array is not a frame or 2d array.')
    array = array.copy()
    cy, cx = frame_center(array)
    yy, xx = np.mgrid[:array.shape[0], :array.shape[1]]
    circle = np.sqrt((xx - cx)**2 + (yy - cy)**2)                                                                               
    donut_mask = (circle <= (inner_radius + width)) & (circle >= inner_radius)
    if output_values and not output_indices:
        values = array[donut_mask]
        return values
    elif output_indices and not output_values:      
        indices = np.array(np.where(donut_mask))
        y = indices[0]
        x = indices[1]
        return y, x
    elif output_indices and output_values:
        raise ValueError('output_values and output_indices cannot be both True.')
    else:
        array_masked = array*donut_mask
        return array_masked
    

def get_annulus_quad(array, inner_radius, width, output_values=False):                                          
    """ Returns indices or values in quadrants of a centerered annulus from a 
    2d ndarray. 
    
    Parameters
    ----------
    array : array_like
        Input 2d array or image. 
    inner_radius : int
        The inner radius of the donut region.
    width : int
        The size of the annulus.
    output_values : {False, True}, optional
        If True returns the values of the pixels in the each quadrant instead
        of the indices.
    
    Returns
    -------
    Depending on output_values:
    values : array_like with shape [4, npix]
        Array with the values of the pixels in each quadrant of annulus.
    ind : array_like with shape [4,2,npix]
        Coordinates of pixels for each quadrant in annulus.
        
    """
    if not array.ndim == 2:
        raise TypeError('Input array is not a frame or 2d array.')
    
    cy, cx = frame_center(array)
    xx, yy = np.mgrid[:array.shape[0], :array.shape[1]]
    circle = np.sqrt((xx - cx)**2 + (yy - cy)**2)                                                                               
    q1 = (circle >= inner_radius) & (circle <= (inner_radius + width)) & (xx >= cx) & (yy <= cy)  
    q2 = (circle >= inner_radius) & (circle <= (inner_radius + width)) & (xx <= cx) & (yy <= cy)
    q3 = (circle >= inner_radius) & (circle <= (inner_radius + width)) & (xx <= cx) & (yy >= cy)
    q4 = (circle >= inner_radius) & (circle <= (inner_radius + width)) & (xx >= cx) & (yy >= cy)
    
    if output_values:
        values = [array[mask] for mask in [q1,q2,q3,q4]]
        return np.array(values)
    else:      
        ind = [np.array(np.where(mask)) for mask in [q1,q2,q3,q4]]          
        return np.array(ind)

    
def get_annulus_cube(array, inner_radius, width, output_values=False):     
    """ Returns a centerered annulus from a 3d ndarray. All the rest pixels are 
    set to zeros. 
    
    Parameters
    ----------
    array : array_like
        Input 2d array or image. 
    inner_radius : int
        The inner radius of the donut region.
    width : int
        The size of the annulus.
    output_values : {False, True}, optional
        If True returns the values of the pixels in the annulus.
    
    Returns
    -------
    Depending on output_values:
    values : array_like
        1d array with the values of the pixels in the circular region.
    array_masked : array_like
        Input array with the annular mask applied.

    """
    if not array.ndim == 3:
        raise TypeError('Input array is not a cube or 3d array.')
    arr_annulus = np.empty_like(array)
    if output_values:
        values = []
        for i in range(array.shape[0]):
            values.append(get_annulus(array[i], inner_radius, width,
                                      output_values=True))
        return np.array(values)
    else:
        for i in range(array.shape[0]):
            arr_annulus[i] = get_annulus(array[i], inner_radius, width)
        return arr_annulus
    

def get_ell_annulus(array, a, b, PA, width, output_values=False,
                    output_indices=False, cy=None, cx=None):
    """Returns a centered elliptical annulus from a 2d ndarray. All the rest 
    pixels are set to zeros. 

    Parameters
    ----------
    array : array_like
        Input 2d array or image. 
    a : flt
        Semi-major axis.
    b : flt
        Semi-minor axis.
    PA : deg
        The PA of the semi-major axis.
    width : flt
        The size of the annulus along the semi-major axis; it is proportionnally 
        thinner along the semi-minor axis).
    output_values : {False, True}, optional
        If True returns the values of the pixels in the annulus.
    output_indices : {False, True}, optional
        If True returns the indices inside the annulus.
    cy,cx: float, optional
        Location of the center of the annulus to be defined. If not provided, 
    it assumes the annuli are centered on the frame.

    Returns
    -------
    Depending on output_values, output_indices:
    values : array_like
        1d array with the values of the pixels in the annulus.
    array_masked : array_like
        Input array with the annular mask applied.
    y, x : array_like
        Coordinates of pixels in annulus.

    """
    if not array.ndim == 2:
        raise TypeError('Input array is not a frame or 2d array.')
    if cy is None or cx is None:
        cy, cx = frame_center(array)
    sy, sx = array.shape

    width_a = width
    width_b = width * b / a

    # Definition of big ellipse
    f_big = np.sqrt((a + width_a / 2.) ** 2 - (
    b + width_b / 2.) ** 2)  # distance between center and foci of the ellipse
    PA_rad = np.deg2rad(PA)
    pos_f1_big = (cy + f_big * np.cos(PA_rad),
                  cx + f_big * np.sin(PA_rad))  # coords of first focus
    pos_f2_big = (cy - f_big * np.cos(PA_rad),
                  cx - f_big * np.sin(PA_rad))  # coords of second focus

    # Definition of small ellipse
    f_sma = np.sqrt((a - width_a / 2.) ** 2 - (
    b - width_b / 2.) ** 2)  # distance between center and foci of the ellipse
    pos_f1_sma = (cy + f_sma * np.cos(PA_rad),
                  cx + f_sma * np.sin(PA_rad))  # coords of first focus
    pos_f2_sma = (cy - f_sma * np.cos(PA_rad),
                  cx - f_sma * np.sin(PA_rad))  # coords of second focus

    yy, xx = np.ogrid[:sy, :sx]
    big_ellipse = dist(yy, xx, pos_f1_big[0], pos_f1_big[1]) + dist(yy, xx,
                                                                    pos_f2_big[
                                                                        0],
                                                                    pos_f2_big[
                                                                        1])
    small_ellipse = dist(yy, xx, pos_f1_sma[0], pos_f1_sma[1]) + dist(yy, xx,
                                                                      pos_f2_sma[
                                                                          0],
                                                                      pos_f2_sma[
                                                                          1])
    ell_ann_mask = (big_ellipse < 2 * (a + width / 2.)) & (
    small_ellipse >= 2 * (a - width / 2.))  # mask of 1's and 0's

    if output_values and not output_indices:
        values = array[ell_ann_mask]
        return values
    elif output_indices and not output_values:
        indices = np.array(np.where(ell_ann_mask))
        y = indices[0]
        x = indices[1]
        return y, x
    elif output_indices and output_values:
        msg = 'output_values and output_indices cannot be both True.'
        raise ValueError(msg)
    else:
        array_masked = array * ell_ann_mask
        return array_masked


def mask_circle(array, radius):                                      
    """ Masks (sets pixels to zero) a centered circle from a frame or cube. 
    
    Parameters
    ----------
    array : array_like
        Input frame or cube.
    radius : int
        Radius of the circular aperture.
    
    Returns
    -------
    array_masked : array_like
        Masked frame or cube.
        
    """
    if len(array.shape) == 2:
        sy, sx = array.shape
        cy = sy/2
        cx = sx/2
        xx, yy = np.ogrid[:sy, :sx]
        circle = (xx - cx)**2 + (yy - cy)**2    # squared distance to the center
        hole_mask = circle > radius**2                                             
        array_masked = array*hole_mask
        
    if len(array.shape) == 3:
        n, sy, sx = array.shape
        cy = sy/2
        cx = sx/2
        xx, yy = np.ogrid[:sy, :sx]
        circle = (xx - cx)**2 + (yy - cy)**2    # squared distance to the center
        hole_mask = circle > radius**2      
        array_masked = np.empty_like(array)
        for i in range(n):
            array_masked[i] = array[i]*hole_mask
        
    return array_masked    
    

        

