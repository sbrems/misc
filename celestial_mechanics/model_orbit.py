from __future__ import print_function
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

import time
import ipdb



def project_orbit(omega,Omega,i,m1,m2,a,e,T,t0=0.):

    omega = math.radians(omega)
    Omega = math.radians(Omega)
    i     = math.radians(i)
    x,y,r,f = make_orbit(m1,m2,a,e,T,t0=t0)
    
    xyz = flip_orbit(x,y,omega,Omega,i)
    
    orbit1 = -m2/(m1+m2) * xyz
    orbit2 =  m1/(m1+m2) * xyz
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.axis('equal')
    #ax.plot(xyz[:,0],xyz[:,1],xyz[:,2],'black')
    ax.plot(orbit1[:,0],orbit1[:,1],orbit1[:,2],'red')
    ax.plot(orbit2[:,0],orbit2[:,1],orbit2[:,2],'green')
    #ax.plot(orbit1[:,1],orbit1[:,2],zdir='x',color='red')
    #ax.plot(orbit2[:,1],orbit2[:,2],zdir='x',color='green')
    ax.scatter(0,0,0,'b')
    ax.scatter(orbit1[0,0],orbit1[0,1],orbit1[0,2],'b')
    ax.set_xlabel('x (line of sight)')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    
    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array([xyz[:,0].max()-xyz[:,0].min(), xyz[:,1].max()-xyz[:,1].min(), xyz[:,2].max()-xyz[:,2].min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(xyz[:,0].max()+xyz[:,0].min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(xyz[:,1].max()+xyz[:,1].min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(xyz[:,2].max()+xyz[:,2].min())
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')
    #ax.plot_surface([-1,-1,1,1],[-1,1,-1,1],[0,0,0,0],cstride=1,rstride=1,color='red')
    #plot some arrows
    arrstart = np.array([[0.,],[0.,],[0.,]])
    arrend   = np.array([[1.,],[0.,],[0.,]])
    ax.quiver(arrstart[0]+arrend[0],arrstart[1]+arrend[1],arrstart[2]+arrend[2],arrend[0],arrend[1],arrend[2])

    plt.show()


def make_orbit(m1,m2,a,e,T,t0=0.,plot=False):
    n_points=2000
    times = np.linspace(0,T,num = n_points)
    x = np.full((n_points),np.nan)
    y = np.full((n_points),np.nan)
    r = np.full((n_points),np.nan)
    f = np.full((n_points),np.nan)
    for i,t in enumerate(times):
        x[i],y[i],r[i],f[i] = calc_point(m1,m2,a,e,T,t,t0=t0)
    if plot:
        plt.plot(x,y)
        plt.ylim(min(min(x),min(y)),max(max(x),max(y)))
        plt.xlim(min(min(x),min(y)),max(max(x),max(y)))
        plt.plot(x[0],y[0],'rx')
        plt.plot(x[-1],y[-1],'go')
        plt.show()
    
    return x,y,r,f
    
def calc_point(m1,m2,a,e,T,t,t0=0.,prec=1e-12):
    '''Gives the position of the bodies m1 and m2 at time t (in units of T).
    t_0 is the time of periastron passage.
    a is semimajor axis, e eccentricity. For projection use the
    flip_orbit routine. Notation is taken from Seagers Exoplanets.
    return x,y,r,f (=theta)'''
    #G = 6.67408e-11 #m3 kg-1 s-2
    
    n = 2.*math.pi / T
    M = n* (t - t0)
    #solving Keplers equation
    E = M
    E_old = M+1
    if e != 0:
        while abs((E-E_old)) > prec:
            delta_E = (E-e*math.sin(E)-M) / (1.-e*math.cos(E))
            E_old = E
            E -= delta_E
    #ipdb.set_trace()
    r = a*(1.-e * math.cos(E))
    if e != 0:
        f = math.acos(round( ((a*(1-e**2)-r) / (e*r)) ,int(-math.log10(prec)-1) ))
        if (((t-t0)%T)/T) > 0.5:
            f *= -1.
    else:
        f = (((t-t0)/T)%1)*2*math.pi
    x = r*math.cos(f)
    y = r*math.sin(f)

    return x,y,r,f


def flip_orbit(x,y,omega,Omega,i):
    n = len(x)
    v_out = np.full((n,3), np.nan)
    for ii in xrange(n):
        v_in = [x[ii],y[ii],0.]
        P = np.dot(np.dot(P_z(Omega),P_x(i)),P_z(omega))
        v_out[ii,:] = np.dot(P,v_in)
    return v_out
    
def P_x(phi):
    P = [[1,0            , 0],
         [0,math.cos(phi),-math.sin(phi)],
         [0,math.sin(phi), math.cos(phi)]]
    return P

def P_z(phi):
    P = [[math.cos(phi),-math.sin(phi),0],
         [math.sin(phi), math.cos(phi),0],
         [0            , 0            ,1]]
    return P
