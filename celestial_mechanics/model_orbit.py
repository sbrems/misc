# Nomenclature based on Seagers Exoplanets book

from __future__ import print_function
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from mpl_toolkits.mplot3d import Axes3D
# import math

import time


def project_orbit(omega, Omega, i, m1, m2, a, e, T, t0=0., plot=True):

    omega = omega.to('rad')
    Omega = Omega.to('rad')
    i = i.to('rad')
    x, y, r, f = make_orbit(m1, m2, a, e, T, t0=t0)

    xyz = flip_orbit(x, y, omega, Omega, i)

    orbit1 = (-m2 / (m1 + m2) * xyz).to('AU')
    orbit2 = (m1 / (m1 + m2) * xyz).to('AU')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.axis('equal')
    ax.set_aspect('equal')
    # ax.plot(xyz[:,0],xyz[:,1],xyz[:,2],'black')
    ax.plot(orbit1[:, 0].value, orbit1[:, 1].value,
            orbit1[:, 2].value, 'red', label='Stellar orbit')
    ax.plot(orbit2[:, 0].value, orbit2[:, 1].value,
            orbit2[:, 2].value, 'green', label='Companion orbit')
    # ax.plot(orbit1[:,1],orbit1[:,2],zdir='x',color='red')
    # ax.plot(orbit2[:,1],orbit2[:,2],zdir='x',color='green')
    ax.scatter(0, 0, 0, 'b')
    ax.scatter(orbit1[0, 0].value, orbit1[0, 1].value,
               orbit1[0, 2].value, 'b')
    ax.set_xlabel('x [AU] ($\Omega$ reference direction)')
    ax.set_ylabel('y [AU]')
    ax.set_zlabel('z [AU] (Line of sight)')

    # Create cubic bounding box to simulate equal aspect ratio
    max_range = [xyz[:, 0].max() - xyz[:, 0].min(), xyz[:, 1].max() -
                 xyz[:, 1].min(), xyz[:, 2].max() - xyz[:, 2].min()]
    max_range = np.max((max_range * max_range[0].unit))
    Xb = 0.5 * max_range * \
        np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * \
        (xyz[:, 0].max() + xyz[:, 0].min())
    Yb = 0.5 * max_range * \
        np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * \
        (xyz[:, 1].max() + xyz[:, 1].min())
    Zb = 0.5 * max_range * \
        np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * \
        (xyz[:, 2].max() + xyz[:, 2].min())
    # Comment or uncomment following both lines to test the fake bounding box:

    for xb, yb, zb in zip(Xb.value, Yb.value, Zb.value):
        ax.plot([xb], [yb], [zb], 'w')
    # ax.plot_surface([-1,-1,1,1],[-1,1,-1,1],[0,0,0,0],cstride=1,rstride=1,color='red')
    # plot some arrows
    arrstart = np.array([[0., ], [0., ], [0., ]])
    arrend = np.array([[0., ], [0., ], [np.max(Xb.value) / 4., ]])
    ax.quiver(arrstart[0] + arrend[0], arrstart[1] + arrend[1],
              arrstart[2] + arrend[2], arrend[0], arrend[1], arrend[2],
              label='Line of sight')
    # also plot the projection on the xy plane
    ax.plot(orbit1[:, 0], orbit1[:, 1], 'r--', zs=-max_range)
    ax.plot(orbit2[:, 0], orbit2[:, 1], 'g--', zs=-max_range)
    ax.legend()
    plt.savefig('3d_orbit.pdf')
    if plot:
        plt.show()
    plt.close('all')
    # make the 2d projection
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(orbit1[:, 0], orbit1[:, 1], 'r', label='Stellar orbit')
    ax.plot(orbit2[:, 0], orbit2[:, 1], 'g', label='Companion orbit')
    ax.set_xlabel('x [AU] ($\Omega$ reference direction)')
    ax.set_ylabel('y [AU]')
    ax.set_aspect('equal')
    ax.legend()
    plt.savefig('projected_orbit.pdf')
    plt.close('all')
    return


def make_orbit(m1, m2, a, e, T, t0=0., plot=False):
    n_points = 5000
    times = np.linspace(0, T, num=n_points)
#    x = np.full((n_points),np.nan)
#    y = np.full((n_points),np.nan)
#    r = np.full((n_points),np.nan)
#    f = np.full((n_points),np.nan)
    x, y, r, f = [], [], [], []
    for i, t in enumerate(times):
        xyrf = calc_point(m1, m2, a, e, T, t, t0=t0)
        x.append(xyrf[0])
        y.append(xyrf[1])
        r.append(xyrf[2])
        f.append(xyrf[3])

    x = x * x[-1].unit
    y = y * y[-1].unit
    r = r * r[-1].unit
    f = f * f[-1].unit
    if plot:
        plt.plot(x, y)
        plt.ylim(min(min(x), min(y)).value, max(max(x), max(y)).value)
        plt.xlim(min(min(x), min(y)).value, max(max(x), max(y)).value)
        plt.plot(x[0], y[0], 'rx')
        plt.plot(x[-1], y[-1], 'go')
        plt.show()

    return x, y, r, f


def calc_point(m1, m2, a, e, T, t, t0=0., prec=1e-12):
    '''Gives the position of the bodies m1 and m2 at time t (in units of T).
    t_0 is the time of periastron passage.
    a is semimajor axis, e eccentricity. For projection use the
    flip_orbit routine. Notation is taken from Seagers Exoplanets.
    return x,y,r,f (=theta)'''
    # G = 6.67408e-11 #m3 kg-1 s-2

    n = 2. * np.pi / T * u.rad
    M = n * (t - t0)
    # solving Keplers equation
    E = M
    E_old = M + 1 * u.rad
    if e != 0.:
        while abs((E - E_old)) > prec:
            delta_E = (E - e * np.sin(E) - M) / (1. - e * np.cos(E))
            E_old = E
            E -= delta_E

    r = a * (1. - e * np.cos(E))
    if e != 0.:
        f = np.acos(round(((a * (1 - e**2) - r) / (e * r)),
                          int(-np.log10(prec) - 1))) * u.rad
        if (((t - t0) % T) / T) > 0.5:
            f *= -1.
    else:
        f = (((t - t0) / T) % 1) * 2 * np.pi * u.rad
    x = r * np.cos(f)
    y = r * np.sin(f)

    return x, y, r, f


def flip_orbit(x, y, omega, Omega, i):
    n = len(x)
    XYZ = []
    # np.full((n,3), np.nan)
    for ii in range(n):
        xyz = [x[ii], y[ii], 0. * x[ii].unit]
        xyz = xyz * xyz[0].unit
        P = np.dot(np.dot(P_z(Omega), P_x(i)), P_z(omega))
        XYZ.append(np.dot(P, xyz))
    return XYZ * XYZ[0].unit


def P_x(phi):
    P = [[1, 0, 0],
         [0, np.cos(phi), -np.sin(phi)],
         [0, np.sin(phi),  np.cos(phi)]]
    return P


def P_z(phi):
    P = [[np.cos(phi), -np.sin(phi), 0],
         [np.sin(phi),  np.cos(phi), 0],
         [0,  0, 1]]
    return P
