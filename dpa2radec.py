import numpy as np

def do(d,dd,pa,dpa):
    print('Input:\n Dist: %s+-%s mas/px, PA: %s+-%s deg'%(d,dd,pa,dpa))
    pa = pa*np.pi/180.
    dpa=dpa*np.pi/180.
    ra  = d * np.sin(pa)
    dec = d * np.cos(pa)
    ddec = np.abs(ra* np.sqrt((dd/d)**2 + ((np.cos(pa+dpa)-np.cos(pa))/(np.cos(pa)))**2))
    dra  = np.abs(dec*np.sqrt((dd/d)**2 + ((np.sin(pa+dpa)-np.sin(pa))/(np.sin(pa)))**2))
    print('Out:\n RA:%s +- %s mas, DEC: %s +- %s mas'%(ra,dra,dec,ddec))
    return ra,dra,dec,ddec


def inverse(ra,dra,dec,ddec):
    print('Input:\n Ra: {}+-{} mas/px, Dec: {}+-{} mas/\
px'.format(ra,dra,dec,ddec))
    pa = np.arctan2(ra,dec) *180/np.pi %360
    d  = np.sqrt(ra**2+dec**2)
    dpa= np.abs(dra/ d**2 *ddec - dec /d**2 *dra)
    dd = dra + ddec
    print('Orientation: 0 is up, positive PA to east (pos ra)')
    print('Out:\n Dist: {}+-{} mas/px, PA: {}+-{} deg'.format(d,dd,pa,dpa))
    return d,dd,pa,dpa
