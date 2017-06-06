import numpy as np
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
import astropy.units as u

from spt_magnitude_conversion import split_spt

class Star:
        '''Save properties of a star. So far available:
        aliases (auto from Simbad)
        Coordinates (=['auto',] for Simbad query),
        coord_units: default: ['u.hourangle','u.deg'],
        default coord_frame ='icrs'
        Magnitudes (with the filterband, default filterband = None)
        SpT (also num)
        aliases
        PM
        parallax
        '''
        _kind = 'star'
        
        def __init__(self, name,
                     coordinates=[None,[u.hourangle, u.deg],'icrs'],
                     aliases = None,
                     mag = np.array([None,]), filterband = np.array([None,]),
                     SpT=None, SpC=None):
           self.name = name
           self.SpT          = SpT

           self._aliases = aliases
           self._coordinates = coordinates
           self._mag         = mag
           self._filterband  = filterband

        @property
        def coordinates(self):
            return self._coordinates

        @coordinates.setter
        def coordinates(self,coords):
            if coords[0] == 'auto' or coords =='auto':
                coords = Simbad.query_object(self.name)['RA','DEC']
                coords = [coords['RA'][0]+' '+coords['DEC'][0],]
                self._coordinates = SkyCoord(coords, unit=(u.hourangle,u.deg),frame='icrs')
            else:
                assert not isinstance(coords, basestring) #make sure it is a list
                if len(coords) not in [1,2,3]:
                    raise ValueError('Coordinates not understood. Set first to auto or provide\
them in a tuple with \
first: ra,dec, second coord_units, third: coord_frame. Two and three are optional.\
E.g. [["2:00:00 -1:00:00"],[u.hourangle, u.deg],"icrs"]')
                crds = coords[0]
                coord_units = self._coord_units
                coord_frome = self._coord_frame
                if len(coords) >= 2:
                    coord_units = coords[1]
                elif len(coords) ==3:
                    coord_frame = coords[2]
                    
                self._coordinates = SkyCoord(coords, unit=coord_units,frame = coord_frame)
                
        @property
        def mag(self):
            return self._mag
        
        @mag.setter
        def mag(self,value,filterband=[None,]):
            self._mag =       np.hstack((self._mag,np.array(value)))
            self._filterband= np.hstack((self._filterband,np.array(filterband)))

        @mag.deleter
        def mag(self,value):
            indices = np.where(self._mag == value)
            print('Removing entries nr {} from mag and filterband'.format(indices))
            self._mag        = np.delete(self._mag,indices,axis=0)
            self._filterband = np.delete(self._filterband,indices,axis=0)
        
        @property
        def SpT_num(self):
            return np.sum(split_spt(self.SpT)[0:2])

        @property
        def SpC_num(self):
            return split_spt(self.SpT)[2]

        @property
        def aliases(self):
            if self._aliases == None:
                self._aliases = list(Simbad.query_objectids(self.name)['ID'])
            return self._aliases
            

