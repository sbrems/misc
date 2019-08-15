import os
import pickle
import numpy as np
import time
from starclass import Star
from collections import defaultdict
from tqdm import tqdm


_pnmanualmainids = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "knownmainids.p")


def get_mainids(stars, verbose=True, sleeptime=0.5, manual_edit=True):
    '''Return al list with the Simbad_mainid for each of the stars.
    Sleep is needed as Simbad blocks otherwise. In seconds.'''
    mainlist = []
    if verbose:
        print('Querying aliases for {} stars'.format(len(stars)))
        iterator = tqdm(stars)
    else:
        iterator = stars
    for star in iterator:
        dumstar = Star(star)
        try:
            mainlist.append(dumstar.simbad_main_ID)
        except TypeError:
            if manual_edit:
                if os.path.exists(_pnmanualmainids):
                    man_converter = pickle.load(open(_pnmanualmainids, "rb"))
                    if star in man_converter:
                        print('Manually converted {} to {} (select \
manual_edit=False to avoid this)'.format(
                            star, man_converter[star]))
                        mainlist.append(man_converter[star])
                        continue
                else:
                    man_converter = {}
                man_entered = input('Could not get the alias for {}. Please enter \
the alias manually (without ") (select manual_edit=False to avoid this):\n'.format(star))
                man_converter[star] = man_entered
                print('Added {} as main id for {}'.format(man_entered, star))
                pickle.dump(man_converter, open(_pnmanualmainids, "wb"))
            else:
                if verbose:
                    print('Alias for star {} could not be retrieved. \
Using empty string instead.'.format(star))
            mainlist.append("")
        time.sleep(sleeptime)
    return mainlist


def get_aliasindices(stars, verbose=True, sleeptime=0.5):
    '''Return a dictionary with aliases indices. If the dict is empty,
    no aliases were found. Emptystring is where the aliasing failed,
    e.g. because Simbad couldnt resolve the star name.'''
    mainstars = get_mainids(stars, verbose=verbose, sleeptime=sleeptime)
    dduplicates = list_duplicates(mainstars)
    if verbose:
        print('Found aliases:')
        for key, item in dduplicates.items():
            print('{} has aliases {}'.format(
                key, np.array(stars)[item]))
    return dduplicates


def list_duplicates(seq):
    '''list the duplicates of an already converted list'''
    tally = defaultdict(list)
    for i, item in enumerate(seq):
        tally[item].append(i)
    dict = {}
    for key, locs in tally.items():
        if len(locs) > 1:
            dict[key] = locs
    return dict
