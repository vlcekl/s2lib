from __future__ import print_function #, unicode_literals
from __future__ import absolute_import, division
try:
    xrange = xrange
    # We have Python 2
except:
    xrange = range
    # We have Python 3
#!/usr/local/Python-2.7.6/bin/python

import sys
import re
import numpy as np
from scipy import optimize

#def sd2_loss(params, stats, targets, utot_func, dl=0.05, verbose=0):

def sd_hist(p, q, grs, grp, hrs, hrp, hru):
    """Statistical distance between histograms (surface, profile)"""

    # apply bounds on parametes
    p = np.where(p < -1.0, -1.0, p)
    p = np.where(p >  1.0,  1.0, p)

    # nearest and next nearest interactions between unlike particles
    pp = np.array([0.0, p[0], 0.0, 0.0, p[1], 0.0])
    qq = np.array([0.0, q[0], 0.0, 0.0, q[1], 0.0])

    # energy diference: bulk(1,2), surface(1,2), surface(1,1)
    uuu = beta*np.sum(hru*(pp - qq), axis=1)
    uave = np.sum(uuu)*fn
    uuu -= uave
    eee = np.exp(-uuu)
    fx = 1/np.sum(eee)

    # statistical distance for surface configuration histogram
    dloss  = np.arccos(np.sum(np.sqrt(np.sum(hrs*eee, axis=1)*fx*grs[:])))**2

    fx = -np.log(fn/fx)
    eee = np.exp(0.5*(fx - uuu))
    db = -2.0*np.log(np.sum(eee)*fn)
    ge = (fx + uave)/beta

    return dloss

if __name__ == "__main__":

    # read target data (histograms)
    with open(sys.argv[1], 'r') as fi:
        # surface histogram
        fi.readline()
        lsmax = int(re.findall('\S+', fi.readline())[1])
        grs = np.array(list(map(float, re.findall('\S+', fi.readline())[1:3]) for _ in range(lsmax)))
        grs[:,0] = grs[:,0]/np.sum(grs[:,0]) # normalize histogram 1
    print('grs.shape', grs.shape, np.sum(grs, axis=0))

    # read reference data (energies, histograms)
    with open(sys.argv[2], 'r') as fi:
        hrs = []
        hrx = []

        line = fi.readline()
        # cycle over reference histograms
        nmax = 0
        for line in iter(fi.readline, "ENDHST\n"):
            nmax = nmax + 1

            # surface histogram
            line = fi.readline()
            hrs.append(list(map(float, re.findall('\S+', fi.readline())[1:3]) for _ in range(lsmax)))

            # pair interaction histogram - bulk[0] and surface[1], 4,5, and 7 are useful
            line = fi.readline()
            lumax = 10
            hrx.append(list(map(float, re.findall('\S+', fi.readline())[2:6]) for _ in range(lumax)))

        hrs = np.array(hrs).transpose()
        hrs[0] = hrs[0]*float(nmax)/np.sum(hrs[0])
        hrx = np.array(hrx)
        print('hrs shape', hrs.shape)
        print('hrx shape', hrx.shape)

    # read reference and starting parameters
    with open(sys.argv[3], 'r') as fi:
        # reference parameters
        pref = np.array(map(float, re.findall('\S+', fi.readline())))
        # initial parametes for search
        pars = []
        for line in iter(fi.readline, ''):
            pars.append(np.array(map(float, re.findall('\S+', line))))

    # chose appropriate histograms for selected parameters
    hru = np.zeros((hrx.shape[0],8), dtype=float)
    hru[:,0] = hrx[:,0,0] # NN 1-1
    hru[:,1] = hrx[:,1,0] # NN 1-2
    hru[:,2] = hrx[:,2,0] # NN 2-2
    hru[:,3] = hrx[:,0,1] # NNN 1-1
    hru[:,4] = hrx[:,1,1] # NNN 1-2
    hru[:,5] = hrx[:,2,1] # NNN 2-2

    temp = 0.35
    temp = float(sys.argv[4])
    beta = 1.0/temp
    uuu = np.zeros((nmax), dtype=float)
    eee = np.zeros((nmax), dtype=float)
    fn = 1.0/float(nmax)

    for par_in in pars: # cycle over differet starting parameter sets
        print('# start ', sd_hist(par_in, pref, grs, grp, hrs, hrp, hru), par_in, dg(par_in, pref, hru))
        output = optimize.fmin(sd_hist, par_in, args=(pref, grs, grp, hrs, hrp, hru), maxiter=100000, maxfun=10000, disp=0, full_output=1)
        print('# end   ', output[1], output[0][:], dg(output[0][:4], pref, hru))
        print('')
