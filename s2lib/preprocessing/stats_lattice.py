from __future__ import print_function #, unicode_literals
from __future__ import absolute_import, division

try:
    xrange = xrange
    # We have Python 2
except:
    xrange = range
    # We have Python 3

from collections import Counter
import numpy as np

def get_stats_latt(hrs, hrx):

    hrs = np.array(hrs).T
    hrs = hrs*float(hrs.shape[1])/np.sum(hrs)
    hrx = np.array(hrx)

    print('hrs shape', hrs.shape)
    print('hrx shape', hrx.shape)

    return hrs, hrx


def latt_to_real_coords(trj, scale=1.0):

    """
    Read xyz file with lattice coordinates
    and return rescaled atomic configurations in real units and (orthogonal) box dimensions.
    """
    
    boxs = []
    xyzs = []
    for box, xyz in zip(trj['box_latt'], trj['xyz_latt']):
        boxs.append(box*scale)
        xyzs.append(xyz*scale)
        # scale to fractional coordinates
        xyzs[-1] /= np.diag(boxs[-1])
    trj['box'] = boxs
    trj['xyz'] = xyzs
    return trj


def add_experimental_noise(trj, loss_rate=0.0, disp=[0.0, 0.0, 0.0], unknown_frac=0.0):
    """
    Apply random noise emulating APT experiment.
    Includes detector efficiency (only 33% retained) and random displacement.
    
    Parameters
    ----------
    xyz_in: list of ndarrays
        list of atom coordinates
    box: ndarray (3x3)
        box dimensions
    loss_rate: float or list of floats
        loss rate, overall (float) or per atom type (list)
    disp: array-like, shape(3)
        random displacement of atoms in x, y, z directions
        standard deviation of a Gaussian distribution
        
    Returns
    -------
    config_out: ndarray
        list of atoms and their coordinates in xyz format (type, x)
    """    
    
    xyzs = trj['xyz']
    boxs = trj['box']
    typs = trj['atom_type']

    xyzs_mod = []
    typs_mod = []

    for xyz, box, typ in zip(xyzs, boxs, typs):
        
        boxv = np.diag(box)

        xyz_mod = []
        typ_mod = []
        
        for r, t in zip(xyz, typ):
        
            # Step 1: remove a given fraction of particles
            if np.random.random() < loss_rate:
                continue
            
            # Step 2: displace atoms using Gaussian noise
            rr = boxv*r
            rr += np.array([np.random.normal(loc=0.0, scale=disp[i]) for i in range(3)])

            # adjust positions using periodic boundary conditions
            rr = np.where(rr < 0.0, rr+boxv, rr)
            rr = np.where(rr > boxv, rr-boxv, rr)
            
            rr /= boxv
                
            xyz_mod.append(rr)
            
            # unknown type (type==6)
            if np.random.random() < unknown_frac:
                typ_mod.append(6)
            else:
                typ_mod.append(t)

                
        xyzs_mod.append(np.array(xyz_mod))
        typs_mod.append(np.array(typ_mod))


    return xyzs_mod, typs_mod


def real_coords(xyzs, boxs):
    xyzs_mod = []
    for xyz, box in zip(xyzs, boxs):
        boxv = np.diag(box)
        xyzs_mod.append(xyz*boxv)

    return xyzs_mod


def select_core(xyzs, boxs, tis):
    """
    Selects the central (core) region of a configuration with all coordinates
    closer to the center than surface.
    """
    
    xyzs_core = []    
    tis_core = []
    for xyz, box, ti in zip(xyzs, boxs, tis):

        bv = np.diag(box)
    
        r_core = []
        t_core = []
        for r, t in zip(xyz, ti):
            if r[0]<0.25*bv[0] or r[1]<0.25*bv[1] or r[2]<0.25*bv[2]:
                continue
            if r[0]>=0.75*bv[0] or r[1]>=0.75*bv[1] or r[2]>=0.75*bv[2]:
                continue

            r_core.append(r)
            t_core.append(t)
            
        xyzs_core.append(np.array(r_core))
        tis_core.append(np.array(t_core))

    return xyzs_core, tis_core

def get_knn_stats(indices, ti_core, ti, knns=[]):
    """
    Returns
    -------
        Atom-atom statistics for a given configuration
    """
    
    ntypes = len(Counter(ti))

    k_stats = []
    for k in knns:
        hst = np.zeros((ntypes, ntypes, k+1), dtype=int)
        
        for i, ind in enumerate(indices):
            nbr_dict = Counter(ti[ind[1:k+1]])
            for t in range(1, ntypes+1):
                c = nbr_dict.get(t, 0)
                hst[ti_core[i]-1, t-1, c] += 1
            
        k_stats.append(hst)
            
    return k_stats
