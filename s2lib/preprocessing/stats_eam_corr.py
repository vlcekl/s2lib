from __future__ import print_function #, unicode_literals
from __future__ import absolute_import, division
try:
    xrange = xrange
    # We have Python 2
except:
    xrange = range
    # We have Python 3

import copy
import numpy as np
from .pair_dist import pair_dist, pair_dist_cutoff

def get_stats_EAM_per_atom(config, atom_type=None, sc=[2., 3., 4.], rcut=None):#, rmax=None):
    """
    Takes atom pair distances and calculates per atom-statistics needed
    for the parameterization of a cubic spline-based EAM model similar to
    Marinica (2013), having a full spline representation for electronic
    density function.
 
    Parameters
    ----------
    config: tuple(2)
            xyz and box information about a particular configuration
    atom_type: list of int
            atom type ids
    sc : python list of floats
         If not rcut and rmax parameters are given, sc expected to be ordered on a regular grid
         In this format last three knots will be used as helper knots needed for
         cubic b-spline definition
    rcut: float
          EAM potential cutoff distance

    Returns
    -------
    ar, a1, a2 : numpy arrays (len(sc))
                 atom energy-related statistics
                 el_density**0.5, el_density, el_density**2
    br, b1, b2 : numpy arrays (len(sc), natoms, 3 coordinates)
                 atom force-related statistics (gradients of energy)
                 grad(el_density**0.5), grad(el_density), grad(el_density**2)
    """
 
#    # set rcut to the potential cutoff
#    if rcut == None:
#        rcut = sc[-4]
#
#    # set rmax to the most distant knot for b-spline definition
#    if rmax == None:
#        rmax = sc[-1]

    if rcut == None:
        rcut = sc[-1]

    xyz = config[0]
    box = config[1]

    # get pair distances (absolute and Cartesian components)
    rr, rx = pair_dist_cutoff(xyz, box, rcut)

    # number of atoms in configuration
    n_atom = rr.shape[0]

    # number of atoms in the replicated box
    n_neighbor = rr.shape[1]
    
    # energy-related statistics
    ar = np.zeros((len(sc)), dtype=float)  # box statitics for square root part of embedding function
    a1 = np.zeros_like(ar)                 # box statistics for pair potential
    a2 = np.zeros_like(ar)                 # box statitics for square part of embedding function
    ax = np.zeros((len(sc), n_atom), dtype=float) # per atom stats for density function
    
    # force-related statistics
    br = np.zeros((len(sc), n_atom, 3), dtype=float)
    b1 = np.zeros_like(br)
    b2 = np.zeros_like(br)
    zero3 = np.zeros((3), dtype=float)

    # cycle over spline knots
    for ks, rc in enumerate(sc):

        # cycle over atoms
        for i in range(n_atom):

            # sum electronic density over all neighbors of i within rc
            aa = sum([(rc - r)**3 for r in rr[i] if (r < rcut and r < rc and r > 0.01)])
            ax[ks, i] = aa

            # if el. density larger than zero, calculate force statistics
            if aa > 0.0:

                # precompute a list of recurring values for force statistics
                ff = [1.5*(rc - r)**2*x/r if (r > 0.01 and r < rc and r < rcut) else zero3 for r, x in zip(rr[i], rx[i])]

                # sum contributions to force statistics from all neighbors of i
                b1[ks, i] = sum([2*f       for f in ff])
                br[ks, i] = sum([ -f/np.sqrt(aa) for f in ff])
                b2[ks, i] = sum([4*f*aa for f in ff])

        # sum contributions to per box energy statistics for a given spline
        # knot
        ar[ks] = np.sum(np.sqrt(ax[ks,:]))
        a1[ks] = np.sum(ax[ks,:])
        a2[ks] = np.sum(ax[ks,:]**2)

    # energy correction component (number of particles * density)
    corr = float(n_atom*n_atom)/np.linalg.det(box)
    c1 = np.array([corr])

    return a1, ar, a2, ax, b1, br, b2, c1

def get_stats_EAM_per_box(xyz, box, atom_type=None, sc=[2., 3., 4.], rcut=None):#, rmax=None):
    """
    Takes atom pair distances and calculates per box-statistics needed
    for the parameterization of a cubic spline-based EAM model by Bonny et al. (2017),
    having a simple cubic polynomial representation for electronic
    density function. This allows us to sum atom contributions into a whole
    box contributions to system energy.
 
    Parameters
    ----------
    rr : numpy array
         set of pair distances
    rx : numpy array
         set of pair distance coordinates
    sc : python list of floats
         spline knots

    Returns
    -------
    ar, a1, a2 : numpy arrays (len(sc))
                 atom energy-related statistics
                 el_density**0.5, el_density, el_density**2
    br, b1, b2 : numpy arrays (len(sc), natoms, 3 coordinates)
                 atom force-related statistics (gradients of energy)
                 grad(el_density**0.5), grad(el_density), grad(el_density**2)
    """
 
    # set rcut to max if None
    if rcut == None:
        rcut = sc[-1]

#    # set rcut to the potential cutoff
#    if rcut == None:
#        rcut = sc[-4]
#
#    # set rmax to the most distant knot for b-spline definition
#    if rmax == None:
#        rmax = sc[-1]

    # get pair distances (absolute and Cartesian components)
    rr, rx = pair_dist_cutoff(xyz, box, rcut)

    # number of atoms in configuration
    n_atom = rr.shape[0]

    # number of atoms in the replicated box
    n_neighbor = rr.shape[1]
    
    # energy-related statistics
    aa = np.empty((n_atom), dtype=float)
    ar = np.zeros((len(sc)), dtype=float)
    a1 = np.zeros_like(ar)
    a2 = np.zeros_like(ar)
    
    # force-related statistics
    br = np.zeros((len(sc), n_atom, 3), dtype=float)
    b1 = np.zeros_like(br)
    b2 = np.zeros_like(br)
    zero3 = np.zeros((3), dtype=float)

    # cycle over spline knots
    for ks, rc in enumerate(sc):

        # cycle over atoms
        for i in range(n_atom):

            # sum electronic density over all neighbors of i within rc
            aa[i] = sum([(rc - r)**3 for r in rr[i] if (r < rcut and r < rc and r > 0.01)])

            # if el. density larger than zero, calculate force statistics
            if aa[i] > 0.0:

                # precompute a list of recurring values for force statistics
                ff = [1.5*(rc - r)**2*x/r if (r < rcut and r > 0.01 and r < rc) else zero3 for r, x in zip(rr[i], rx[i])]

                # sum contributions to force statistics from all neighbors of i
                b1[ks, i] = sum([2*f       for f in ff])
                br[ks, i] = sum([ -f/np.sqrt(aa[i]) for f in ff])
                b2[ks, i] = sum([4*f*aa[i] for f in ff])

        # sum contributions to energy statistics for a given spline knot
        ar[ks] = np.sum(np.sqrt(aa))
        a1[ks] = np.sum(aa)
        a2[ks] = np.sum(aa**2)

    return a1, ar, a2, b1, br, b2
#    u_stats = np.array([a1, ar, a2])
#    f_stats = np.array([b1, br, b2])
#
#    return u_stats, f_stats

def get_stats_EAM_limited(rr, rx, sc):
    """
    Takes atom pair distances and calculates sufficeint statistics needed
    for the parameterization of a cubic spline-based EAM model by Bonny et al. (2017).
 
    Parameters
    ----------
    rr : numpy array
         set of pair distances
    rx : numpy array
         set of pair distance coordinates
    sc : python list of floats
         spline knots

    Returns
    -------
    ar, a1, a2 : numpy arrays (len(sc))
                 atom energy-related statistics
                 el_density**0.5, el_density, el_density**2
    br, b1, b2 : numpy arrays (len(sc), natoms, 3 coordinates)
                 atom force-related statistics (gradients of energy)
                 grad(el_density**0.5), grad(el_density), grad(el_density**2)
    """
 
    # number of atoms in configuration
    n_atom = rr.shape[0]
    
    # energy-related statistics
    aa = np.empty((n_atom), dtype=float)
    ar = np.zeros((len(sc)), dtype=float)
    a1 = np.zeros_like(ar)
    a2 = np.zeros_like(ar)
    
    # force-related statistics
    br = np.zeros((len(sc), n_atom, 3), dtype=float)
    b1 = np.zeros_like(br)
    b2 = np.zeros_like(br)
    zero3 = np.zeros((3), dtype=float)

    # cycle over spline knots
    for ks, rc in enumerate(sc):

        # cycle over atoms
        for i in range(n_atom):

            # sum electronic density over all neighbors of i within rc
            aa[i] = sum([(rc - r)**3 for r in rr[i] if (r < rc and r > 0.01)])

            # if el. density larger than zero, calculate force statistics
            if aa[i] > 0.0:

                # precompute a list of recurring values for force statistics
                ff = [1.5*(rc - r)**2*x/r if (r > 0.01 and r < rc) else zero3 for r, x in zip(rr[i], rx[i])]

                # sum contributions to force statistics from all neighbors of i
                b1[ks, i] = sum([2*f       for f in ff])
                br[ks, i] = sum([ -f/np.sqrt(aa[i]) for f in ff])
                b2[ks, i] = sum([4*f*aa[i] for f in ff])

        # sum contributions to energy statistics for a given spline knot
        ar[ks] = np.sum(np.sqrt(aa))
        a1[ks] = np.sum(aa)
        a2[ks] = np.sum(aa**2)

    u_stats = [a1, ar, a2]
    f_stats = [b1, br, b2]

    return u_stats, f_stats


def tpf_to_bsplines(stats_tpf):
    """
    Convert statistics data from the cubic truncated power function (TPF) basis to b-splines.
    Only works for the special case of evenly separated knots. 

    Parameters
    ----------
    stats_tpf: dict
               Trajectory statistics information using TPF basis
               Should be based on evenly spaced knots with the last three
               being the boundary knots

    Returns
    -------
    stats_bspline: dict
               Trajectory statistics information using b-spline basis

    """

    knots_tpf = copy.deepcopy(stats_tpf['hyperparams'])['pair']

    # check if the selected knots are evenly spaced
    print('len',len(knots_tpf))
    diff = [knots_tpf[i+1] - knots_tpf[i] for i in range(len(knots_tpf)-1)]
    assert sum([abs(d - diff[0]) for d in diff]) < 1e-6, 'Knots are not evenly spaced'

    # binomial coefficients for cubic splines
    binom = [1.0, -4.0, 6.0, -4.0, 1.0]

    stats_bspline = {}
    stats_bspline['hyperparams'] = copy.deepcopy(stats_tpf['hyperparams'])
    # index splines based on the shortest-distance knot
    #stats_bspline['hyperparams']['pair'] = stats_bspline['hyperparams']['pair'][:-4]
    #stats_bspline['hyperparams']['edens'] = stats_bspline['hyperparams']['edens'][:-4]
    stats_bspline['hyperparams']['pair'] = stats_bspline['hyperparams']['pair'][:]
    stats_bspline['hyperparams']['edens'] = stats_bspline['hyperparams']['edens'][:]

    stats_bspline['function'] = stats_tpf['function']

    for key, traj in stats_tpf.items():

        # skip non-trajectory items
        if 'hyperparams' in key or 'function' in key:
            continue

        # create a new trajectory
        traj_new = []

        # cycle over configurations of the trajectory
        for conf in traj['energy']:

            # create a new configuration
            conf_new = []
 
            # run through the different energy statistics for the configuration
            # 0:sum(rho^0.5), 1:sum(rho^2), 2:pair, 3:rho-per-atom)
            for ir, stat in enumerate(conf[0:4]):

                # create a new statistics
                stat_new = []

                # add contributions from tpf 
                #for i in range(len(knots_tpf)-4):
                for i in range(-4, len(knots_tpf)-4):
                    ii = i + 4

                    if ir < 2:
                        # this is for stats 0 and 1 (no b-splines)
                        bs = copy.deepcopy(stat[ii])
                    else:
                        # this is for stats 2 and 3: convert tpf to b-splines
                        if isinstance(stat[ii], float):
                            bs = 0.0
                        elif isinstance(stat, np.ndarray):
                            bs = np.zeros_like(stat[ii])
                        else:
                            raise TypeError('stat is neither float or ndarray'+str(type(stat))+stat)
                        
                        for j, bc in enumerate(binom):
                            assert i+j <= len(knots_tpf), "B-spline components exceed TPF knots"
                            if i+j < 0:
                                bs += 0.0
                            else:
                                bs += bc*stat[i+j]

                    stat_new.append(bs)
 
                # append new b-spline statistics
                conf_new.append(np.array(stat_new))
 
            traj_new.append(conf_new)

        stats_bspline[key] = {'energy':traj_new}

    return stats_bspline


