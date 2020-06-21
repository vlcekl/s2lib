from __future__ import print_function #, unicode_literals
from __future__ import absolute_import, division
try:
    xrange = xrange
    # We have Python 2
except:
    xrange = range
    # We have Python 3

import numpy as np
from .pair_dist import pair_dist, pair_dist_cutoff

def get_stats_EAM_per_atom(config, atom_type=None, sc=[2., 3., 4.], rcut=None):
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
         spline knots
    rcut: float
          potential cutoff distance

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
        rcut = max(sc)

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

    # cycle over spline nodes
    for ks, rc in enumerate(sc):

        # cycle over atoms
        for i in range(n_atom):

            # sum electronic density over all neighbors of i within rc
            aa = sum([(rc - r)**3 for r in rr[i] if (r < rc and r > 0.01)])
            ax[ks, i] = aa

            # if el. density larger than zero, calculate force statistics
            if aa > 0.0:

                # precompute a list of recurring values for force statistics
                ff = [1.5*(rc - r)**2*x/r if (r > 0.01 and r < rc) else zero3 for r, x in zip(rr[i], rx[i])]

                # sum contributions to force statistics from all neighbors of i
                b1[ks, i] = sum([2*f       for f in ff])
                br[ks, i] = sum([ -f/np.sqrt(aa) for f in ff])
                b2[ks, i] = sum([4*f*aa for f in ff])

        # sum contributions to per box energy statistics for a given spline node
        ar[ks] = np.sum(np.sqrt(ax[ks,:]))
        a1[ks] = np.sum(ax[ks,:])
        a2[ks] = np.sum(ax[ks,:]**2)

    # energy correction component (number of particles * density)
    corr = float(n_atom*n_atom)/np.linalg.det(box)
    c1 = np.array([corr])

    return a1, ar, a2, ax, b1, br, b2, c1

def get_stats_EAM_per_box(xyz, box, atom_type=None, sc=[2., 3., 4.], rcut=None):
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
         spline nodes

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
        rcut = max(sc)

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

    # cycle over spline nodes
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

        # sum contributions to energy statistics for a given spline node
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
         spline nodes

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

    # cycle over spline nodes
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

        # sum contributions to energy statistics for a given spline node
        ar[ks] = np.sum(np.sqrt(aa))
        a1[ks] = np.sum(aa)
        a2[ks] = np.sum(aa**2)

    u_stats = [a1, ar, a2]
    f_stats = [b1, br, b2]

    return u_stats, f_stats


def tpf_to_bsplines(stats_tpf, knots_id):
    """
    Convert statistics data from the cubic truncated power function (TPF) basis to b-splines.
    Only works for the special case of evenly separated knots. 

    Parameters
    ----------
    stats_tpf: dict
               Trajectory statistics information using TPF basis
    bknots: list of ints
            Indices of knots from the stats_tpf
               
    Returns
    -------
    stats_bspline: dict
               Trajectory statistics information using b-spline basis

    """

    knots_tpf = stats['hyperparams']

    # get knot vector
    knots_b = [round(knots_tpf[i], 6) for i in knots_id]

    # check if the selected knots are evenly spaced
    diff = [knots_b[i+1] - knots_b[i] for i in range(len(knots_b)-1)]
    assert sum([abs(d - diff[0]) for d in diff]) < 1e-6, 'Knots are not evenly spaced'

    stats_bspline = {}
    stats_bspline['hyperparams'] = knots_b

    # binomial coefficients for cubic splines
    binom = [1.0, -4.0, 6.0, -4.0, 1.0]

    stats_old = stats_tpf['energy']
    stats_new = []

    # cycle over configurations of the trajectory
    for ic, s_old in enumerate(stats_old):
        s_new = []

        # run through all components
        for k, so in enumerate(s_old):
            # create a component ndarray of the right shape
            sn = np.zeros_like(so)

            # add contributions from tpf
            for i, knot in enumerate(knots_id):
                for j, bc in enumerate(binom):
                    sn += bc*s_old[k][i+j]

            # append new b-spline components
            s_new.append(sn)

        stats_new.append(s_new)

    stats_bspline['energy'] = stats

    return stats_bspline


