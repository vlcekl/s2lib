from __future__ import print_function #, unicode_literals
from __future__ import absolute_import, division
try:
    xrange = xrange
    # We have Python 2
except:
    xrange = range
    # We have Python 3

import numpy as np
from itertools import combinations_with_replacement
from collections import defaultdict
from .pair_dist import pair_dist, pair_dist_cutoff

def get_stats_Mie(xyz, box, atom_type=None, ms=[12,6], rcut=None):
    """
    Takes atom pair distances and calculates sufficeint statistics needed
    for the parameterization 1/r^m type potentials (Lennard-Jones, Mie).
 
    Parameters
    ----------
    rr : 2D numpy array of floats
         set of pair distances
    rx : numpy array
         set of pair distance coordinates
    ms : python list of integers 
         exponents of 1/r^m
    atom_type: 1D numpy array of integers (particle types)


    Returns
    -------
    ar, a1, a2 : numpy arrays (len(ms))
                 atom energy-related statistics
                 el_density**0.5, el_density, el_density**2
    br, b1, b2 : numpy arrays (len(ms), natoms, 3 coordinates)
                 atom force-related statistics (gradients of energy)
                 grad(el_density**0.5), grad(el_density), grad(el_density**2)
    """

    # set rcut to max if None
    if rcut == None:
        rcut = 0.5*min(box[0,0], box[1,1], box[2,2])

    # get pair distances (absolute and Cartesian components)
    rr, rx = pair_dist_cutoff(xyz, box, rcut)

    # if needed, set the atom number types
    if type(atom_type) == None:
        atom_type1 = np.ones((rr.shape[0]), dtype=int)
        atom_type2 = np.ones((rr.shape[1]), dtype=int)
    else:
        atom_type1 = atom_type
        # how many times the original box was replicated?
        nreplicas = rr.shape[1]//rr.shape[0] 
        atom_type2 = np.concatenate([atom_type]*nreplicas)


    # Create 2D matrix of inverse distances (eliminate rr=0 and distances beyond cutoff)
    with np.errstate(divide='ignore'):
        rinv = np.where(np.logical_and(rr > 0.01, rr <= rcut), 1.0/rr, 0.0)

    # particle type mesh grids
    mti, mtj = np.meshgrid(atom_type1, atom_type2, indexing='ij')
    #assert mti.shape == rr.shape, f"stats_mie: Shapes of mti {mti.shape} and rr {rr.shape} are incompatible"
    assert mti.shape == rr.shape, "stats_mie: Shapes of mti {} and rr {} are incompatible".format(mti.shape, rr.shape)

    # ENERGY cacluations

    # Create lists of 2D matrices with 1/r^m values for energy
    rpow_u = [rinv**m for m in ms]
    print('rcut:', rcut, mti.shape, mtj.shape, rpow_u[0].shape)

    # meshgrid to eliminate duplicate pair interactins
    ii, jj = np.meshgrid(range(rr.shape[0]), range(rr.shape[1]), indexing='ij')
    #assert ii.shape == rr.shape, f"stats_mie: Shapes of ii {ii.shape} and rr {rr.shape} are incompatible"
    assert ii.shape == rr.shape, "stats_mie: Shapes of ii {} and rr {} are incompatible".format(ii.shape, rr.shape)

    # cycle over all combinations of pairs of particle types for energy contributions
    u_stats = defaultdict(list)
    for ti, tj in combinations_with_replacement(set(atom_type1), 2):
        for m, rmat_u in zip(ms, rpow_u):
            # erase duplicate interactions (i,j)==(j,i)
            rmat_u = np.where(jj > ii, rmat_u, 0.0)

            # halve in-box out-box interaction contributions
            rmat_u = np.where(jj < rr.shape[0], rmat_u, 0.5*rmat_u)

            # sum contributions from the given atom types (ti, tj)
            u_stats[(ti, tj)].append(np.sum(np.where(np.logical_and(mti==ti, mtj==tj), rmat_u, 0.0)))


    # FORCE cacluations

    # particle type mesh grids for forces - stack 
    fti = np.stack((mti, mti, mti), axis=-1)
    ftj = np.stack((mtj, mtj, mtj), axis=-1)
    #assert fti.shape == rx.shape, f"The shapes of fti {fti.shape} and rx {rx.shape} are different"
    assert fti.shape == rx.shape, "The shapes of fti {} and rx {} are different".format(fti.shape, rx.shape)

    # Create lists of 2D matrices with 1/r^(m+2) values for forces
    rpow_x = [rinv**(m+2) for m in ms]

    # Multiply the force values by the components of atom-atom distance vectors
    rpow_f = [np.empty_like(rx) for m in ms]
    for m in range(len(ms)):
        for i in range(3):
            rpow_f[m][:,:,i] = rpow_x[m]*rx[:,:,i]

    # cycle over all combinations of pairs of particle types for force contributions
    f_stats = defaultdict(list)
    for ti, tj in combinations_with_replacement(set(atom_type1), 2):
        for m, rmat_f in zip(ms, rpow_f):
            f_stats[(ti, tj)].append(np.sum(np.where(np.logical_and(fti==ti, ftj==tj), rmat_f, 0.0), axis=1))

    return dict(u_stats), dict(f_stats)

