from __future__ import print_function #, unicode_literals
from __future__ import absolute_import, division
try:
    xrange = xrange
    # We have Python 2
except:
    xrange = range
    # We have Python 3

import numpy as np

def cfg_replicate(xyz, box, vec_a=1, vec_b=1, vec_c=1):

    # system size scaling
    multiply = vec_a*vec_b*vec_c

    nat = len(xyz)

    # create a new array accommodating the replicated configuration
    new_xyz = np.empty((nat*multiply, 3), dtype=float)
    new_box = np.empty_like(box)

    # replicate coordinates, add new atoms at the end of the xyz array
    j = 0
    for ia in range(vec_a):
        for ib in range(vec_b):
            for ic in range(vec_c):
                new_xyz[j*nat:(j+1)*nat,0] = xyz[0:nat,0] + float(ia)
                new_xyz[j*nat:(j+1)*nat,1] = xyz[0:nat,1] + float(ib)
                new_xyz[j*nat:(j+1)*nat,2] = xyz[0:nat,2] + float(ic)
                j += 1

    new_xyz[:,0] /= float(vec_a)
    new_xyz[:,1] /= float(vec_b)
    new_xyz[:,2] /= float(vec_c)

    new_box[0,:] = box[0,:]*float(vec_a)
    new_box[1,:] = box[1,:]*float(vec_b)
    new_box[2,:] = box[2,:]*float(vec_c)

    return new_xyz, new_box


def pair_dist_cutoff(xyz, box, rcut):
    """
    Calculates nearest image pair distances between all atoms in xyz array.
    Up to the given cutoff. If box is too small, it is replicated

    Parameters
    -----------
    xyz : numpy array
          particle x, y, z coordinates
    box : numpy 2D array of unit cell vectors or a float
          simulation box dimensions/shape
    rcut: float
          cutoff

    Returns
    -------
    rr  : (natom, natom) numpy array of pair distances
    rx  : (natom, natom, 3) numpy array of pair distance coordinates
    """
    
    # number of unique atoms in a configuration
    n_atom = xyz.shape[0] 

    # make sure that the box is a 3x3 matrix (if box is float, multiply it by a unit matrix)
    box = np.eye(3).dot(box)

    # evaluate need for box replication
    vec_a = int(rcut//(0.5*box[0,0])) + 1
    vec_b = int(rcut//(0.5*box[1,1])) + 1
    vec_c = int(rcut//(0.5*box[2,2])) + 1

    # replicate if needed
    if max(vec_a, vec_b, vec_c) > 1:
        print('Replicating:', vec_a, vec_b, vec_c)
        xyz, box = cfg_replicate(xyz, box, vec_a, vec_b, vec_c)

    # updated number of particles in the replicated configuration
    n_atom2 = xyz.shape[0]

    rr = np.empty((n_atom, n_atom2), dtype=float)
    rx = np.empty((n_atom, n_atom2, 3), dtype=float)
    
    boxT = box.T
    
    for i, pa in enumerate(xyz[0:n_atom]):
        for j, pb in enumerate(xyz[0:]):
            dp = pb - pa
            dp = np.where(dp < -0.5, dp + 1.0, dp)
            dp = np.where(dp >  0.5, dp - 1.0, dp)
            
            dp = boxT.dot(dp)
            
            rr[i,j] = np.sum(dp*dp)**0.5
            rx[i,j] = dp

    return rr, rx


def pair_dist(xyz, box):
    """
    Calculates nearest image pair distances between all atoms in xyz array.

    Parameters
    -----------
    xyz : numpy array
          particle x, y, z coordinates
    box : numpy 2D array of unit cell vectors or a float
          simulation box dimensions/shape

    Returns
    -------
    rr  : (natom, natom) numpy array of pair distances
    rx  : (natom, natom, 3) numpy array of pair distance coordinates
    """
    
    # make sure that the box is a 3x3 matrix (if box is float, multiply it by a unit matrix)
    box = np.eye(3).dot(box)

    n_atom = xyz.shape[0] # number of atoms in a configuration
    rr = np.empty((n_atom, n_atom), dtype=float)
    rx = np.empty((n_atom, n_atom, 3), dtype=float)
    
    boxT = box.T
    
    for i, pa in enumerate(xyz):
        for j, pb in enumerate(xyz):
            dp = pa - pb
            dp = np.where(dp < -0.5, dp + 1.0, dp)
            dp = np.where(dp >  0.5, dp - 1.0, dp)
            
            dp = boxT.dot(dp)
            
            rr[i,j] = np.sum(dp*dp)**0.5
            rx[i,j] = dp

    return rr, rx
