from __future__ import print_function #, unicode_literals
from __future__ import absolute_import, division
try:
    xrange = xrange
    # We have Python 2
except:
    xrange = range
    # We have Python 3

import os
import numpy as np
import pickle
import copy
from .pair_dist import pair_dist
from .stats_eam import get_stats_EAM_per_atom, get_stats_EAM_per_box
from .stats_mie import get_stats_Mie
from .utils import universal_eos

ff_func = {'EAM-cubic-spline':get_stats_EAM_per_atom,
           'Mie':get_stats_Mie}

def force_targ(forces, fatoms):
    """
    Creates an array of forces on atoms in all 6N directions (+ and -) along
    with zero forces for the original configuration.
    
    Parameters
    ----------
    forces: ndarray
            array of 3N forces on each atom

    Returns
    -------
    force_flat: ndarray
                flattened array of forces in all 6N+1 perturbation directions.
    """

    force_flat = []
    for i, frc in enumerate(forces):
        fr = np.concatenate((np.array([0.0]), frc[fatoms].flatten(), -frc[fatoms].flatten()))
        force_flat.append(fr)

    #return np.array(force_flat)
    return force_flat


def traj_stats(trj_files, get_stats_func, params=None, weights=None):
    """
    Calculates statistics relevant for a particular forcefield and target data from a given sets of trajectories.

    Parmeters
    ---------
    trejectories: list of str
                  names of trajectory files
    get_stats_func: function to calculate the statistics
    params: floats
            parameters needed by the stats_func to calculate the statistics
    weights: list of floats
             weights of the different trajectories

    Returns
    -------
    stats_data: list of dicts
                statistics data
    target_data: list of dicts
                 target data
    """

    # Assign unit weights if not specified otherwise
    if weights is None:
        weights = [1.0 for _ in range(len(trj_files))]
    
    # List of statistics and target data to be used in optimization
    stats_data = []
    target_data = []

    # cycle over trajectory data and save target and statistics information
    for di, (filin, weight) in enumerate(zip(trj_files, weights)):

        ## Target data ##

        # Create a target dataset directory with exhaustive target information
        target_dict = {'type':'trajectory', 'weight':weight}

        # read pickled trajectory dictionary
        with open(os.path.join(target_proc, filin+'.pickle'), 'rb') as fi:
            traj_dict = pickle.load(fi)

        # save trajectory data
        target_dict['box'] = traj_dict['box']
        target_dict['xyz'] = traj_dict['xyz']
        target_dict['energy'] = traj_dict['energy']
        target_dict['temp'] = traj_dict['temp']

        # read and transform forces into (6N+1) arrays
        if 'forces' in traj_dict.keys():
            target_dict['forces'] = force_targ(traj_dict['forces'])

        # save inverse temperature data (if T=0, set beta=1/300)
        target_dict['beta'] = np.empty_like(target_dict['temp'])
        for i, temp in enumerate(target_dict['temp']):
            if temp == 0.0:
                target_dict['beta'][i] = 1.0/300.0
            else:
                target_dict['beta'][i] = 1.0/temp


        ## Statistics data ##

        # Collect energy and force statistics from reference configurations
        stats_dict = {'energy':[], 'forces':[]}
        for xyz, box in zip(target_dict['xyz'], target_dict['box']):

            # check if box size large enough for nearest neighbor periodic boundary conditions
            if 0.5*box < sc[-1]:
                raise ValueError('box size ({box}) is too small for the force field cutoff {sc[-1]}')

            # calculate pair distance matrices (absolute values-rr, components-rx)
            rr, rx = pair_dist(xyz, box)

            # calculate sufficient statistics for energies and forces from pair distances
            a1, ar, a2, f1, fr, f2 = get_stats_func(rr, rx, sc)

            #print('mindist', np.where(rr > 0.0, rr, 10000.0).min())
            #print(xyz.shape, box)
            #print('x', a1.shape, rr.shape, np.sum(np.abs(a1)))

            stats_dict['energy'].append(np.array([ar, a2, a1]))
            stats_dict['forces'].append(np.array([fr, f2, f1]))

        # add datasets
        stats_data.append(stats_dict)
        target_data.append(target_dict)

    return stats_data, target_data


def get_stats(trj_datasets, ff_form):
    """
    Calculates statisitics for a given trajectory and hyperparameters.
    
    Parameters
    ----------
    trj_datasets: dict
             set of trajectories (list of box parameters and particle configurations)
    ff_form: dict
             functional form of the force field: potential function and
             hyperparmeters
            
    Returns
    -------
    stats_data: dict
            relevant trajectory statistics and the corresponding hyperparamters
    """

    stats_data = {}

    params = ff_form['hyperparams']
    stats_func = ff_func[ff_form['potential']]

    for key, trj in trj_datasets.items():
    
        stats_dict = {'energy':[]}
        
        for ii, (xyz, box) in enumerate(zip(trj['xyz'], trj['box'])):
        
            #a1, ar, a2, f1, fr, f2 = 
            u_stats, f_stats = stats_func(xyz, box, params)

            stats_dict['energy'].append(u_stats)

        stats_data[key] = stats_dict
    
    stats_data['ff_form'] = ff_form
    
    return stats_data


def scale_configuration(trj, scale, system='W'):
    """
    Scales box size by 10 to separate all atoms beyond cutoff, and sets energies and forces to 0.
    
    Parameters
    ----------
    trj: Trajectory object (or dict)
         Trajectory to be rescaled
    
    Returns
    -------
    trj: Trajectory object (or dict)
         Rescaled trajectory
    """
    trj['box'][0] = trj['box'][0]*scale
    trj['box0'] = trj['box0']*scale
    trj['energy'][0] = universal_eos(scale, system)*len(trj['xyz'][0])
    trj['free_energy'][0] = universal_eos(scale, system)*len(trj['xyz'][0])
    trj['total_energy'][0] = universal_eos(scale, system)*len(trj['xyz'][0])
    trj['forces'][0] = np.zeros_like(trj['forces'][0])

    return trj


if __name__ == '__main__':

    # trajectory files + directory information
    trj_files = ['structs_0k']#, 'liq_4000k', 'bcc_300k']#, 'german_dft.h5']
    target_proc = '../../../data/target_processed'
    trj_files = [os.path.join(target_proc, trj) for trj in trj_files]

    # Static parameters of the EAM potential: set of spline nodes
    sc = [2.56, 2.73, 3.252, 3.804, 4.20, 4.77]

    # extract statistics and target data from an ab initio trajectory
    stats_data, target_data = traj_stats(trj_files, get_stats_EAM, params=sc)
    
    # processed data directory
    output_dir = '../../../data/working'

    # pickle stats and target data to be used for optimization
    with open(os.path.join(output_dir, 'stats.pickle'), 'wb') as fo:
        pickle.dump(stats_data, fo)

    with open(os.path.join(output_dir, 'target.pickle'), 'wb') as fo:
        pickle.dump(target_data, fo)

