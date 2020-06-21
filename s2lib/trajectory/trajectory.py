from __future__ import print_function #, unicode_literals
from __future__ import absolute_import, division
try:
    xrange = xrange
    # We have Python 2
except:
    xrange = range
    # We have Python 3

import sys
import re
import numpy as np
import copy

class Trajectory(object):
    """
    Class that stores a trajectory of particle coordinates with methods for
    manipulation of the trajectory data (e.g., append, merge, sample, select, ...)

    The goal is for the class to behave similar to DataFrame of pandas.
    
    Note: will need to be refined - divide different trajectory data to
    attributes applicable to the trajectory as a whole and data. This is
    similar to hdf5 format.
    """

    def __init__(self, traj, inplace=True):
        """
        Creates a trajectory object from a dictionary or another trajectory object
        """

        if traj is None:
            self._trajectory = {}

        elif isinstance(traj, dict):
            if inplace:
                self._trajectory = traj
            else:
                self._trajectory = copy.deepcopy(traj)

        elif isinstance(traj, Trajectory):
            if inplace:
                self._trajectory = traj._trajectory
            else:
                self._trajectory = copy.deepcopy(traj._trajectory)


    def __getitem__(self, key):
        """
        Implement column (property) access and row slicing of trajectories

        Parameters
        ----------
        key : str, int, or slice
            if str, select appropriate property (e.g., box), if slice select
            configurations in a trajectory

        Returns
        -------
        data: ndarray or trajectory object
              Requested data
        """

        if isinstance(key, int):
            if key < 0 : #Handle negative indices
                key += len( self['xyz'] )
            if key < 0 or key >= len( self['xyz'] ):
                raise IndexError('The index {} is out of range.'.format(key))

            # Create a new instance
            trj_handle = Trajectory(self, inplace=False)
            trj_handle._trajectory = dict.fromkeys(self._trajectory)

            #Cycle over trajectory properties and select a particular configuration
            for k in self._trajectory:
                if type(self[k]) == list and 'atom_name' not in k and '0' not in k:
                    trj_handle[k] = self[k][key:key+1]
                else:
                    trj_handle[k] = self[k]

            return trj_handle

        elif isinstance(key, slice):
            # Create a new instance
            trj_handle = Trajectory(self, inplace=False)
            trj_handle._trajectory = dict.fromkeys(self._trajectory)

            #Cycle over trajectory properties and select slices of lists
            for k in self._trajectory:
                if type(self[k]) == list and 'atom_name' not in k and '0' not in k:
                    trj_handle[k] = self[k][key.start:key.stop:key.step]
                else:
                    trj_handle[k] = self[k]

            return trj_handle

        elif isinstance(key, str) or isinstance(key, unicode):
            return self._trajectory[key]

        else:
            raise TypeError('Invalid argument type: {}: {}.'.format(key, type(key)))


    def __setitem__(self, key, value):
        self._trajectory[key] = value

    def merge(self, traj_object):
        pass

    def append(self, new_traj):
        """
        Appends a trajectory object. Tries to append only features that change
        during the simulation (currently NPT - keep particles types and numbers constant).

        Appending a large number of trajectories can be done by concat function.
        """

        all_keys = set(list(self._trajectory) + list(new_traj._trajectory))

        # define special treatment of different trajectory items
        # Only connect trajectories with the same ensembles

        assert self['ensemble'] == new_traj['ensemble'], "Trying to append trajectories of incompatible ensembles"

        # Check that nvt and npt ensembles keep the same particle names, types, and numbers
        if self['ensemble'] == 'nvt' or self['ensemble'] == 'npt':
            assert self['atom_name'] == new_traj['atom_name'], "Append: atom_name does not match"
            #assert self['atom_num'] == new_traj['atom_num'], "Append: atom_num does not match"
            #assert np.array_equal(self['atom_type'][0], new_traj['atom_type'][0]), "Append: atom_type arrays do not match"

        # Check that nvt and uvt ensembles have the same box parameters
        if self['ensemble'] == 'nvt' or self['ensemble'] == 'uvt':
            assert self['box'] == new_traj['box'], "Append: box parameters do not match"

        # define which items will be appended
        append_keys = ['xyz', 'box', 'atom_type', 'temp', 'forces']
        append_keys.extend(['energy'])#, 'free_energy', 'total_energy'])

        for key in append_keys:
            self[key] += new_traj[key]


    def replicate(self, vec_a=1, vec_b=1, vec_c=1, inplace=False):
        """
        Replicates atomic configurations and corresponding box sizes and
        energies in given directions.

        Parameters
        ----------
        vec_{abc}: int
                   number of replications in the direction of the lattice
                   vectors a, b, and c. All default to 1.
        inplace: bool
                 If False (default), returns a new trajectory object,
                 otherwise replicated configurations stored in the current
                 object.
        
        Returns
        -------
        trj_handle: Trajectory object
                    If inplace=False, returns a new trajectory object with replicated configurations
        """

        # set trajectory object handle
        if inplace:
            trj_handle = self
        else:
            trj_handle = Trajectory(self, inplace=False)
            trj_handle._trajectory = copy.deepcopy(self._trajectory)


        # system size scaling
        multiply = vec_a*vec_b*vec_c

        # multiply atom numbers and types
        trj_handle['atom_name'] = self['atom_name']*multiply

        # cycle over trajectory configurations
        for i in range(len(self['box'])):
            # number of original atoms
            nat = sum(self['atom_num'][i])
            trj_handle['atom_num'][i] = self['atom_num'][i]*multiply

            # create a new array accommodating the replicated configuration
            new_xyz = np.empty((nat*multiply, 3), dtype=float)
            new_type = np.empty((nat*multiply), dtype=int)

            # replicate coordinates, add new atoms at the end of the xyz array
            j = 0
            for ia in range(vec_a):
                for ib in range(vec_b):
                    for ic in range(vec_c):
                        new_xyz[j*nat:(j+1)*nat,0] = self['xyz'][i][0:nat,0] + float(ia)
                        new_xyz[j*nat:(j+1)*nat,1] = self['xyz'][i][0:nat,1] + float(ib)
                        new_xyz[j*nat:(j+1)*nat,2] = self['xyz'][i][0:nat,2] + float(ic)
                        new_type[j*nat:(j+1)*nat] = self['atom_type'][i][0:nat]
                        j += 1
                                
            trj_handle['xyz'][i] = new_xyz
            trj_handle['atom_type'][i] = new_type

            # scale original coordinates
            trj_handle['xyz'][i][:,0] /= float(vec_a)
            trj_handle['xyz'][i][:,1] /= float(vec_b)
            trj_handle['xyz'][i][:,2] /= float(vec_c)

            trj_handle['box'][i][0,:] = self['box'][i][0,:]*float(vec_a)
            trj_handle['box'][i][1,:] = self['box'][i][1,:]*float(vec_b)
            trj_handle['box'][i][2,:] = self['box'][i][2,:]*float(vec_c)


            # multiply all energy data
            for key in [k for k in self._trajectory.keys() if 'energy' in k]:
                trj_handle[key][i] = self[key][i]*float(multiply)

        if not inplace:
            return trj_handle

    def set_zero_energy(self, zero_energy, inplace=True):
        """
        Sets energy of infinitely separated atoms to zero by subtracting
        intra-atomic energy of all system atoms.

        Parameters
        ----------
        zero_energy: float
                     energy of an isolated atom
        inplace: bool
                 scale energy of the current trajectory (True) or make a new one (False)
        """

        # set trajectory object handle
        if inplace:
            trj_handle = self
        else:
            trj_handle = Trajectory(self, inplace=False)
            trj_handle._trajectory = copy.deepcopy(self._trajectory)

        for key, trj in trj_handle._trajectory.items():
            if 'energy' in key:
                for i in range(len(trj_handle[key])):
                    trj_handle[key][i] -= zero_energy*trj_handle['xyz'][i].shape[0]

        if not inplace:
            return trj_handle



    def to_xyz(self, file_name):
        """
        Save trajectory to .xyz file format
        """

        with open(file_name, 'w') as f:
            # cycle through configurations in trajectory, assign atom names, and write to file

            for box, xyz in zip(self['box'], self['xyz']):

                # write total number of atoms
                nat = sum(self['atom_num'])
                #f.write(f'{nat}\n')
                f.write('{}\n'.format(nat))

                # write box parameters
                ax, ay, az = box[0,0], box[0,1], box[0,2]
                bx, by, bz = box[1,0], box[1,1], box[1,2]
                cx, cy, cz = box[2,0], box[2,1], box[2,2]
                #f.write(f'{ax} {ay} {az} {bx} {by} {bz} {cx} {cy} {cz}\n')
                f.write('{} {} {} {} {} {} {} {} {}\n'.format(ax, ay, az, bx, by, bz, cx, cy, cz))

                # write atom coordinates
                i = 0
                for atom_name, atom_num in zip(self['atom_name'], self['atom_num']):
                    for _ in range(atom_num):
                        x, y, z = (box.T).dot(xyz[i])
                        #f.write(f'{atom_name} {x} {y} {z}\n')
                        f.write('{} {} {} {}\n'.format(atom_name, x, y, z))
                        i += 1

