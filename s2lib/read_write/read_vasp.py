from __future__ import print_function#, unicode_literals
from __future__ import absolute_import, division
try:
    xrange = xrange
    # We have Python 2
except:
    xrange = range
    # We have Python 3

import os
import re
import numpy as np
import glob

def make_atom_types(atom_nums):
    """
    Assign atom type ids to atom based on information from VASP atom counts

    Parameters
    ----------
    atom_nums: list of int
               counts of corresponding atom types

    Returns
    -------
    atom_types: 1d ndarray of ints
              atom type ids for each atom
    """

    atom_types = []

    ti = 0
    for atom_num in atom_nums:
        for i in range(atom_num):
            atom_types.append(ti)
        ti += 1

    return np.array(atom_types)



def read_poscar(filename):
    """
    Reads VASP POSCAR and CONTCAR files

    Parameters
    ----------
    filename: str
              full path and name of the POSCAR

    Returns
    -------
    traj: dict
          trajectory information with keys given below
    box: 3x3 ndarray
         Box dimensions
    xyz: natom x 3 ndarray
         Atomic configuration
    atom_name: list of str
                atom types (names)
    atom_num: list of ints
                atom numbers for each type
    """

    with open(filename, 'r') as fc:
        
        # atom names
        atom_names = [name for name in re.findall('\S+', fc.readline())]

        scale = float(re.findall('\S+', fc.readline())[0])
            
        # box parameters
        box = np.empty((3, 3), dtype=float)
        for i in range(3):
            box[i,:] = [float(x)*scale for x in re.findall('\S+', fc.readline())]

        # atom names (again, should be same as above)
        atom_names = [name for name in re.findall('\S+', fc.readline())]

        # number of atoms
        atnums = [int(num) for num in re.findall('\S+', fc.readline())]
        nat = sum(atnums)

        line = fc.readline()
            
        # atomic configuration
        xyz = np.empty((nat, 3), dtype=float)
        for i in range(nat):
            xyz[i] = [float(x) for x in re.findall('\S+', fc.readline())]

        # create a list of atom types from atom_name and atom_num
        atom_types = make_atom_types(atnums)

            
    traj = {'box0':box, 'xyz0':xyz, 'atom_type0':atom_types, 'atom_name':atom_names, 'atom_num0':atnums}

    return traj



def read_xdatcar(filename):
    """
    Reads VASP XDATCAR

    Parameters
    ----------
    filename: str
              full path and name of the XDATCAR

    Returns
    -------
    traj: dict
          trajectory information with keys given below
    box: list of 3x3 ndarrays
          Box dimensions
    xyz: list of natom x 3 ndarrays
          Atomic configurations
    atom_name: list of str
                atom types (names)
    atom_num: list of ints
                atom numbers for each type
    """

    with open(filename, 'r') as fc:

        xyzs = [] ; boxs = [] ; atom_types = [] ; atom_nums = []

        for line in iter(fc.readline, ''):
            if re.search('Direct configuration', line):
                box = np.copy(boxs[-1])
                atom_types.append(np.copy(atom_types[-1]))
                atom_nums.append(np.copy(atom_nums[-1]))

            else:
                scale = float(re.findall('\S+', fc.readline())[0])
            
                # box parameters
                box = np.empty((3, 3), dtype=float)
                for i in range(3):
                    box[i,:] = [float(x)*scale for x in re.findall('\S+', fc.readline())]

                # atom names
                atom_names = [name for name in re.findall('\S+', fc.readline())]

                # number of atoms
                atom_nums.append([int(num) for num in re.findall('\S+', fc.readline())])
                nat = sum(atom_nums[-1])

                # create a list of atom types from atom_name and atom_num
                atom_types.append(make_atom_types(atom_nums[-1]))

                #assert len(atom_types[-1]) == nat, f'Length of atom_types {len(atom_types[-1])} does not match number of atoms {nat}'
                assert len(atom_types[-1]) == nat, 'Length of atom_types {} does not match number of atoms {}'.format(len(atom_types[-1]), nat)

                line = fc.readline()

            

            # atomic configuration
            xyz = np.empty((nat, 3), dtype=float)
            for i in range(nat):
                xyz[i] = [float(x) for x in re.findall('\S+', fc.readline())]
            
            boxs.append(box)
            xyzs.append(xyz)


    traj = {'box':boxs, 'xyz':xyzs, 'atom_type':atom_types, 'atom_name':atom_names, 'atom_num':atom_nums}

    return traj


def read_outcar(filename):
    """
    Reads VASP OUTCAR

    Parameters
    ----------
    filename: str
              full path and name of the OUTCAR

    Returns
    -------
    traj: dict
          trajectory information with keys given below
    box: list of 3x3 ndarrays
          Box dimensions
    xyz: list of natom x 3 ndarrays
          Atomic configurations
    energy: list of floats
            list of configurational energies
    free_energy: list of floats
            list of free energies
    total_energy: list of floats
            list of total energies (including kinetic and thermostat)
    atom_num: list of ints
                atom numbers for each type
    """

    with open(filename, 'r') as f:
    
        # initialize trajectory dataset
        boxs = [] ; xyzs = [] ; enes = [] ; forces = [] ; temps = []
        vects = [] ; enes_free = [] ; enes_tot = [] ; atom_types = [] ; atom_nums = []
        ensemble = 'relax'
    
        for line in iter(f.readline, ''):
        
            # number of ions/atoms
            if re.search('number of ions', line):
                nat = int(re.findall('\S+', line)[-1])
        
            # number of atoms of each type
            elif re.search('ions per type', line):
                atnums = [int(n) for n in re.findall('\S+', line)[4:4+nat]]

            elif re.search('molecular dynamics for ions', line):
                ensemble = 'md'

            elif re.search('relaxation of ions', line):
                ensemble = 'relax'
        
            # box shape and dimensions
            elif re.search('VOLUME and BASIS-vectors are now', line):
                for _ in range(4):
                    line = f.readline()
                
                # read box information
                box = np.empty((3,3), dtype=np.float64)
                for i in range(3):
                    box[i,:] = [float(x) for x in re.findall('\S+', f.readline())][0:3]
                boxs.append(box)
            
                for _ in range(2):
                    line = f.readline()
                
                # read a, b, c vector lengths
                vect = np.array([float(x) for x in re.findall('\S+', f.readline())][0:3])
                vects.append(vect)

            # atom cartesian coordinates [A] and forces [eV/A]
            elif re.search('POSITION.*TOTAL-FORCE', line):                
                line = f.readline()
            
                # read coordinate and force data for all nat atoms
                data = np.array([[float(x) for x in f.readline().split()] for _ in range(nat)])
            
                # create new coordinate array
                xyz = np.empty((nat, 3), dtype=np.float64)
                xyz[:,:] = data[:,0:3]
            
                #assert len(xyzs) + 1 == len(boxs), f'lengths of xyzs {len(xyzs)+1} and boxs {len(boxs)} do not match'
                assert len(xyzs) + 1 == len(boxs), 'lengths of xyzs {} and boxs {} do not match'.format(len(xyzs)+1, len(boxs))
            
                # convert cartesian coordinates into lattice units
                box_inv = np.linalg.inv(boxs[-1].T)
                xyz = np.matmul(box_inv, xyz.T).T

                # create a new force array
                force = np.empty((nat, 3), dtype=np.float64)
                force[:,:] = data[:,3:6]
            
                xyzs.append(xyz)
                forces.append(force)

                # create a list of atom types from atom_nums
                atom_nums.append(atnums)
                atom_types.append(make_atom_types(atom_nums[-1]))
                #assert len(atom_types[-1]) == sum(atom_nums[-1]), f'Length of atom_types {len(atom_types[-1])} does not match number of atoms {sum(atom_nums)}'
                assert len(atom_types[-1]) == sum(atom_nums[-1]), 'Length of atom_types {} does not match number of atoms {}'.format(len(atom_types[-1]), sum(atom_nums))
            
            # E0 energy without entropy for sigma->0
            elif re.search('FREE ENERG.*\s+OF\s+THE\s+ION.ELECTRON\s+SYSTEM\s+\(eV\)', line):
                
                # check if the format agrees with the current assumptions
                if not re.search('------------', f.readline()):
                    raise ValueError('Could not find a separator line (----).')
                
                # read free energy (without kinetics)
                line = f.readline()
                if re.search('free\s+energy\s+TOTEN\s+=.+eV', line):
                    ene_free = float(re.findall('\S+', line)[-2])
                    enes_free.append(ene_free)
                else:
                    raise ValueError('Could not find a line with free energy (TOTEN).')
                    
                line = f.readline()
                
                # read energy without entropy for sigma->0
                line = f.readline()
                if re.search('energy\s+without\s+entropy.+sigma', line):
                    ene = float(re.findall('\S+', line)[-1])
                    enes.append(ene)
                else:
                    raise ValueError('Could not find a line with free energy (TOTEN).')
                
           
            # Total energy including thermostat 
            elif re.search('ETOTAL', line):
                ene_tot = float(line.split()[-2])
                enes_tot.append(ene_tot)
            
            # Instantaneous temperature 
            elif re.search('EKIN_LAT', line):
                mo = re.search('\(temperature\s+(\d+\.?\d*)\s+K\)', line)
                temp = float(mo.group(1))
                temps.append(temp)


    # check if the lengths of trajectory lists match
    #assert len(enes) == len(xyzs), f'{dataset} energy and XYZ lenghts do not match: {len(enes)}, {len(xyzs)}'
    assert len(enes) == len(xyzs), '{} energy and XYZ lenghts do not match: {}, {}'.format(dataset, len(enes), len(xyzs))
    #assert len(atom_types) == len(xyzs), f'{dataset} atom_types and XYZ lenghts do not match: {len(atom_types)}, {len(xyzs)}'
    assert len(atom_types) == len(xyzs), '{} atom_types and XYZ lenghts do not match: {}, {}'.format(dataset, len(atom_types), len(xyzs))
    
    # combine trajectory data in a dictionary
    traj = {'box':boxs, 'xyz':xyzs, 'atom_type':atom_types, 'energy':enes, 'forces':forces, 'temp':temps}
    traj.update({'atom_num':atom_nums})
    traj.update({'ensemble':ensemble})
    if len(enes_tot) > 0:
        traj.update({'total_energy':enes_tot})
    if len(enes_free) > 0:
        traj.update({'free_energy':enes_free})
    
    return traj

def read_oszicar(filename):
    # read configurational energies

    with open(filename, 'r') as f:
        enes = [] ; temps = [] ; enes_tot = [] ; enes_free = []
        for line in iter(f.readline, ''):
            if re.search('T=', line) and re.search('F=', line):
                ensemble = 'md'
                sarr = re.findall('\S+', line)
                temps.append(float(sarr[2]))
                enes_tot.append(float(sarr[4]))
                enes_free.append(float(sarr[6]))
                enes.append(float(sarr[8]))

            elif (not re.search('T=', line)) and re.search('F=', line):
                ensemble = 'relax'
                sarr = re.findall('\S+', line)
                temps.append(0.0)
                enes_free.append(float(sarr[2]))
                enes_tot.append(float(sarr[4])) # set total_energy at 0K to same as energy
                enes.append(float(sarr[4]))

    # combine trajectory data in a dictionary
    traj = {'energy':enes,
            'temp':temps,
            'free_energy':enes_free,
            'total_energy':enes_tot,
            'ensemble':ensemble}

    return traj

def read_incar(filename):
    traj = {}
    return traj

def read_vasp(vasp_dir, verbose=False):
    """
    Reads configuration and energy files from a VASP MD simulation in a given directory
    and returns trajectory data in a dictionary.
    
    Parameters
    ----------
    vasp_dir : string
              directory with VASP MD simulation data, has to contain XDATCAR and md.out files
    verbose: bool, default: True
              If True, print runtime information.
             
    Returns
    -------
    traj : dictionary
           trajectory information (configuration, box, energy, forces)
    """

    # dict of vasp_files and functions to read them
    vasp_files = {
            'OUTCAR':read_outcar,
            'POSCAR':read_poscar,
            'CONTCAR':read_poscar,
            #'XDATCAR':read_xdatcar,
            #'OSZICAR':read_oszicar,
            'INCAR':read_incar
            }

    # data obtained from different files
    alldata = {}

    for file_name, read_func in vasp_files.items():


        # find all file names of type 'file_name'
        file_names = glob.glob(os.path.join(vasp_dir, file_name+'*'))

        if file_names:
            # make sure maximum one file of each type exists in the directory
            #assert len(file_names) <= 1, f'Too many files of type {file_name}.'
            assert len(file_names) <= 1, 'Too many files of type {}.'.format(file_name)

            if os.path.isfile(file_names[0]):
                if verbose:
                    #print(f"Reading {file_names[0]}")
                    print("Reading {}".format(file_names[0]))

                alldata[file_name] = read_func(file_names[0])
            else:
                #print(f'{file_name} not present')
                print('{} not present'.format(file_name))


    # Perform consistency checks between data from different VASP files

    # Check system composition and trajectory lengths
    if 'OUTCAR' in alldata and 'POSCAR' in alldata:
        assert alldata['OUTCAR']['atom_num'][0] == alldata['POSCAR']['atom_num0'], 'Atom numbers in OUTCAR AND POSCAR do not match'
        cfg_shape_out = alldata['OUTCAR']['box'][0].shape
        cfg_shape_pos = alldata['POSCAR']['box0'].shape
        assert cfg_shape_out == cfg_shape_pos, 'Configuration shape in OUTCAR AND POSCAR do not match'

    if 'OUTCAR' in alldata and 'CONTCAR' in alldata:
        assert alldata['OUTCAR']['atom_num'][0] == alldata['CONTCAR']['atom_num0'], 'Atom numbers in OUTCAR AND POSCAR do not match'
        cfg_shape_out = alldata['OUTCAR']['box'][0].shape
        cfg_shape_pos = alldata['CONTCAR']['box0'].shape
        assert cfg_shape_out == cfg_shape_pos, 'Configuration shape in OUTCAR AND CONTCAR do not match'

    if 'OUTCAR' in alldata and 'XDATCAR' in alldata:
        assert alldata['OUTCAR']['atom_num'][-1] == alldata['XDATCAR']['atom_num'][-1], 'Atom numbers in OUTCAR AND XDATCAR do not match'
        traj_len_out = len(alldata['OUTCAR']['xyz'])
        traj_len_xdat = len(alldata['XDATCAR']['xyz'])
        #assert traj_len_out == traj_len_xdat, f'Trajectory lengths in OUTCAR ({traj_len_out}) AND XDATCAR ({traj_len_xdat}) do not match'
        assert traj_len_out == traj_len_xdat, 'Trajectory lengths in OUTCAR ({}) AND XDATCAR ({}) do not match'.format(traj_len_out, traj_len_xdat)

    if 'POSCAR' in alldata and 'CONTCAR' in alldata:
        assert alldata['POSCAR']['atom_num0'] == alldata['CONTCAR']['atom_num0'], 'Atom numbers in CONTCAR AND POSCAR do not match'
        assert alldata['POSCAR']['atom_name'] == alldata['CONTCAR']['atom_name'], 'Atom types in CONTCAR AND POSCAR do not match'

    if 'POSCAR' in alldata and 'XDATCAR' in alldata:
        assert alldata['POSCAR']['atom_num0'] == alldata['XDATCAR']['atom_num'][0], 'Atom numbers in OUTCAR AND XDATCAR do not match'
        assert alldata['POSCAR']['atom_name'] == alldata['XDATCAR']['atom_name'], 'Atom numbers in OUTCAR AND XDATCAR do not match'

    if 'CONTCAR' in alldata and 'XDATCAR' in alldata:
        assert alldata['CONTCAR']['atom_num0'] == alldata['XDATCAR']['atom_num'][-1], 'Atom numbers in CONTCAR AND XDATCAR do not match'
        assert alldata['CONTCAR']['atom_name'] == alldata['XDATCAR']['atom_name'], 'Atom numbers in CONTCAR AND XDATCAR do not match'

    if 'OUTCAR' in alldata and 'OSZICAR' in alldata:
        traj_len_out = len(alldata['OUTCAR']['energy'])
        traj_len_oszi = len(alldata['OSZICAR']['energy'])
        #assert traj_len_out == traj_len_oszi, f'Trajectory lengths in OUTCAR ({traj_len_out}) AND OSZICAR ({traj_len_oszi}) do not match'
        assert traj_len_out == traj_len_oszi, 'Trajectory lengths (energy) in OUTCAR ({}) AND OSZICAR ({}) do not match'.format(traj_len_out, traj_len_oszi)
        #traj_len_out = len(alldata['OUTCAR']['total_energy'])
        #traj_len_oszi = len(alldata['OSZICAR']['total_energy'])
        #assert traj_len_out == traj_len_oszi, f'Trajectory lengths in OUTCAR ({traj_len_out}) AND OSZICAR ({traj_len_oszi}) do not match'
        #assert traj_len_out == traj_len_oszi, 'Trajectory (total_energy) lengths in OUTCAR ({}) AND XDATCAR ({}) do not match'.format(traj_len_out, traj_len_oszi)

    # Check numerical values of boxes, coordinates, energies, and temperatures

    # If passed, combine data into the most complete trajectory dict

    traj = {}
    for key in alldata:
        traj.update(alldata[key])


    return traj
