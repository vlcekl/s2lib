from __future__ import print_function #, unicode_literals
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
from collections import defaultdict, Counter

def read_hstfile_ising(filename):
    """
    Reads lg.hst file with outptu statistics

    Parameters
    ----------
    filename: str
              full path and name of the lg.hst file

    Returns
    -------
    traj: dict
          trajectory information with keys given below
    atom_name: list of str
                atom types (names)
    atom_num: list of ints
                atom numbers for each type
    """

    with open(filename, 'r') as fi:
        enes = [] 
        hu = []

        npars = int(re.findall('\S+', fi.readline())[-1])

        # cycle over reference histograms
        nmax = 0
        for line in iter(fi.readline, ''):
            nmax = nmax + 1

            #  energy line
            sarr = re.findall('\S+', fi.readline())
            assert abs(float(sarr[-2]) - float(sarr[-1])) < 0.01, "Energies and statistics do not match {} {}".format(float(sarr[2]), float(sarr[3]))
            enes.append(float(sarr[-1]))

            ustats = []
            for i in range(npars):
                sarr = re.findall('\S+', fi.readline())
                ustats.append(int(sarr[-1]))

            hu.append(ustats)

    traj = {'energy':enes, 'interaction_stats':hu}

    return traj

def read_mldfile_ising(filename):
    """Read configurational energies"""

    with open(filename, 'r') as f:

        pars = []

        nn_pars = int(re.findall('\S+', f.readline())[1])

        # assert nn_pars == ntypes*(ntypes+1)//2, "Wrong number of parameters"

        for _ in range(nn_pars):
            sarr = re.findall('\S+', f.readline())
            pars.append(float(sarr[-1]))

    params = {'ref_params':np.array(pars)}

    return params

def read_runfile_ising(filename):
    """Read configurational energies"""

    with open(filename, 'r') as f:
        enes = [] ; temps = [] ; mag = []
        for line in iter(f.readline, ''):
            sarr = re.findall('\S+', line)
            temps.append(float(sarr[1]))
            enes.append(float(sarr[2]))
            mag.append(float(sarr[3]))

    # combine trajectory data in a dictionary
    traj = {'energy':enes, 'temp':temps, 'mag':mag}

    return traj

def read_lattice_ising(latt_dir, verbose=False):
    """
    Reads configuration and energy files from a VASP MD simulation in a given directory
    and returns trajectory data in a dictionary.
    
    Parameters
    ----------
    latt_dir : string
              directory with VASP MD simulation data, has to contain XDATCAR and md.out files
    verbose: bool, default: True
              If True, print runtime information.
             
    Returns
    -------
    traj : dictionary
           trajectory information (configuration, box, energy, forces)
    """

    # dict of latt_files and functions to read them
    latt_files = {
            'lg.hst':read_hstfile_ising,
            'lg.run':read_runfile_ising,
            'lg.mld':read_mldfile_ising,
            'lg.xyz':read_xyzfile
            }

    # data obtained from different files
    alldata = {}

    for name, read_func in latt_files.items():

        file_name = os.path.join(latt_dir, name)

        if os.path.isfile(file_name):
            if verbose:
                print(f"Reading {file_name}")
                #print("Reading {}".format(file_name))

            alldata[name] = read_func(file_name)
        else:
            #print(f'{file_name} not present')
            print('{} not present'.format(file_name))

    # Check system composition and trajectory lengths
    if 'lg.hst' in alldata and 'lg.run' in alldata:
        hst_ene = np.array(alldata['lg.hst']['energy'])
        run_ene = np.array(alldata['lg.run']['energy'])
        assert hst_ene.shape == run_ene.shape, 'Trajectory lengths in lg.hst and lg.run do not match'
        #assert np.allclose(hst_ene, run_ene), 'Energies in lg.hst and lg.run do not match'

    if 'lg.xyz' in alldata and 'lg.run' in alldata:
        xyz_xyz = np.array(alldata['lg.xyz']['xyz_latt'])
        run_ene = np.array(alldata['lg.run']['energy'])
        assert xyz_xyz.shape[0] == run_ene.shape[0], 'Trajectory lengths in lg.xyz and lg.run do not match'

    traj = {}
    for key in alldata:
        traj.update(alldata[key])

    return traj

def read_lattice_pair(latt_dir, verbose=False):
    """
    Reads configuration and energy files from a VASP MD simulation in a given directory
    and returns trajectory data in a dictionary.
    
    Parameters
    ----------
    latt_dir : string
              directory with VASP MD simulation data, has to contain XDATCAR and md.out files
    verbose: bool, default: True
              If True, print runtime information.
             
    Returns
    -------
    traj : dictionary
           trajectory information (configuration, box, energy, forces)
    """

    # dict of latt_files and functions to read them
    latt_files = {
            'lg.hst':read_histfile,
            'lg.run':read_runfile_old,
            'lg.mld':read_modeldef
            }

    # data obtained from different files
    alldata = {}

    for name, read_func in latt_files.items():

        file_name = os.path.join(latt_dir, name)

        if os.path.isfile(file_name):
            if verbose:
                print(f"Reading {file_name}")
                #print("Reading {}".format(file_name))

            alldata[name] = read_func(file_name)
        else:
            #print(f'{file_name} not present')
            print('{} not present'.format(file_name))

    # Check system composition and trajectory lengths
    if 'lg.hst' in alldata and 'lg.run' in alldata:
        hst_ene = np.array(alldata['lg.hst']['energy'])
        run_ene = np.array(alldata['lg.run']['energy'])
        assert hst_ene.shape == run_ene.shape, 'Trajectory lengths in lg.hst and lg.run do not match'
        #assert np.allclose(hst_ene, run_ene), 'Energies in lg.hst and lg.run do not match'

    if 'lg.xyz' in alldata and 'lg.run' in alldata:
        xyz_xyz = np.array(alldata['lg.xyz']['xyz_latt'])
        run_ene = np.array(alldata['lg.run']['energy'])
        assert xyz_xyz.shape[0] == run_ene.shape[0], 'Trajectory lengths in lg.xyz and lg.run do not match'

    traj = {}
    for key in alldata:
        traj.update(alldata[key])

    return traj

def read_lattice_triple(latt_dir, verbose=False):
    """
    Reads configuration and energy files from a VASP MD simulation in a given directory
    and returns trajectory data in a dictionary.
    
    Parameters
    ----------
    latt_dir : string
              directory with VASP MD simulation data, has to contain XDATCAR and md.out files
    verbose: bool, default: True
              If True, print runtime information.
             
    Returns
    -------
    traj : dictionary
           trajectory information (configuration, box, energy, forces)
    """

    # dict of latt_files and functions to read them
    latt_files = {
            'lg.hst':read_hstfile_triple,
            'lg.run':read_runfile,
            #'lg.xyz':read_xyzfile,
            'lg.mld':read_mldfile
            }

    # data obtained from different files
    alldata = {}

    for name, read_func in latt_files.items():

        file_name = os.path.join(latt_dir, name)

        if os.path.isfile(file_name):
            if verbose:
                print(f"Reading {file_name}")
                #print("Reading {}".format(file_name))

            alldata[name] = read_func(file_name)
        else:
            #print(f'{file_name} not present')
            print('{} not present'.format(file_name))

    # Check system composition and trajectory lengths
    if 'lg.hst' in alldata and 'lg.run' in alldata:
        hst_ene = np.array(alldata['lg.hst']['energy'])
        run_ene = np.array(alldata['lg.run']['energy'])
        assert hst_ene.shape == run_ene.shape, 'Trajectory lengths in lg.hst and lg.run do not match'
        #assert np.allclose(hst_ene, run_ene), 'Energies in lg.hst and lg.run do not match'

    if 'lg.xyz' in alldata and 'lg.run' in alldata:
        xyz_xyz = np.array(alldata['lg.xyz']['xyz_latt'])
        run_ene = np.array(alldata['lg.run']['energy'])
        assert xyz_xyz.shape[0] == run_ene.shape[0], 'Trajectory lengths in lg.xyz and lg.run do not match'

    traj = {}
    for key in alldata:
        traj.update(alldata[key])

    return traj

def read_hstfile_triple(filename):
    """
    Reads lg.hst file with outptu statistics

    Parameters
    ----------
    filename: str
              full path and name of the lg.hst file

    Returns
    -------
    traj: dict
          trajectory information with keys given below
    atom_name: list of str
                atom types (names)
    atom_num: list of ints
                atom numbers for each type
    """

    with open(filename, 'r') as fi:
        enes = [] ; atom_nums = [] ; atom_names = []
        hrs = [] ; hrx = []

        line = fi.readline()
        # cycle over reference histograms
        nmax = 0
        for line in iter(fi.readline, "ENDHST\n"):
            nmax = nmax + 1

            sarr = re.findall('\S+', line)

            #assert abs(float(sarr[2]) - float(sarr[3])) < 0.01, f"Energies and statistics do not match {float(sarr[2])} {float(sarr[3])}"
            assert abs(float(sarr[2]) - float(sarr[3])) < 0.01, "Energies and statistics do not match {} {}".format(float(sarr[2]), float(sarr[3]))
            enes.append(float(sarr[2]))

            line = fi.readline()
            sarr = re.findall('\S+', fi.readline())
            atom_nums = [int(it) for it in sarr[1:]]
            atom_names = list(range(len(sarr[1:])))

            # surface configuration histograms
            line = fi.readline()
            lsmax = 128
            hrs.append([float(re.findall('\S+', fi.readline())[1]) for _ in range(lsmax)])

            # interaction pairs histogram
            line = fi.readline()
            lumax = 4
            hrx.append([list(map(float, re.findall('\S+', fi.readline())[1:2])) for _ in range(lumax)])

    traj = {'energy':enes, 'atom_name':atom_names, 'atom_num':atom_nums}
    traj.update({'config_stats':hrs, 'interaction_stats':hrx})

    return traj


def read_hstfile(filename):
    """
    Reads lg.hst file with outptu statistics

    Parameters
    ----------
    filename: str
              full path and name of the lg.hst file

    Returns
    -------
    traj: dict
          trajectory information with keys given below
    atom_name: list of str
                atom types (names)
    atom_num: list of ints
                atom numbers for each type
    """

    with open(filename, 'r') as fi:
        enes = [] 
        hu = []

        ntype = int(re.findall('\S+', fi.readline())[-1])

        # cycle over reference histograms
        nmax = 0
        for line in iter(fi.readline, ''):
            nmax = nmax + 1

            #  energy line
            sarr = re.findall('\S+', fi.readline())
            assert abs(float(sarr[-2]) - float(sarr[-1])) < 0.01, "Energies and statistics do not match {} {}".format(float(sarr[2]), float(sarr[3]))
            enes.append(float(sarr[-1]))

            ustats = []
            for i in range(1,ntype+1):
                for j in range(i, ntype+1):
                    sarr = re.findall('\S+', fi.readline())
                    ustats.append(int(sarr[-1]))

            hu.append(ustats)

    traj = {'energy':enes, 'interaction_stats':hu}

    return traj

def read_hstfile(filename):
    """
    Reads lg.hst file with outptu statistics

    Parameters
    ----------
    filename: str
              full path and name of the lg.hst file

    Returns
    -------
    traj: dict
          trajectory information with keys given below
    atom_name: list of str
                atom types (names)
    atom_num: list of ints
                atom numbers for each type
    """

    with open(filename, 'r') as fi:
        enes = [] 
        hu = []

        ntype = int(re.findall('\S+', fi.readline())[-1])

        # cycle over reference histograms
        nmax = 0
        for line in iter(fi.readline, ''):
            nmax = nmax + 1

            #  energy line
            sarr = re.findall('\S+', fi.readline())
            assert abs(float(sarr[-2]) - float(sarr[-1])) < 0.01, "Energies and statistics do not match {} {}".format(float(sarr[2]), float(sarr[3]))
            enes.append(float(sarr[-1]))

            ustats = []
            for i in range(1,ntype+1):
                for j in range(i, ntype+1):
                    sarr = re.findall('\S+', fi.readline())
                    ustats.append(int(sarr[-1]))

            hu.append(ustats)

    traj = {'energy':enes, 'interaction_stats':hu}

    return traj


def read_histfile(filename):
    """
    Reads lg.hst file with outptu statistics

    Parameters
    ----------
    filename: str
              full path and name of the lg.hst file

    Returns
    -------
    traj: dict
          trajectory information with keys given below
    atom_name: list of str
                atom types (names)
    atom_num: list of ints
                atom numbers for each type
    """

    with open(filename, 'r') as fi:
        enes = [] ; atom_nums = [] ; atom_names = []
        hrs = [] ; hrx = []

        line = fi.readline()
        # cycle over reference histograms
        nmax = 0
        for line in iter(fi.readline, "ENDHST\n"):
            nmax = nmax + 1

            sarr = re.findall('\S+', line)

            #assert abs(float(sarr[2]) - float(sarr[3])) < 0.01, f"Energies and statistics do not match {float(sarr[2])} {float(sarr[3])}"
            assert abs(float(sarr[2]) - float(sarr[3])) < 0.01, "Energies and statistics do not match {} {}".format(float(sarr[2]), float(sarr[3]))
            enes.append(float(sarr[2]))

            line = fi.readline()
            sarr = re.findall('\S+', fi.readline())
            atom_nums = [int(it) for it in sarr[1:]]
            atom_names = list(range(len(sarr[1:])))

            # surface configuration histograms
            line = fi.readline()
            lsmax = 128
            hrs.append([float(re.findall('\S+', fi.readline())[1]) for _ in range(lsmax)])

            # interaction pairs histogram
            line = fi.readline()
            lumax = 3
            hrx.append([list(map(float, re.findall('\S+', fi.readline())[2:4])) for _ in range(lumax)])

    traj = {'energy':enes, 'atom_name':atom_names, 'atom_num':atom_nums}
    traj.update({'config_stats':hrs, 'interaction_stats':hrx})

    return traj

def read_runfile_old(filename):
    """Read configurational energies"""

    with open(filename, 'r') as f:
        enes = [] ; temps = []
        for line in iter(f.readline, ''):
            sarr = re.findall('\S+', line)
            enes.append(float(sarr[2]))
            temps.append(float(sarr[1]))

    # combine trajectory data in a dictionary
    traj = {'energy':enes, 'temp':temps}

    return traj

def read_runfile(filename):
    """Read configurational energies"""

    with open(filename, 'r') as f:
        enes = [] ; temps = []
        for line in iter(f.readline, ''):
            sarr = re.findall('\S+', line)
            enes.append(float(sarr[1]))
            temps.append(float(sarr[2]))

    # combine trajectory data in a dictionary
    traj = {'energy':enes, 'temp':temps}

    return traj

def read_xyzfile(filename):
    """
    Reads lattice xyz file

    Parameters
    ----------
    filename: str
              full path and name of the xyz file

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

        xyzs = [] ; boxs = [] ; atom_types = []

        for line in iter(fc.readline, ''):
            nat = int(re.findall('\S+', line)[0])

            box = np.array(list(map(int, re.findall('\S+', fc.readline())[0:3])))
            box = np.diag(box)

            # atomic configuration
            xyz = np.empty((nat, 3), dtype=int)
            ti = []
            for i in range(nat):
                sarr = re.findall('\S+', fc.readline())
                xyz[i] = [int(x) for x in sarr[1:4]]
                ti.append(int(sarr[0]))

            atom_types.append(np.array(ti))
            boxs.append(box)
            xyzs.append(xyz)

    traj = {'box_latt':boxs, 'xyz_latt':xyzs, 'atom_type':atom_types}

    return traj


def read_mldfile(filename):
    """Read configurational energies"""

    with open(filename, 'r') as f:

        pars = []

        ntypes = int(re.findall('\S+', f.readline())[1])
        nn_pars = int(re.findall('\S+', f.readline())[1])

        # assert nn_pars == ntypes*(ntypes+1)//2, "Wrong number of parameters"

        for _ in range(nn_pars):
            sarr = re.findall('\S+', f.readline())
            pars.append(float(sarr[-1]))

    params = {'ref_params':np.array(pars)}

    return params


def read_modeldef(filename):
    """Read configurational energies"""

    with open(filename, 'r') as f:

        pars = {}

        ntypes = int(re.findall('\S+', f.readline())[1])
        nn_pars = int(re.findall('\S+', f.readline())[1])
        for _ in range(nn_pars):
            sarr = re.findall('\S+', f.readline())
            ti, tj = (int(sarr[0]), int(sarr[1]))
            pars[(ti, tj)] = [float(sarr[2])]

        nnn_pars = int(re.findall('\S+', f.readline())[1])
        for _ in range(nnn_pars):
            sarr = re.findall('\S+', f.readline())
            ti, tj = (int(sarr[0]), int(sarr[1]))
            pars[(ti, tj)].append(float(sarr[2]))

    params = {'ref_params':pars}

    return params

def read_lattice_mc(latt_dir, verbose=False):
    """
    Reads configuration and energy files from a VASP MD simulation in a given directory
    and returns trajectory data in a dictionary.
    
    Parameters
    ----------
    latt_dir : string
              directory with VASP MD simulation data, has to contain XDATCAR and md.out files
    verbose: bool, default: True
              If True, print runtime information.
             
    Returns
    -------
    traj : dictionary
           trajectory information (configuration, box, energy, forces)
    """

    # dict of latt_files and functions to read them
    latt_files = {
            'lg.hst':read_hstfile,
            'lg.run':read_runfile,
            'lg.xyz':read_xyzfile,
            'lg.mld':read_mldfile
            }

    # data obtained from different files
    alldata = {}

    for name, read_func in latt_files.items():

        file_name = os.path.join(latt_dir, name)

        if os.path.isfile(file_name):
            if verbose:
                print(f"Reading {file_name}")
                #print("Reading {}".format(file_name))

            alldata[name] = read_func(file_name)
        else:
            #print(f'{file_name} not present')
            print('{} not present'.format(file_name))

    # Check system composition and trajectory lengths
    if 'lg.hst' in alldata and 'lg.run' in alldata:
        hst_ene = np.array(alldata['lg.hst']['energy'])
        run_ene = np.array(alldata['lg.run']['energy'])
        assert hst_ene.shape == run_ene.shape, 'Trajectory lengths in lg.hst and lg.run do not match'
        #assert np.allclose(hst_ene, run_ene), 'Energies in lg.hst and lg.run do not match'

    if 'lg.xyz' in alldata and 'lg.run' in alldata:
        xyz_xyz = np.array(alldata['lg.xyz']['xyz_latt'])
        run_ene = np.array(alldata['lg.run']['energy'])
        assert xyz_xyz.shape[0] == run_ene.shape[0], 'Trajectory lengths in lg.xyz and lg.run do not match'

    traj = {}
    for key in alldata:
        traj.update(alldata[key])

    return traj

def write_modeldef(filename, pars):
    """Read configurational energies"""

    with open(filename, 'r') as f, open(filename+'_temp', 'w') as fo:

        line = f.readline()
        fo.write(line)

        i = 0

        line = f.readline()
        fo.write(line)
        nn_pars = int(re.findall('\S+', line)[1])

        for _ in range(nn_pars):
            sarr = re.findall('\S+', f.readline())
            #fo.write(f'{sarr[0]} {sarr[1]} {pars[i]}\n')
            fo.write('{} {} {}\n'.format(sarr[0], sarr[1], pars[i]))
            i += 1

        line = f.readline()
        fo.write(line)
        nnn_pars = int(re.findall('\S+', line)[1])

        for _ in range(nnn_pars):
            sarr = re.findall('\S+', f.readline())
            #fo.write(f'{sarr[0]} {sarr[1]} {pars[i]}\n')
            fo.write('{} {} {}\n'.format(sarr[0], sarr[1], pars[i]))
            i += 1

        for line in iter(f.readline, ''):
            fo.write(line)

    #assert i == len(pars), f"The number of old ({i}) and new ({len(pars)}) do not match."
    assert i == len(pars), "The number of old ({}) and new ({}) do not match.".format(i, len(pars))

    os.rename(filename, filename+'_old')
    os.rename(filename+'_temp', filename)

