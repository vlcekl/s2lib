from __future__ import print_function #, unicode_literals
from __future__ import absolute_import, division
try:
    xrange = xrange
    # We have Python 2
except:
    xrange = range
    # We have Python 3

"""
Function to save a tabulated EAM potential in LAMMPS format.
"""

import numpy as np
from ..forcefields.eam import f_embed, f_dens, f_pair

def write_eam_lammps(params, filename):
    """
    Writes tabulated EAM functions in LAMMPS format

    Parameters
    ----------
    params: dict
            parameters needed to construct the potential:

    filename: str
              name of the output forcefield file
    """

    # forcefield parameters dict
    fp = params['forcefield']
    # system parameters dict
    sp = params['system']
    # tabulation parameters dict
    tp = params['tabulate']

    # prepare array of density values
    rhomax = tp['rhomax'] # maximum density
    nrho = tp['nrho']     # number of density grid points
    drho = rhomax/nrho    # density step size
    dens = np.linspace(0., rhomax-drho, nrho, endpoint=True)

    # prepare array of pair distances
    rcut = tp['rcut']     # cutoff distance
    nr = tp['nr']         # number of distance grid points
    dr = rcut/nr          # pair distance step size
    r = np.linspace(0., rcut-dr, nr, endpoint=True)

    # tabulated embedding function
    fembd = [f_embed(d, fp['embed']) for d in dens]

    # tabulated electronic density
    edens = [f_dens(x, fp['dens_a'], fp['dens_r']) for x in r]

    # tabulated pair potential
    fpair = [x*f_pair(x, fp['pair_a'], fp['pair_r']) for x in r]


    # general system parameters
    n_el = sp['n_el']        # number of elements
    el_name = sp['el_name']  # element names
    na = sp['na']            # atomic number
    ma = sp['ma']            # atomic mass
    lc = sp['lc']            # lattice constant
    alat = sp['alat']        # lattice type

    # write EAM force field file
    with open(filename, 'w') as fo:
    
        # Comment section
        fo.write('Comment 1\n')
        fo.write('Comment 2\n')
        fo.write('Comment 3\n')
        
        # Number of elements
        fo.write(f"{n_el:5d} ")
        for i in range(n_el):
            fo.write(f"{el_name[i]:2}")
        fo.write("\n")
        
        # number of grid points, step size, and cutoff
        fo.write(f"{nrho:5d}{drho:24.16e}{nr:5d}{dr:24.16e}{rcut:24.16e}\n")
        
        # atomic number, mass, lattice size, lattice type
        fo.write(f"{na:5d} {ma:14.4f} {lc:14.4f} {alat:10}\n")
        
        # Embeding function
        for i in range(nrho//4):
            fo.write("{0:20.12e}{1:20.12e}{2:20.12e}{3:20.12e}\n".format(*fembd[i*4:i*4+4]))
        
        # Electronic density
        for i in range(nr//4):
            fo.write("{0:20.12e}{1:20.12e}{2:20.12e}{3:20.12e}\n".format(*edens[i*4:i*4+4]))
            
        # Pair potential
        for i in range(nr//4):
            fo.write("{0:20.12e}{1:20.12e}{2:20.12e}{3:20.12e}\n".format(*fpair[i*4:i*4+4]))


# Cubic spline function for pair potentials and electronic density
f_spline = lambda r, aa, kk: sum([a*(rk - r)**3 for a, rk in zip(aa, kk) if r < rk and r > 0.01])

# Functional form for the embedding potential
f_embed = lambda d, a: a[0]*d**0.5 + a[1]*d + a[2]*d**2

# prepare lists of values
dens = np.linspace(0., rhomax-drho, nrho, endpoint=True)
r = np.linspace(0., rcut-dr, nr, endpoint=True)

#fembd = [f_embed(d, many_y) for d in dens] # Re
Fe = lambda d, a: f_embed(d/S, a) - C/S*d # rescaled potential
#fembd = [F(d, many_y) for d in dens] # W
fembd = [Fe(d, many_z) for d in dens] # W


#edens = [f_spline(x, rho_x_a, rho_x_r) for x in r] # Re
#edens = np.array([f_spline(x, rho_x_a, rho_x_r) for x in r]) # W
edens = np.array([f_spline(x, rho_x_a, rho_x_r)*S for x in r]) # W

# Pair potential

# 1. Cubic spline for r > r_o
#f_outer = [f_spline(x, pair_y, V_x_r) for i, x in enumerate(r)]
f_outer = [f_spline(x, pair_z, V_x_r) + 2*C*edens[i]/S for i, x in enumerate(r)]

# 2. Repulsive core for r < r_i (precalculate up to r_o)
f_inner = [0.0] + [u_core(x) for x in r[1:]]

# 3. Transition region for r_i < r < r_o
fpair = [x*u_trans(x, f_inner[i], f_outer[i]) for i, x in enumerate(r)]
#fpair = [x*f_outer[i] for i, x in enumerate(r)]
