from __future__ import print_function #, unicode_literals
from __future__ import absolute_import, division
try:
    xrange = xrange
    # We have Python 2
except:
    xrange = range
    # We have Python 3



"""
Collection of EAM functions
"""

import numpy as np
from .potentials import f_spline3

# Functional form for the embedding potential and its scaled form
f_embed = lambda d, a: a[0]*d**0.5 + a[1]*d**2
f_embed_lin = lambda d, a: a[0]*d**0.5 + a[1]*d + a[2]*d**2
f_embed_s = lambda d, a: f_embed_lin(d/S, a) - C*d/S 

# Density function and its scaled form
f_dens = lambda x, dens_a, dens_r: f_spline3(x, dens_a, dens_r)
f_dens_s = lambda x, dens_a, dens_r: f_spline3(x, dens_a, dens_r)*S

def u_equi(r, pair_a, pair_r, dens_a, dens_r):
    """
    Equilibrium (normal, adjustable) part of the pair potential
    based on cubic splines.
    """

    # cubic spline pair potential
    u = f_spline3(x, pair_a, pair_r)

    # gauge transformation into regular form
    u += 2*C*f_spline3(x, dens_a, dens_r)

    return u

# Define the core parts of the potential (kept constant)
def u_core(r, za=74, zb=74):
    """
    Repulsive potential of the atomic cores. Default atomic numbers for W

    Parameters
    ----------
    r: float
       atom pair distance
    za, zb: floats
       atomic numbers of the two atoms

    Returns
    ------
    u: float
       pair energy at pair distance r
    """

    qe_sq = 14.3992 # squared electron charge  
    rs = 0.4683766/(za**(2/3) + zb**(2/3))**0.5
    x = r/rs

    u = 0.0
    if r > 0.0:
        u += 0.1818*np.exp(-3.2*x)
        u += 0.5099*np.exp(-0.9423*x)
        u += 0.2802*np.exp(-0.4029*x)
        u += 0.02817*np.exp(-0.2016*x)
        u *= za*zb*qe_sq/r

    return u

def f_pair(r, param_a, param_r, za=78, zb=78, ri=1.0, ro=2.0):
    """
    Overall EAM pair potential combining inner, transition, and outer parts.
    The inner part is fixed, while the outer part is based on supplied spline
    function (cubic by default). Transition part ensures smooth change from
    inner to outer.

    Parameters
    ----------
    r: float
       pair distance
    param_a, param_r: lists of floats
                      Parameters of the cubic spline for 
    za, zb: floats
            atomic numbers of the two atoms (default to W-W)
    ri, ro: floats
            inner and outer boundary of the transition region

    Returns
    -------
    u: float
       value of the pair potential at pair distance r
    """

    if r < ri:
        u = u_core(r, za, zb)

    elif r < ro:
        x = (ro + ri - 2*r)/(ro - ri)
        eta = 3/16*x**5 - 5/8*x**3 + 15/16*x + 1/2

        unucl = u_core(r, za, zb)
        uequi = u_equi(r, pair_a, pair_r, dens_a, dens_r)

        u = uequi + eta*(unucl - uequi)

    else:
        u = u_equi(r, pair_a, pair_r, dens_a, dens_r)

    return u

def utot_EAM_per_atom(params, ustats, hparams=None):
    """
    Calculates configurational energy from EAM sufficient statistics and model parameters

    Parameters
    ----------
    params : list of lists and floats
             EAM interaction parameters (spline coefficients array and embedding function parameters)
    ustats : list of lists and floats
             Sufficient statistics for a trajectory of configurations
    hparams: dict of lists
             hyperparameters - spline knots for pair potential and density function

    Returns
    -------
    u_total: float
             total configurational energy (sum of pair and manybody interactions) for trajectory of configurations
    """

    n_sample = len(ustats)

    # assign parameters to different functions
    if not hparams:     # no hparams given
        hp = params[2:] # pair interaction coefficients
        hd = [1.0]      # electronic density coefficients. Default single coefficient with value 1
        #hc = [0.0]
    else:
        # pair interaction coefficients
        npair = len(hparams['pair']) 
        hp = params[2:2+npair]

        # electronic density coefficients. The first coefficient is always 1
        ndens = len(hparams['edens'])
        if ndens < 1:
            hd = [1.0]
        else:
            #if len(params[2+npair:-1]) == ndens:
            if len(params[2+npair:]) == ndens:
                hd = params[2+npair:2+npair+ndens]
            else:
                #assert 2+npair+ndens-1 == len(params), f"Wrong number of parameters: {len(params)} vs. {2+npair+ndens-1}"
                assert 2+npair+ndens-1 == len(params), "Wrong number of parameters: {} vs. {}".format(len(params), 2+npair+ndens-1)
                hd = np.concatenate((params[2+npair:2+npair+ndens-1], [1.0]))

        #hc = params[-1:]
    #!!! not general !!! limit last edens parameter to positive values to preserve positive discriminant
    if hd[-1] < 0.0:
        hd[-1] = 0.0
        params[2+npair:2+npair+ndens] = 0.0

    # pair interactions (per box) from array of spline coefficeints and corresponding statistic
    # sum over spline components, make array over samples
    u_pair = np.array([sum([a*s for a, s in zip(hp, ustats[i][2][:])]) for i in range(n_sample)])

    # cycle over samples for manybody interactions
    embed_r = []
    embed_2 = []
    for i in range(n_sample):
        # calculate electronic density for each atom
        # coefficient for the last spline section is 1 by definition
        # rho_func.shape should be (n_atom, )
        rho_func = sum([p*s for p, s in zip(hd, ustats[i][3][:])]) 

        #print('rho_func:\n', sum(rho_func)/len(rho_func))

        #assert rho_func.shape[0] == ustats[i][3][0].shape[0], f"rho_func shape {rho_func_shape[0]} does not match number of atoms == ustats shape {ustats[i][2][0].shape[0]}"
        assert rho_func.shape[0] == ustats[i][3][0].shape[0], "rho_func shape {} does not match number of atoms == ustats shape {}".format(rho_func_shape[0], ustats[i][2][0].shape[0])

        # sum sqrt and squared atom contributions to embedding function
        embed_r.append(np.sum(np.sqrt(rho_func)))
        embed_2.append(np.sum(rho_func**2))


    # manybody interactions from embedding function parameters and corresponding statistics
    # u_many = np.array([params[0]*ustats[i][0][hp] + params[1]*ustats[i][1][hp] for i in range(n_sample)])
    u_many = np.array([params[0]*embed_r[i] + params[1]*embed_2[i] for i in range(n_sample)])


    #assert u_pair.shape == u_many.shape, f"Shapes of u_pair ({u_pair.shape}) and u_many ({u_many.shape}) do not match."
    assert u_pair.shape == u_many.shape, "Shapes of u_pair ({}) and u_many ({}) do not match.".format(u_pair.shape, u_many.shape)

    u_total = 0.5*u_pair + u_many

    # apply long range correction: correction * number of particles * concentration * density (stored in ustats[4]) 
    #u_total += np.array([sum([a*s for a, s in zip(hc, ustats[i][4][:])]) for i in range(n_sample)])

    return u_total


def utot_EAM_per_box(params, ustats, hparams=[-1]):
    """
    Calculates configurational energy from EAM sufficient statistics and model parameters

    Parameters
    ----------
    params : list of lists and floats
             EAM interaction parameters (spline coefficients array and embedding function parameters)
    ustats : list of lists and floats
             Sufficient statistics for a trajectory of configurations
    hparams: list of ints
             hyperparameters - distance cutoff of the density function

    Returns
    -------
    u_total: float
             total configurational energy (sum of pair and manybody interactions) for trajectory of configurations
    """

    n_sample = len(ustats)

    # pair interaction coefficients
    npair = len(hparams['pair']) 
    hp = params[2:2+npair]

    # electronic density coefficients. The first coefficient is always 1
    assert len(hparams['edens']) == 1, 'number of edens knots not equal to 1: {0}'.format(len(hparams['edens'])) 
    assert len(params) == 2 + len(hp), 'number of params does not sum to 2+pair params: {0}'.format(len(params))

    # pair interactions from array of spline coefficeints and corresponding statistic
    u_pair = np.array([sum([a*s for a, s in zip(hp, ustats[i][2][:])]) for i in range(n_sample)])

    # manybody interactions from embedding function parameters and corresponding statistics
    u_many = np.array([params[0]*ustats[i][0][0] + params[1]*ustats[i][1][0] for i in range(n_sample)])

    u_total = 0.5*u_pair + u_many
    #print(u_pair, u_many, u_total)

    return u_total


def ftot_EAM(params, fstats):
    """
    Calculates configurational energy from EAM sufficient statistics and model parameters

    Parameters
    ----------
    params : list of lists and floats
             EAM interaction parameters (spline coefficients array and embedding function parameters)
    fstats : list of lists and floats
             Sufficient statistics

    Returns
    -------
    f_total: float
             total configurational energy (sum of pair and manybody interactions)
    """

    # number of samples and atoms
    n_sample = len(fstats)
    
    
    # cycle over samples
    f_total = []
    for i in range(n_sample):

        # pair interactions from array of spline coefficeints and corresponding statistic
        f_pair = sum([p*s for p, s in zip(params[2:], fstats[i][2][:])]) 

        # manybody interactions from embedding function parameters and corresponding statistics
        f_many = params[0]*fstats[i][0][-1] + params[1]*fstats[i][1][-1]
        
        n_atom = fstats[i][0][0].shape[0]
        # Create a 6N + 1 array of 0, f, and -f
        fx = np.zeros((6*n_atom + 1), dtype=float)
        fx[1:3*n_atom+1] = 0.5*f_pair.flatten() + f_many.flatten()
        fx[3*n_atom+1:] = -fx[1:3*n_atom+1]
        #print('natom', n_atom, type(f_pair), type(f_many), f_pair.shape, f_many.shape)
 
        f_total.append(fx)
        
    return np.array(f_total)


def sd2_loss(params, targets, stats, utot_func, ftot_func=None, dl=0.05, verbose=0):
    """
    Calculates squared statistical distance loss function for configurational energies and forces.

    Parameters
    ----------
    params : ndarray of floats
             EAM interaction parameters (spline coefficients array and embedding function parameters)
    targets: dict of dicts
             target energies and forces
    stats  : dict of dicts
             Statistics describing particle configurations
    utot_func: function
               takes parameters and statistics and returns configurational energies
    ftot_func: function
               takes parameters and statistics and returns atomic forces
    dl: float
        coordinate perturbation magnitude for energy calculation from forces: du = f*dl

    Returns
    -------
    sd2, sd2f: float
               Squared statistical distances between model and target (energy and force-based)
    """

    # apply bounds on parametes
    #p = np.where(p < -1.0, -1.0, p)
    #p = np.where(p >  1.0,  1.0, p)

    hparams = stats['hyperparams']

    # cycle over target system trajectories and statistics to determine SD
    sd2 = sd2f = 0.0
    for key in targets.keys():

        targ = targets[key]
        stat = stats[key]

        beta = np.mean(targ['beta']) # system inverse temperature
        u_targ = np.array(targ['energy']) # target energies
        u_stat = stat['energy'] # energy statistics
        n_sample = len(u_targ)
        w = targ.get('weight', 1.0)

        #print('len', len(u_targ), len(u_stat))

        if w == 0.0:
            continue

        # energy diference array for a given target trajectory
        #print('all hparams:', key, hparams)
        uuu = beta*(utot_func(params, u_stat, hparams) - u_targ) # array(n_sample)
        uuu -= np.mean(uuu)
        eee = np.exp(-uuu)
        #print('uuu', uuu)
        #print('utarg', u_targ)
        #print('ustat', utot_func(params, u_stat))
        #print('eee', eee)
        
        #print('sd2', utot_EAM(params, u_stat)[0], u_targ[0])

        # are we using forces?
        if (not ftot_func) or ('forcesx' not in targ):

            # energy-based free energy difference and statistical distance
            ge = -np.log(np.mean(eee))   # free energy difference (shifted)
            cb = np.mean(np.exp(-0.5*(uuu - ge))) # Bhattacharyya coefficient
            #print('cb', key, cb)
            sd2 += w*np.arccos(cb)**2              # statistical distance

        else:

            betad = beta*dl  # beta * dl
            f_targ = targ['forces'] # target forces (n_sample, 1+6N) (0, 3Nf, -3Nf)
            f_stat = stat['forces'] # force statistics (n_sample, npars, 3N)
            #print('ftarg', f_targ[-1].shape)
            #print('fstat', f_stat[-1].shape)

            eeh = np.exp(-0.5*uuu)
            fff = ftot_func(params, f_stat) # n_sample *(6N + 1) force contributions
            #print('s', fff[-1][:])
            #print('t', f_targ[-1][:])
            #print('ee', np.mean(np.exp(betad*fff[-1])), eee[-1]*np.mean(np.exp(betad*fff[-1]))) 
            #print('eall', eee[:])

            # target and model force terms
            fpave = np.mean([np.mean(np.exp(betad*f_targ[i])) for i in range(n_sample)])
            fqave = np.mean([eee[i]*np.mean(np.exp(betad*fff[i])) for i in range(n_sample)])
            fhave = np.mean([eeh[i]*np.mean(np.exp(0.5*betad*(fff[i]+f_targ[i]))) for i in range(n_sample)])
            
            # force-based free energy difference and statistical distance
            gef = -np.log(fqave/fpave)
            cb = fhave/(fqave*fpave)**0.5
            if cb > 1: cb = 1
            sd2f += w*np.arccos(cb)**2

            #print('fff', fpave, fqave, fhave, gef, cb, betad)
    
    if verbose > 0:
        print('loss', sd2+sd2f, sd2, sd2f)
        #print('params', params, type(params))

    # add regularization condition
    
    return sd2 + sd2f


def udif_print(params, targets, stats, utot_func):
    """
    Calculates squared statistical distance loss function for configurational energies and forces.

    Parameters
    ----------
    params : list of lists and floats
             EAM interaction parameters (spline coefficients array and embedding function parameters)
    targets: list of lists and floats
             target energies and forces
    stats  : list of lists and floats
             Sufficient statistics
    utot_func: function
               takes parameters and statistics and returns configurational energies

    Returns
    -------
    opti_out, targ_out: lists of floats
            model and target configurational energies
    """
    
    hparams = stats['hyperparams']

    opti_out = {}
    targ_out = {}

    # cycle over target system trajectories and statistics to determine SD
    for key in targets.keys():

        targ = targets[key]
        stat = stats[key]

        u_targ = targ['energy'] # target energies
        u_stat = stat['energy'] # energy statistics

        opti_out[key] = list(utot_func(params, u_stat, hparams))
        targ_out[key] = list(u_targ)
    
    return opti_out, targ_out

def u_components(all_params, stats):
    """
    Calculates configurational energy from EAM sufficient statistics and model parameters

    Parameters
    ----------
    all_params : dict of lists of floats
             EAM interaction parameters and hperparameters
    stats : list of lists and floats
             Sufficient statistics for a trajectory of configurations

    Returns
    -------
    u_parts: dict of lists of floats
        components of total energy (+electronic density) for each
        configuration in trajectory
    """

    hparams = all_params['hyperparams']
    params = all_params['params']

    ustats = stats['energy']

    n_sample = len(ustats)

    # pair interaction coefficients
    npair = len(hparams['pair']) 
    hp = params['pair']
    #print('hp', hp)
    #print(ustats)#), len(ustats[0][2]))

    # electronic density coefficients. The first coefficient is always 1
    ndens = len(hparams['edens'])
    hd = params['edens']

    # embedding function
    he = params['embed']

    #print('n', n_sample, npair, ndens)

    # pair interactions (per box) from array of spline coefficeints and corresponding statistic
    # sum over spline components, make array over samples
    u_pair = np.array([sum([a*s for a, s in zip(hp, ustats[i][2][:])]) for i in range(n_sample)])

    # cycle over samples for manybody interactions
    embed_r = []
    embed_2 = []
    edens = []
    for i in range(n_sample):
        # calculate electronic density for each atom
        # coefficient for the last spline section is 1 by definition
        # rho_func.shape should be (n_atom, )

        rho_func = sum([p*s for p, s in zip(hd, ustats[i][3][:])]) 
        #print('type', type(rho_func), rho_func.shape)

        edens.append(np.sum(rho_func)/rho_func.shape[0])

        #assert rho_func.shape[0] == ustats[i][3][0].shape[0], f"rho_func shape {rho_func_shape[0]} does not match number of atoms == ustats shape {ustats[i][2][0].shape[0]}"
        assert rho_func.shape[0] == ustats[i][3][0].shape[0], "rho_func shape {} does not match number of atoms == ustats shape {}".format(rho_func_shape[0], ustats[i][2][0].shape[0])

        # sum sqrt and squared atom contributions to embedding function
        embed_r.append(np.sum(np.sqrt(rho_func)))
        embed_2.append(np.sum(rho_func**2))


    u_many = np.array([he[0]*embed_r[i] + he[1]*embed_2[i] for i in range(n_sample)])

    assert u_pair.shape == u_many.shape, "Shapes of u_pair ({}) and u_many ({}) do not match.".format(u_pair.shape, u_many.shape)

    u_total = 0.5*u_pair + u_many

    u_parts = {'u_pair':u_pair, 'edens':edens, 'u_many':u_many, 'u_total': u_total}

    return u_parts
