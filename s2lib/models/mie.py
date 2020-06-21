from __future__ import print_function #, unicode_literals
from __future__ import absolute_import, division
try:
    xrange = xrange
    # We have Python 2
except:
    xrange = range
    # We have Python 3

"""
Collection of Mie potential functions
"""

def utot_Mie(params, ustats):
    """
    Calculates configurational energy from pair distance statistics and model parameters

    Parameters
    ----------
    params : list of lists and floats
             EAM interaction parameters (spline coefficients array and embedding function parameters)
    ustats : list of lists and floats
             Sufficient statistics for a trajectory of configurations

    Returns
    -------
    u_total: float
             total configurational energy (sum of pair and manybody interactions) for trajectory of configurations
    """

    n_sample = len(ustats)

    # pair interactions from array of spline coefficeints and corresponding statistic
    u_pair = np.array([sum([a*s for a, s in zip(params[2:], ustats[i][2,:])]) for i in range(n_sample)])

    # manybody interactions from embedding function parameters and corresponding statistics
    u_many = np.array([params[0]*ustats[i][0, -1] + params[1]*ustats[i][1, -1] for i in range(n_sample)])

    u_total = 0.5*u_pair + u_many

    return u_total

def ftot_Mie(params, fstats):
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
        f_pair = sum([p*s for p, s in zip(params[2:], fstats[i][2,:])]) 

        # manybody interactions from embedding function parameters and corresponding statistics
        f_many = params[0]*fstats[i][0,-1] + params[1]*fstats[i][1, -1]
        
        n_atom = fstats[i][0,0].shape[0]
        # Create a 6N + 1 array of 0, f, and -f
        fx = np.zeros((6*n_atom + 1), dtype=float)
        fx[1:3*n_atom+1] = 0.5*f_pair.flatten() + f_many.flatten()
        fx[3*n_atom+1:] = -fx[1:3*n_atom+1]
        #print('natom', n_atom, type(f_pair), type(f_many), f_pair.shape, f_many.shape)
 
        f_total.append(fx)
        
    return np.array(f_total)


def sd2_loss(params, stats, targets, utot_func, ftot_func=None, dl=0.05, verbose=0):
    """
    Calculates squared statistical distance loss function for configurational energies and forces.

    Parameters
    ----------
    params : list of lists and floats
             EAM interaction parameters (spline coefficients array and embedding function parameters)
    stats  : list of lists and floats
             Sufficient statistics
    targets: list of lists and floats
             target energies and forces
    utot_func: function
               takes parameters and statistics and returns configurational energies
    ftot_func: function
               takes parameters and statistics and returns atomic forces
    dl: float
        coordinate perturbation magnitude for energy calculation from forces: du = f*dl

    Returns
    -------
    sd2, sd2f: float
               squared statistical distances between model and target (energy and force-based)
    """

    # apply bounds on parametes
    #p = np.where(p < -1.0, -1.0, p)
    #p = np.where(p >  1.0,  1.0, p)

    # cycle over target system trajectories and statistics to determine SD
    sd2 = sd2f = 0.0
    for targ, stat in zip(targets, stats):

        beta = np.mean(targ['beta']) # system inverse temperature
        u_targ = np.array(targ['energy']) # target energies
        u_stat = stat['energy'] # energy statistics
        n_sample = len(u_targ)
        w = targ.get('weight', 1.0)

        # energy diference array for a given target trajectory
        uuu = beta*(utot_EAM(params, u_stat) - u_targ) # array(n_sample)
        uuu -= np.mean(uuu)
        eee = np.exp(-uuu)
        
        #print('sd2', utot_EAM(params, u_stat)[0], u_targ[0])

        # are we using forces?
        if (not ftot_func) and ('forcesx' not in targ):

            # energy-based free energy difference and statistical distance
            ge = -np.log(np.mean(eee))   # free energy difference (shifted)
            cb = np.mean(np.exp(-0.5*(uuu - ge))) # Bhattacharyya coefficient
            sd2 += w*np.arccos(cb)**2              # statistical distance

        else:

            betad = beta*dl  # beta * dl
            f_targ = targ['forces'] # target forces (n_sample, 1+6N) (0, 3Nf, -3Nf)
            f_stat = stat['forces'] # force statistics (n_sample, npars, 3N)

            eeh = np.exp(-0.5*uuu)
            fff = ftot_func(params, f_stat) # n_sample *(6N + 1) force contributions

            # target and model force terms
            fpave = np.mean([np.mean(np.exp(betad*f_targ[i])) for i in range(n_sample)])
            fqave = np.mean([eee[i]*np.mean(np.exp(betad*fff[i])) for i in range(n_sample)])
            fhave = np.mean([eeh[i]*np.mean(np.exp(0.5*betad*(fff[i]+f_targ[i]))) for i in range(n_sample)])
            
            # force-based free energy difference and statistical distance
            gef = -np.log(fqave/fpave)
            cb = fhave/(fqave*fpave)**0.5
            if cb > 1: cb = 1
            sd2f += w*np.arccos(cb)**2
    
    if verbose > 0:
        print('loss', sd2+sd2f)
    
    return sd2 + sd2f
