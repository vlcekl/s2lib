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


def make_input_matrices_forces(target, stats, keytrj=None, combined=0.0, dl_dict=None):
    """
    Creates input data for energy minimization with target as dependent variable and stats as independent.
    Assumes that all appropriate knots from stats have been selected, so it includes everything.
    """
    
    # matrix of independent variables (Embedding and B-spline coefficients)
    X0 = []
    X1 = []
    X2 = []
    X3 = []

    # vector of dependent variable (configurational energies)
    y = []
    # weights of individual trajectories
    weights = []
    # force weights
    dl = []
    # fatoms
    force_atoms = []
    # vector of inverse temperatures
    beta = []
    # bounds of trajectories in the overall design matrix
    bounds = []

    #keys = list(target.keys())
    
    if keytrj is not None:
        keys = keytrj
    else:
        keys = list(target.keys())

    if dl_dict is None:
        dl_dict = {key:0.0 for key in keys}

    max_features = 0
    max_force_pars = 0
    max_atoms = 0
    max_fatoms = 0
    max_nbrs = 0
    max_y = 0
    for key in keys:
        
        w = target[key]['weight']
        fw = dl_dict[key]
        fatoms = target[key]['fatoms']
        
        # eliminate trajectories with 0 weight
        if w == 0.0:
            continue

        lo_bound = len(y)
        
        # cycle over samples (configurations)
        se = stats[key]['energy']
        te = target[key]['energy']
        sf = stats[key]['forces']
        tf = target[key]['forces']
        print('lens forces', len(sf), len(tf), len(se), len(te))
        be = target[key]['beta']

        for i, (config, targ_energy, stat_force, targ_force, bb) in enumerate(zip(se, te, sf, tf, be)):
        #for i, (config, energy, bb) in enumerate(zip(stats[key]['energy'], target[key]['energy'], target[key]['beta'])):
            
            # add energy
            y.append(np.concatenate(([targ_energy], targ_force)))
            
            beta.append(bb)
            #weights.append(w)
            
            # create an array of independent variables
            x_vars = []
            
            # embedding for additive model
            #x_vars += [config[0][0], config[1][0]]

            # pair interactions b-spline stats. Adds a list of descriptors
            x_vars += list(0.5*config[2])

            X0.append(x_vars)
            
            # per atom edens b-spline stats. Adds an array (n_features, n_atoms)
            # n_features == n_params
            xn_vars = config[3]
            X1.append(xn_vars)

            # forces statistics
            # per atom b-spline stats. Adds an array (n_features, n_atoms, 3)
            xnn_vars = stat_force[2]
            X2.append(xnn_vars)

            xnnn_vars = stat_force[3]
            X3.append(xnnn_vars)

            max_atoms = max(max_atoms, xn_vars.shape[1])
            max_features = max(max_features, xn_vars.shape[0])
            max_force_pars = max(max_force_pars, xnn_vars.shape[0])
            max_y = max(max_y, len(y[-1]))
            max_fatoms = max(max_fatoms, xnnn_vars.shape[1])
            max_nbrs = max(max_nbrs, xnnn_vars.shape[2])
            
        bounds.append(slice(lo_bound, len(y), 1))
        weights.append(w)
        force_atoms.append(fatoms)
        dl.append(fw)

    if combined > 0.0:
        # add trajectory of zeros by replicating 'inf'
        
        config = stats['inf']['energy'][0]
        targ_energy = target['inf']['energy'][0]
        bb = target['inf']['beta'][0]
        stat_force = stats['inf']['forces'][0]
        targ_force = target['inf']['forces'][0]
        
        for i in range(200):
            # add energy
            y.append(np.concatenate(([targ_energy], targ_force)))
            beta.append(bb)
            #weights.append(w)
            
            # create an array of independent variables
            x_vars = []
            
            # embedding for additive model
            #x_vars += [config[0][0], config[1][0]]

            # pair interactions b-spline stats. Adds a list of descriptors
            x_vars += list(0.5*config[2])
            
            # per atom edens b-spline stats. Adds an array (n_features, n_atoms)
            xn_vars = config[3]

            # forces statistics
            xnn_vars = stat_force[2]
            xnnn_vars = stat_force[3]
            
            max_atoms = max(max_atoms, xn_vars.shape[1])
            max_features = max(max_features, xn_vars.shape[0])
            max_force_pars = max(max_force_pars, xnn_vars.shape[0])
            max_y = max(max_y, len(y[-1]))
            max_fatoms = max(max_fatoms, xnnn_vars.shape[1])
            max_nbrs = max(max_nbrs, xnnn_vars.shape[2])

            X0.append(x_vars)
            X1.append(xn_vars)
            X2.append(xnn_vars)
            X3.append(xnnn_vars)
            
        bounds.append(slice(0, len(y), 1))
        weights.append(combined)
        dl.append(0.0)
        force_atoms.append([])
    
    # Additive features to a 2D array in X[0] 
    X0 = np.array(X0)
    X = [X0]
    
    # Non-additive features to a 3D array to be filled with density function statistics.
    # Organize the dimensions as (n_samples, n_atoms, n_features) so that dot product
    # between edens parameters and the array to compute density on individual atoms
    # is along the last (contiguous) dimension.
    X.append(np.zeros((len(X1), max_atoms, max_features), dtype=float))
    for i in range(len(X1)):
        X[1][i,:X1[i].shape[1],:] = X1[i].T

    # Non-additive features to a 3D array to be filled with density function statistics.
    # Organize the dimensions as (n_samples, n_atoms, n_features) so that dot product
    # between edens parameters and the array to compute density on individual atoms
    # is along the last (contiguous) dimension.
    X.append(np.zeros((len(X2), max_atoms, 3, max_force_pars), dtype=float))
    for i in range(len(X1)):
        X[2][i, :X2[i].shape[1], 0:3, :] = np.transpose(X2[i], (1, 2, 0)) 

    X.append(np.zeros((len(X3), max_fatoms, max_nbrs, 3, max_force_pars), dtype=float))
    for i in range(len(X1)):
        X[3][i, :X3[i].shape[1], :X3[i].shape[2], 0:3, :] = np.transpose(X3[i], (1, 2, 3, 0)) 
    
    yy = np.zeros((len(y), max_y), dtype=float)
    for i in range(len(y)):
        yy[i,:len(y[i])] = y[i]

    #y = np.array(y)
    
    assert len(yy) == len(X[0]), "Shapes of y and X[0] do not match"
    assert len(yy) == len(X[1]), "Shapes of y and X[1] do not match."
    assert len(yy) == len(X[2]), "Shapes of y and X[2] do not match."
    assert len(yy) == len(X[3]), "Shapes of y and X[3] do not match."
    
    print('bounds', bounds)
    #print('weights', weights)
    #print('y.shape', yy.shape)
    #print('yy',yy[0], yy[800])

    return X, yy, np.array(weights), np.array(beta), np.array(dl), bounds, force_atoms


def energy(params, X):
    """ Configurational energy of an EAM model.
    """
    
    n_edens = X[1].shape[-1]
    
    # Pair energy
    energy = X[0].dot(params[1:-n_edens])
    
    # calculates an (n_samples, n_atoms) matrix of atomic densities
    edens = X[1].dot(params[-n_edens:])

    # Manybody energy: A*sum(dens**0.5) + B*sum(dens**2)
    # Here we set A to -1 to eliminate colinearity
    energy += -1.0*np.sum(np.sqrt(edens), axis=1)
    energy += params[0]*np.sum(edens**2, axis=1)
        
    return energy

def loss_energy(params, X, y, weights):
    """Total energy loss (least squares)"""

    du = y - energy(params, X)
    loss = du.T.dot(np.diag(weights)).dot(du)

    return loss


def loss_sd2(params, X, y, weights, bounds, beta):
    """Statistical distance loss"""

    beta_du = beta*(energy(params, X) - y)

    # divide system into individual trajectories (use bounds)
    loss = 0.0
    for ib, bound_slice in enumerate(bounds):
        du = beta_du[bound_slice]                # du (view of the original beta*du)
        du_ave = np.mean(du)                     # average du
        exp_duh = np.exp(-0.5*(du - du_ave))     # exp[-beta*du/2]
        cb = np.mean(exp_duh)                    # cb = <exp[-beta*du/2]>
        cb /= np.sqrt(np.mean(exp_duh**2))       # Bhattacharyya coeff.: cb/exp[-beta*dF/2]
        #print('loss cb', cb)
        loss += weights[ib]*np.arccos(cb)**2   # statistical distance

    return loss


def forces_EAM(params, X, bound_slice, fatoms):

    n_pair = X[0].shape[-1]
    n_edens = X[1].shape[-1]
    
    # Pair energy
    energy = X[0][bound_slice].dot(params[1:-n_edens])
    
    # calculates an (n_samples, n_atoms) matrix of atomic densities
    edens = X[1][bound_slice].dot(params[-n_edens:])
    edens_sqrt = np.sqrt(edens)

    # Manybody energy: A*sum(dens**0.5) + B*sum(dens**2)
    # Here we set A to -1 to eliminate colinearity
    energy += -1.0*np.sum(edens_sqrt, axis=1)
    energy += params[0]*np.sum(edens**2, axis=1)

    # pair forces (n_sample, n_atom, n_dim, n_params)->(n_sample, n_atom, n_dim)
    #forces = X[2][bound_slice, :, :, 0:n_pair].dot(params[1:-n_edens])

    n_fatoms = len(fatoms)
    n_atom = X[2].shape[1]
    n_replicas = X[3].shape[2]//X[2].shape[1]
    n_nbrs = n_replicas*n_atom
    # pair forces (n_sample, n_fatom, n_nbrs, n_dim, n_params)->(n_sample, n_atom, n_dim)
    # sum over neighbors
    forces = np.sum(X[3][bound_slice, :n_fatoms, :n_nbrs, :, 0:n_pair].dot(params[1:-n_edens]), axis=2)
    #print('f1', forces[0,0])

    # add manybody forces ->(n_sample, n_atom, n_dim) for all atoms
    with np.errstate(divide='ignore'):
        # (-1) is there for the constant parameter value
        dens_deriv = np.nan_to_num(-1.0/(2.0*np.sqrt(edens)) + 2.0*params[0]*edens)
        # stretch to copies

    # edens grad (n_sample, n_fatom, n_nbrs, n_dim, n_params)->(n_sample, n_fatom, n_nbrs, n_dim)
    forces_edens = X[3][bound_slice, :, :, :, 0:n_edens].dot(params[-n_edens:])
    
    # density contribution from neighbors
    nbr_deriv = np.concatenate(n_replicas*[dens_deriv], axis=1)
    #print('bnd_slice', bound_slice, n_fatoms, forces.shape, nbr_deriv.shape, forces_edens[:, :n_fatoms, :n_replicas*n_atom, :].shape)
    forces += np.sum(nbr_deriv[:, None, :, None]*forces_edens[:, :n_fatoms, :n_replicas*n_atom, :], axis=2) 
    #print('f2', forces[0,0])

    # density contribution from fatoms
    fat_deriv = dens_deriv[:,fatoms]
    #print('bnd_slice', bound_slice, len(fatoms), forces.shape, fat_deriv.shape, forces_edens[:, :n_fatoms, :n_replicas*n_atom, :].shape)
    forces += np.sum(fat_deriv[:, :, None, None]*forces_edens[:, :n_fatoms, :n_replicas*n_atom, :], axis=2) 
    #print('f3', forces[0,0])

    #forces += np.sum(dens_self_deriv[:,:,:,None]*forces_edens, axis=2) 
    #forces += (-1/(2*edens_sqrt[:,:,None]) + 2*edens[:,:,None]*params[0])*forces_edens

    n_sample, n_fatom, n_dim = forces.shape
    n_comps = n_fatom*n_dim

    forces = np.reshape(forces, (n_sample, -1))

    forces_flat = np.empty((n_sample, 2*n_comps + 1), dtype=float)
    forces_flat[:, 0] = 0.0
    forces_flat[:, 1:n_comps+1] = -forces
    forces_flat[:, n_comps+1:] = +forces

    #return energy, forces_flat
    return forces_flat

def loss_sd2_forces(params, X, y, weights, bounds, beta, dl, force_atoms):
    """Statistical distance loss including forces

    Parameters
    ----------
    dl : ndarray
        delta l - force weights for each trajectory
    """

    # energy differences
    beta_du = beta*(energy(params, X) - y[:, 0])

    loss = 0.0
    for ib, bound_slice in enumerate(bounds):
        du = beta_du[bound_slice]             # du (view of the original beta*du)
        du_ave = np.mean(du)                   # average du
        exp_du = np.exp(-(du - du_ave))
        exp_duh = np.sqrt(exp_du)     # exp[-beta*du/2]
        dd = dl[ib]
        fatoms = force_atoms[ib]
        n_fatoms = len(fatoms)

        if dd == 0.0:
            cb = np.mean(exp_duh)              # cb = <exp[-beta*du/2]>
            cb /= np.sqrt(np.mean(exp_du))       # Bhattacharyya coeff.: cb/exp[-beta*dF/2]
        else:
            db = dd*beta[bound_slice]
            f_targ = np.exp(db[:, None]*y[bound_slice, 1:2+6*n_fatoms])
            f_modl = np.exp(db[:, None]*forces_EAM(params, X, bound_slice, fatoms)) # n_sample * (6N + 1) force contributions
            print('targ', np.sum((y[bound_slice, 1:][2,1:4]-forces_EAM(params, X, bound_slice, fatoms)[2, 1:4])**2))
            print('targ', y[bound_slice, 1:][0,1:4], forces_EAM(params, X, bound_slice, fatoms)[0, 1:4])
            print('targ', y[bound_slice, 1:][1,1:4], forces_EAM(params, X, bound_slice, fatoms)[1, 1:4])
            print('targ', y[bound_slice, 1:][2,1:4], forces_EAM(params, X, bound_slice, fatoms)[2, 1:4])
            print('targ', y[bound_slice, 1:][3,1:4], forces_EAM(params, X, bound_slice, fatoms)[3, 1:4])
            print('targ', y[bound_slice, 1:][4,1:4], forces_EAM(params, X, bound_slice, fatoms)[4, 1:4])
            print('targ', y[bound_slice, 1:][5,1:4], forces_EAM(params, X, bound_slice, fatoms)[5, 1:4])

            fpave = np.mean(f_targ)
            fqave = np.mean(exp_du*np.mean(f_modl, axis=1))
            fhave = np.mean(exp_duh*np.mean(np.sqrt(f_modl*f_targ), axis=1))

            cb = fhave/(fqave*fpave)**0.5
            if cb > 1.0:
                cb = 1.0

            #fpave = np.mean([np.mean(np.exp(betad*f_targ[i])) for i in range(n_sample)])
            #fqave = np.mean([eee[i]*np.mean(np.exp(betad*fff[i])) for i in range(n_sample)])
            #fhave = np.mean([eeh[i]*np.mean(np.exp(0.5*betad*(fff[i]+f_targ[i]))) for i in range(n_sample)])
            #gef = -np.log(fqave/fpave)

        loss += weights[ib]*np.arccos(cb)**2   # statistical distance

    return loss

def loss_diff_penalty(params, penalty_mat, alpha):
    """Difference penalty loss for B-splines"""
    return 0.5*alpha*params.T.dot(penalty_mat).dot(params)


def loss_energy_penalized(params, X, y, weights, penalty_mat, alpha):
    """Total energy loss with difference penalty"""

    loss = loss_energy(params, X, y, weights)
    loss_diff = loss_diff_penalty(params, penalty_mat, alpha)
    
    print(loss + loss_diff, loss, loss_diff)

    return loss + loss_diff

def loss_sd2_penalized(params, X, y, weights, bounds, beta, penalty_mat, alpha):
    """Total sd2 loss with difference penalty"""
    
    loss = loss_sd2(params, X, y, weights, bounds, beta)
    loss_diff = loss_diff_penalty(params, penalty_mat, alpha)

    print(loss + loss_diff, loss, loss_diff)

    return loss + loss_diff

def loss_sd2f_penalized(params, X, y, weights, bounds, beta, dl, force_atoms, penalty_mat, alpha):
    """Total sd2 loss with difference penalty"""
    
    loss = loss_sd2_forces(params, X, y, weights, bounds, beta, dl, force_atoms)
    loss_diff = loss_diff_penalty(params, penalty_mat, alpha)

    print(loss + loss_diff, loss, loss_diff)

    return loss + loss_diff


def gradient_energy(params, X):
    """Calculates gradient of energy with respect to parameters.
    
    Returns
    -------
    grad : ndarray, shape (N, p)
    """

    # electronic densities
    n_edens = X[1].shape[-1]
    edens = X[1].dot(params[-n_edens:])
    
    with np.errstate(divide='ignore'):
        # (-1) is there for the constant parameter value
        tmp = np.nan_to_num(-1.0/(2.0*np.sqrt(edens)) + 2.0*params[0]*edens)

    grad = np.empty((X[0].shape[0], len(params)), dtype=float)
    grad[:, 1:-n_edens] = X[0]                                 # pair
    grad[:, 0] = np.sum(edens**2, axis=1)                      # embed
    grad[:, -n_edens:] = np.sum(tmp[:, :, None]*X[1], axis=1)  # edens

    return grad


def jacobian_energy(params, X, y, weights):
    """Calculates jacobian of energy loss function"""
    
    du = y - energy(params, X)
    grad = gradient_energy(params, X)
    jac = -2.0*grad.T.dot(np.diag(weights)).dot(du)
    
    return jac

def jacobian_sd2(params, X, y, weights, bounds, beta):
    """Calculates jacobian of statistical distance loss function"""
    
    # use reduced units beta*energy throughout
    beta_du = beta*(y - energy(params, X))                    # shape (N,)
    grad_beta_du = beta[:, np.newaxis]*gradient_energy(params, X)  # shape (N, p)

    jac = np.zeros((len(params)), dtype=float)
    
    for ib, bound_slice in enumerate(bounds):
        du = beta_du[bound_slice]               # du (view of the original beta*du)
        du_ave = np.mean(du)                    # average du
        exp_duh = np.exp(-0.5*(du - du_ave))    # exp[-beta*du/2]
        exp_du = exp_duh**2                     # exp[-beta*du]
        exp_dfi = 1.0/np.mean(exp_du)           # 1/<exp[-beta*du]> = exp(beta*dF)
        cb = np.mean(exp_duh)*np.sqrt(exp_dfi)  # Bhattacharyya coefficient
        #print('jac cb', cb)

        # Gradient of free energy (with respect to model parameters)
        # grad_df = <grad_beta_du * exp[-beta*du]> / <exp[-beta*du]>
        grad_du = grad_beta_du[bound_slice]
        grad_df = np.mean(grad_du*exp_du[:, None], axis=0)*exp_dfi
        
        # Gradient of the Bhattacharyya coeff. shape (p,)
        # -1/2 * <exp[-beta*du/2] * (grad_beta_du - grad_df)> / exp[-beta*df/2]
        grad_cb = np.mean((grad_du - grad_df[None, :])*exp_duh[:, None], axis=0)
        grad_cb *= -0.5*np.sqrt(exp_dfi)
        
        # Jacobian
        jac += -2.0*weights[ib]*np.arccos(cb)/np.sqrt(1.0 - cb**2)*grad_cb
    
    return jac

def jacobian_diff_penalty(params, penalty_mat, alpha):
    """Jacobian contribution of the B-spline difference penalty"""
    return alpha*penalty_mat.dot(params)

def jacobian_energy_penalized(params, X, y, weights, penalty_mat, alpha):
    jac = jacobian_energy(params, X, y, weights)
    jac += jacobian_diff_penalty(params, penalty_mat, alpha)
    return jac

def jacobian_sd2_penalized(params, X, y, weights, bounds, beta, penalty_mat, alpha):
    jac = jacobian_sd2(params, X, y, weights, bounds, beta)
    jac += jacobian_diff_penalty(params, penalty_mat, alpha)
    return jac

def make_input_matrices(target, stats, keytrj=None, combined=0.0):
    """
    Creates input data for energy minimization with target as dependent variable and stats as independent.
    Assumes that all appropriate knots from stats have been selected, so it includes everything.
    """
    
    # matrix of independent variables (Embedding and B-spline coefficients)
    X0 = []
    X1 = []
    # vector of dependent variable (configurational energies)
    y = []
    # weights of individual trajectories
    weights = []
    # vector of inverse temperatures
    beta = []
    # bounds of trajectories in the overall design matrix
    bounds = []

    keys = list(target.keys())
    
    if keytrj is not None:
        keys = keytrj
    else:
        keys = list(target.keys())

    max_features = 0
    max_atoms = 0
    for key in keys:
        
        w = target[key]['weight']
        
        # eliminate trajectories with 0 weight
        if w == 0.0:
            continue

        lo_bound = len(y)
        
        # cycle over samples (configurations)

        for i, (config, energy, bb) in enumerate(zip(stats[key]['energy'], target[key]['energy'], target[key]['beta'])):
            
            # add energy
            y.append(energy)
            beta.append(bb)
            #weights.append(w)
            
            # create an array of independent variables
            x_vars = []
            
            # embedding for additive model
            #x_vars += [config[0][0], config[1][0]]

            # pair interactions b-spline stats. Adds a list of descriptors
            x_vars += list(0.5*config[2])
            
            # per atom edens b-spline stats. Adds an array (n_features, n_atoms)
            xn_vars = config[3]
            
            max_features = max(max_features, xn_vars.shape[0])
            max_atoms = max(max_atoms, xn_vars.shape[1])

            X0.append(x_vars)
            X1.append(xn_vars)
            
        bounds.append(slice(lo_bound, len(y), 1))
        weights.append(w)
    
    if combined > 0.0:
        # add trajectory of zeros by replicating 'inf'
        
        config = stats['inf']['energy'][0]
        energy = target['inf']['energy'][0]
        bb = target['inf']['beta'][0]
        
        for i in range(200):
            # add energy
            y.append(energy)
            beta.append(bb)
            #weights.append(w)
            
            # create an array of independent variables
            x_vars = []
            
            # embedding for additive model
            #x_vars += [config[0][0], config[1][0]]

            # pair interactions b-spline stats. Adds a list of descriptors
            x_vars += list(0.5*config[2])
            
            # per atom edens b-spline stats. Adds an array (n_features, n_atoms)
            xn_vars = config[3]
            
            max_features = max(max_features, xn_vars.shape[0])
            max_atoms = max(max_atoms, xn_vars.shape[1])

            X0.append(x_vars)
            X1.append(xn_vars)
            
        bounds.append(slice(0, len(y), 1))
        weights.append(combined)

    # Additive features to a 2D array in X[0] 
    X0 = np.array(X0)
    X = [X0]
    
    # Non-additive features to a 3D array to be filled with density function statistics.
    # Organize the dimensions as (n_samples, n_atoms, n_features) so that dot product
    # between edens parameters and the array to compute density on individual atoms
    # is along the last (contiguous) dimension.
    X.append(np.zeros((len(X1), max_atoms, max_features), dtype=float))
    for i in range(len(X1)):
        X[1][i,:X1[i].shape[1],:] = X1[i].T
    
    y = np.array(y)
    #y = y[:, np.newaxis]
    
    assert len(y) == len(X[0]), "Shapes of y and X[0] do not match"
    assert len(y) == len(X[1]), "Shapes of y and X[1] do not match."
    
    print('bounds', bounds)
    print('weights', weights)

    return X, y, np.array(weights), np.array(beta), bounds
