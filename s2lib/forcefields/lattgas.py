from __future__ import print_function #, unicode_literals
from __future__ import absolute_import, division
try:
    xrange = xrange
    # We have Python 2
except:
    xrange = range
    # We have Python 3

"""
Collection of lattice gas potential functions
"""

import numpy as np
from scipy.stats import chi2

def df_geom(params_list, X_list):
    """Calculates free energy differences between reference systems using geometric estimates.
    
    Parameters
    ----------
    params_list : list of k ndarrays
        list of k parameter lists of n_params length for k systems
    X_list : list of k ndarrays shape(n_samples, n_params)
        configurational statistics needed for energy computation
    
    Returns
    -------
    df : ndarray, shape (k, k)
        pairwise free energy differences
    """
    
    Ktot = len(params_list)
    
    # Calculate energy list len(u)=k of ndarrays, shape(n, k)
    params_mat = np.array(params_list)  # shape (k, p)
    us = [X.dot(params_mat.T) for X in X_list]
    
    #print(np.array([np.mean(X.dot(p)) for p, X in zip(params_list, X_list)]))

    df = np.zeros((Ktot, Ktot), dtype=np.float64)
    
    # Create a (k, k) matrices of free energy estimates and Bhattacharyya coefficients
    for k in range(Ktot):
        for j in range(k+1, Ktot):

            du_jk = -0.5*(us[k][:, j] - us[k][:, k])
            du_ave_jk = np.mean(du_jk)
            du_jk -= du_ave_jk 
            exp_jk = np.mean(np.sort(np.exp(du_jk)))
            
            du_kj = -0.5*(us[j][:, k] - us[j][:, j])
            du_ave_kj = np.mean(du_kj)
            du_kj -= du_ave_kj
            exp_kj = np.mean(np.sort(np.exp(du_kj)))
            
            df[k, j] = -(np.log(exp_jk/exp_kj) + (du_ave_jk - du_ave_kj))
            df[j, k] = -df[k, j]
        
    return df


def average_histogram(params, ref_params_list, X_list, df, hist_list):
    """Combines histogram data from different simulations using MBAR"""
    
    params_mat = np.array(ref_params_list)  # shape (k, p)
    # k lists of ndarrays shape(n_k, k)
    hst = np.concatenate(hist_list, axis=0)
    X = np.concatenate(X_list, axis=0)
    us = X.dot(params_mat.T)

    # k lists of ndarrays shape (n_k,) (new energies for all configurations)
    u_new = X.dot(params[:, None])
    
    # Number of samples for each system
    Ns = np.array([Xi.shape[0] for Xi in X_list])

    us -= u_new
    us = df - us
    us_max = np.max(us)
    us -= us_max
    
    sum_k = np.sum(Ns*np.exp(us), axis=1)
        
    c_a = np.sum(1.0/sum_k)
    c_hist = np.sum(hst/sum_k[:, None], axis=0)
    hist_ave = c_hist/c_a
        
    return hist_ave



def loss_sd2_hist_all(params, X, params_ref, beta, hist_ref, hist_targ):

    s2 = 0.0

    return s2


def loss_sd2_hist(params, X, params_ref, beta, hist_ref, hist_targ):
    """Loss based on a single reference simulation"""

    params = np.where(params < -8.0, -8.0, params)
    params = np.where(params >  1.0,  1.0, params)

    beta_du = beta*X.dot(params - params_ref)
    # histogram reweighting factor
    eee = np.exp(-beta_du + np.max(beta_du))
    #print(eee[:15], np.sum(eee))
    eee /= np.sum(eee)
    #print(params - params_ref, eee[:5], np.sum(eee))

    #hist_ave = average_histogram(pars_in, params_list, X_list[key], df_est[key][0], hist_list[key])

    s2 = 0.0
    # cycle over knn values
    for i, targ_hist in enumerate(hist_targ):
        ref_hist = hist_ref[:,i,:,:]

        #model_hist = np.sum(ref_hist.T*eee, axis=1) # rescaled average reference histogram
        model_hist = np.sum(eee[:,None, None]*ref_hist, axis=0) # rescaled average reference histogram

        #n_nn = np.sum(model_hist)
        #n_tt = np.sum(targ_hist)

        #print(targ_hist)
        #print(model_hist)
        

        #for nn, tt in zip(nnn, ttt): 
        for j, (nni, tti) in enumerate(zip(model_hist[:], targ_hist[:])): 
            n_nn = np.sum(nni)
            n_tt = np.sum(tti)
            nn = nni/n_nn
            tt = tti/n_tt
            print(j)
            print(nn)
            print(tt)
            cb = sum([np.sqrt(t*n) for t, n in zip(tt, nn)])
            print(cb)
            s2 += 4*n_nn*n_tt/(n_nn + n_tt)*np.arccos(cb)**2

    print(s2, params)
    return s2


#def sd2(params, stats, targets):
def sd2_simple(params, stats, targets):
    """Statistical distance between histograms collected from different
    sources

    Parameters
    ----------
    params : list of lists and floats
             lattice model interaction parameters
    stats  : list of lists and floats
    targets: list of lists and floats
             target histograms

    Returns
    -------
    sd2: float
         squared statistical distance between model and target histograms
    """

    # apply bounds on parameters
    params = np.where(params < -10.0, -10.0, params)
    params = np.where(params >  10.0,  10.0, params)

    # nearest and next nearest interactions between unlike particles
    par = np.array([0.0, params[0], 0.0, 0.0, params[1], 0.0])
    #par = np.array([0.0, params[0], params[1], 0.0, params[2], 0.0])

    sd2 = 0.0
    for key in targets.keys():
        targ = targets[key]
        stat = stats[key]

        beta = 1.0/np.mean(stat['temp'])
        hru = stat['interaction_stats'] # interaction histogram statistics
        u_ref = stat['energy'] # reference system energy

        uuu = beta*(np.sum(hru*par, axis=1) - u_ref)
        #print(u_ref[10], np.sum(hru*par, axis=1)[10])
        uuu -= np.mean(uuu)

        # histogram reweighting factor
        eee = np.exp(-uuu)
        eee /= np.sum(eee)

        # statistical distance for surface configuration histograms
        phst = targ['config_stats']     # target histogram
        rhst = stat['config_stats']     # reference histograms for each configuration
        qhst = np.sum(rhst.T*eee, axis=1) # rescaled average reference histogram
        w = targ.get('weight', 1.0)     # weight of the target data set

        # Statistical distance contribution
        sd2 += w*np.arccos(np.sum(np.sqrt(qhst*phst)))**2
        #print(key, w, np.arccos(np.sum(np.sqrt(qhst*phst)))**2)

    #print('---', sd2)
    return sd2

#def sd2_extended(params, stats, targets):
def sd2(params, stats, targets):
    """Statistical distance between histograms collected from different
    sources. Takes into account the uncertainty stemming from distance from
    the reference system.

    Parameters
    ----------
    params : list of lists and floats
             lattice model interaction parameters
    stats  : list of lists and floats
    targets: list of lists and floats
             target histograms

    Returns
    -------
    sd2: float
         squared statistical distance between model and target histograms
    """

    # apply bounds on parameters
    params = np.where(params < -2.0, -2.0, params)
    params = np.where(params >  2.0,  2.0, params)

    # nearest and next nearest interactions between unlike particles
    par = np.array([0.0, params[0], 0.0, 0.0, params[1], 0.0])
    #par = np.array([0.0, params[0], params[1], 0.0, params[2], 0.0])

    sd2 = 0.0
    sd2_ref = 0.0
    for key in targets.keys():
        targ = targets[key]
        stat = stats[key]
        w = targ.get('weight', 1.0)     # weight of the target data set
        w = 1.0

        beta = 1.0/np.mean(stat['temp'])
        hru = stat['interaction_stats'] # interaction histogram statistics
        u_ref = stat['energy'] # reference system energy

        uuu = beta*(np.sum(hru*par, axis=1) - u_ref)
        #print(u_ref[10], np.sum(hru*par, axis=1)[10])
        uuu -= np.mean(uuu)
        eee = np.exp(-uuu)

        # statistical distance from reference to model
        ge = -np.log(np.mean(eee))   # free energy difference (shifted)
        cb = np.mean(np.exp(-0.5*(uuu - ge))) # Bhattacharyya coefficient
        #sd2_ref += w*np.arccos(cb)**2/eee.shape[0]      # statistical distance
        #sd2_ref += w/(cb**2*eee.shape[0])      # statistical distance
        sd2_ref += w*(1-cb**2)/eee.shape[0]
        #print('s2', 1-cb**2, cb)

        # histogram reweighting factor
        eee /= np.sum(eee)

        # statistical distance for surface configuration histograms
        phst = targ['config_stats']     # target histogram
        rhst = stat['config_stats']     # reference histograms for each configuration
        #qhst = np.sum(rhst.T*eee, axis=1) # rescaled average reference histogram
        qhst = rhst.T.dot(eee)
        #qhst = np.mean(rhst, axis=0)#.T.dot(eee)
        #qhst_var = np.var(rhst.T*eee, axis=1) # variance of the rescaled reference histogram 
        #print('rhst.shape', rhst.shape, qhst_var.shape, qhst_var)
        #print('qhst', qhst)

        # Statistical distance contribution
        sd2 += w*np.arccos(np.sum(np.sqrt(qhst*phst)))**2
        #sd2 += w*np.arccos(np.sum(np.sqrt(qhst).dot(np.sqrt(phst))))**2
        #print(key, w, np.arccos(np.sum(np.sqrt(qhst*phst)))**2)
        #sd2_ref += w*np.sum(qhst_var)

    #print('---', sd2, sd2+sd2_ref, sd2, sd2_ref, eee.shape)
    return sd2  #, sd2_ref  # + sd2_ref


def get_chi2_two(key, targ, null, plot=True):
    """
    """
        
    ntype = targ.shape[0]
    
    for i in range(ntype):

        tt = np.array([sum([i*p for i, p in enumerate(targ[i,j,:])]) for j in range(ntype)])
        nn = np.array([sum([i*p for i, p in enumerate(null[i,j,:])]) for j in range(ntype)])
        
        n_nn = np.sum(nn)
        n_tt = np.sum(tt)
        k1 = np.sqrt(n_nn/n_tt)
        k2 = 1/k1

        chi2_stat = sum([(k1*t - k2*n)**2/(t + n) for t, n in zip(tt, nn)])

        df = len(tt) - 1
        print(i, tt, nn/np.sum(nn)*np.sum(tt), np.array(tt) - np.array(nn)/np.sum(nn)*np.sum(tt))
        #print(key, i, 'chi2', chi2_stat, df)
        p_value = chi2.sf(chi2_stat, df)
        print(k, i, 'p-value', p_value)

def get_s2_two(key, targ, null, plot=True):
    """
    """
        
    ntype = targ.shape[0]
    
    for i in range(ntype):

        tt = np.array([sum([i*p for i, p in enumerate(targ[i,j,:])]) for j in range(ntype)])
        nn = np.array([sum([i*p for i, p in enumerate(null[i,j,:])]) for j in range(ntype)])
        
        n_nn = np.sum(nn)
        n_tt = np.sum(tt)
        
        print(i, tt, nn/np.sum(nn)*np.sum(tt), np.array(tt) - np.array(nn)/np.sum(nn)*np.sum(tt))

        nn = nn/n_nn
        tt = tt/n_tt
        
        cb = sum([np.sqrt(t*n) for t, n in zip(tt, nn)])
        s2 = 4*n_nn*n_tt/(n_nn + n_tt)*np.arccos(cb)**2

        df = len(tt) - 1
        p_value = chi2.sf(s2, df)
        print(key, i, 'p-value', p_value)


def get_s2_two_old(key, targ, null, plot=True):
    """
    """
        
    ntype = targ.shape[0]
    
    for i in range(ntype):

        tt = np.array([sum([i*p for i, p in enumerate(targ[i,j,:])]) for j in range(ntype)])
        nn = np.array([sum([i*p for i, p in enumerate(null[i,j,:])]) for j in range(ntype)])
        
        n_nn = np.sum(nn)
        n_tt = np.sum(tt)
        
        print(i, tt, nn/np.sum(nn)*np.sum(tt), np.array(tt) - np.array(nn)/np.sum(nn)*np.sum(tt))

        nn = nn/n_nn
        tt = tt/n_tt
        
        cb = sum([np.sqrt(t*n) for t, n in zip(tt, nn)])
        s2 = 4*n_nn*n_tt/(n_nn + n_tt)*np.arccos(cb)**2

        df = len(tt) - 1
        p_value = chi2.sf(s2, df)
        print(key, i, 'p-value', p_value)


def data_lists(trjs, minval=200):
    """Converts reference trajectory data into lists of input matrices.
    Each list item represents one reference trajectory.
    """

    X_list = []
    ene_list = []
    beta_list = []
    hist_list = []
    pars_list = []
    
    for key, trj in trjs.items():
        X_list.append(np.array(trj['interaction_stats'][minval:]))
        beta_list.append(1/np.array(trj['temp'][minval:]))
        ene_list.append(np.array(trj['energy'][minval:]))
        hist_list.append(trj['knn'][minval:])
        pars_list.append(trj['ref_params'])
        
    return X_list, beta_list, pars_list, ene_list, hist_list

 
def make_target_matrices(targets):
    """Select only the relevant statistics and parameters.
    """

    k_list = sorted(targets.keys())

    hist_targ = []

    # cycle over KNNs
    for k in k_list:
        hist_k = targets[k]
        # cycle over particle i types
        ntype = len(hist_k)
        nn = []
        for i in range(ntype):
            nn.append(np.array([sum([i*p for i, p in enumerate(hist_k[i,j,:])]) for j in range(ntype)]))

        hist_targ.append(np.array(nn))  # append (ntype, ntype) matrix

    return np.array(hist_targ)


def make_reference_matrices(hist_list, X_list, pars_list, pars_select, knn_max=1):
    """Select only the relevant statistics and parameters.
    """

    X_input = []
    pars_input = []
    hist_input = []
    
    for X, pars, hists in zip(X_list, pars_list, hist_list):
        X_input.append(X[:, pars_select])
        pars_input.append(pars[pars_select])

        # cycle over samples
        hist_sample = []
        for hists_i in hists:
            # cycle over KNNs
            knn = []
            for hist_k in hists_i[:knn_max]:
                # cycle over particle i types
                ntype = len(hist_k)
                nn = []
                for i in range(ntype):
                    nn.append(np.array([sum([i*p for i, p in enumerate(hist_k[i,j,:])]) for j in range(ntype)]))

                knn.append(np.array(nn))  # append (ntype, ntype) matrix

            hist_sample.append(np.array(knn)) # apend (knn, ntype, ntype) matrix for sample i

        hist_input.append(np.array(hist_sample))
    
    return X_input, pars_input, hist_input

 
