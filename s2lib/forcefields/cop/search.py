import sys
import string as s
import re
import numpy as np
from scipy.optimize import fmin

def f_dhs(p, d_u, d_f, p0):
    """Free energy perturbation"""
    beta = 1000.0/(8.314472*temp)
    nmx = len(d_u[0][:])
    fn = 1.0/float(nmx)
    #print 'temp', temp, beta

    dall_u = [hee, sa, sc, qr]
    dall_f = [fr, gar, gcr, gqr]

    cc = 4.0*p[1]*p[0]**6
    aa = cc*p[0]**6
    qq = p[2]

    #uuu = aa*sa - cc*sc + qq*qq*qr + kk*(k2-2.0*k1*r0+float(2*nw)*r0**2) + ka*(a2-2.0*a1*r0a+float(nw)*r0a**2)
    uuu = aa*d_u[1] - cc*d_u[2] + qq*qq*d_u[3]
    uuu = beta*(uuu - d_u[0])

    uave = sum(uuu)*fn
    uuu = uuu - uave

    eee = np.exp(-uuu)

    ge = -np.log(np.sum(eee)*fn)
    he = np.sum((uuu + beta*d_u[0])*eee)/np.sum(eee)
    he = he - beta*np.sum(hee)*fn
    se = he - ge

    eeh = np.exp(-0.5*uuu)
    fqave = 0.0
    fhave = 0.0
    fmm = 0.0
    for i in range(nmx):
        fff[:] = aa*d_f[1][i,:] - cc*d_f[2][i,:] + qq*qq*d_f[3][i,:]
        fqave = fqave + eee[i]*np.sum(np.exp(betad*fff[:]))*finf
        fhave = fhave + eeh[i]*np.sum(np.exp(0.5*betad*(fff[:]+d_f[0][i,:])))*finf
        fmm = fmm + np.sum((fff[:] - d_f[0][i,:])**2)*finf

    fqave = fqave*fn
    fhave = fhave*fn
    fmm = fmm*fn

    gef = -np.log(fqave/fpave)
    cb = fhave/(fqave*fpave)**0.5
    if cb > 1:
        cb = 1
    ds2f = np.arccos(cb)**2
    #print 'loss', gef, cb, ds2f

    eee = np.exp(0.5*(ge - uuu))
    cb = np.sum(eee)*fn
    ds2 = np.arccos(cb)**2
    
    return fmm

if __name__ == "__main__":

    # load data
    fi = open(sys.argv[1], 'r')
    line = fi.readline()
    sarr = re.findall('\S+', line)
    nmax = int(sarr[0])
    finmax = 1.0/float(nmax)
    temp = float(sarr[1])
    beta = 1.0/(8.314472*temp/1000.0)
    ddel = 0.01
    ddel2 = 0.001
    betad = beta*ddel
    betax = beta*ddel

    nw = 125
    nat = 3*nw
    nf = 6*nat + 1
    finf = 1.0/float(nf)

    hee = np.zeros((nmax), dtype=float)
    eee = np.zeros((nmax), dtype=float)
    eeh = np.zeros((nmax), dtype=float)
    uuu = np.zeros((nmax), dtype=float)
    fff = np.zeros((nf), dtype=float)

    dat_u = []
    dat_f = []
    for i in range(nmax):
        line = fi.readline()
        if not line:
            break
        # ener
        dat_u.append(map(float, re.findall('\S+', line)))
        dat_f.append([])
        for ip in range(nat):
            dat_f[-1].append(map(float, re.findall('\S+', fi.readline())))

    fi.close()
    nd = len(dat_f[0][0])
    print 'nd', nd, len(dat_f[-1][-1])
        
    dat_u = np.array(dat_u)
    dat_f = np.array(dat_f)

    print 'dat_u.shape', dat_u.shape
    hee = dat_u[:,1]
    sa = dat_u[:,3]
    sc = dat_u[:,4]
    qr = dat_u[:,5]

    print 'dat_f.shape', dat_f.shape

    fr = np.zeros((nmax,nf), dtype=float)
    gar = np.zeros((nmax,nf), dtype=float)
    gcr = np.zeros((nmax,nf), dtype=float)
    gqr = np.zeros((nmax,nf), dtype=float)
    
    for i in range(nmax):
        j = 0
        for ip in range(nat):
            fr[i,j] = dat_f[i,ip,1]
            gar[i,j] = dat_f[i,ip,4]
            gcr[i,j] = dat_f[i,ip,5]
            gqr[i,j] = dat_f[i,ip,10]
            j = j + 1
            fr[i,j] = -dat_f[i,ip,1]
            gar[i,j] = -dat_f[i,ip,4]
            gcr[i,j] = -dat_f[i,ip,5]
            gqr[i,j] = -dat_f[i,ip,10]
            j = j + 1
            fr[i,j] = dat_f[i,ip,2]
            gar[i,j] = dat_f[i,ip,6]
            gcr[i,j] = dat_f[i,ip,7]
            gqr[i,j] = dat_f[i,ip,11]
            j = j + 1
            fr[i,j] = -dat_f[i,ip,2]
            gar[i,j] = -dat_f[i,ip,6]
            gcr[i,j] = -dat_f[i,ip,7]
            gqr[i,j] = -dat_f[i,ip,11]
            j = j + 1
            fr[i,j] = dat_f[i,ip,3]
            gar[i,j] = dat_f[i,ip,8]
            gcr[i,j] = dat_f[i,ip,9]
            gqr[i,j] = dat_f[i,ip,12]
            j = j + 1
            fr[i,j] = -dat_f[i,ip,3]
            gar[i,j] = -dat_f[i,ip,8]
            gcr[i,j] = -dat_f[i,ip,9]
            gqr[i,j] = -dat_f[i,ip,12]
            j = j + 1

        fr[i,j] = 0.0
        gar[i,j] = 0.0
        gcr[i,j] = 0.0
        gqr[i,j] = 0.0


    # precompute reference system force averages
    fpave = 0.0
    for i in range(nmax):
        fpave = fpave + np.sum(np.exp(betad*fr[i,1:nf]))*finf

    fpave = fpave*finmax

    dall_u = [hee, sa, sc, qr]
    dall_f = [fr, gar, gcr, gqr]

    # load params
    fi = open(sys.argv[2], 'r')
    para = []
    while 1:
        line = fi.readline()
        if not line: break
        para.append(np.array(map(float, re.findall('\S+', line))))

    fi.close()
    para = np.array(para)
    print 'para.shape', para.shape, para.shape[0]

    for j in range(para.shape[0]):
        # continue optimization from the grid minima
        sig = para[j,0]
        eps = para[j,1]*4.184
        qqq = para[j,2]
        par_in = [sig, eps, qqq]
        #par_in = para[j,:]
        par_0 = par_in[:]

        output = fmin(f_dhs, par_in, args=(dall_u, dall_f, par_0), maxiter=100000, maxfun=100000, disp=0, full_output=1,ftol=1e-6)
        xopt = output[0] 
        soi = xopt[0]
        eoi = xopt[1]
        ofunc = output[1]
        print '# fmin ', xopt[:7]
        print '# all ', f_dhs(xopt[:7], dall_u, dall_f, par_0)
        print '# ent ', f_ss(xopt[:7], dall_u, dall_f)

