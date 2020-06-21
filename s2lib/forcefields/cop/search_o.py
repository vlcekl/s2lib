#!/usr/bin/python
#
# File name:   harmdet.py
# Date:        2012/01/26 17:49
# Author:      Lukas Vlcek
#
# Description: 
#

import sys
import string as s
import re
import numpy as np
from scipy.optimize import fmin, anneal

def f_ss(p, d_u, d_f):
    """Free energy perturbation"""
    beta = 1000.0/(8.314472*temp)
    nmx = len(d_u[0][:])
    fn = 1.0/float(nmx)

    dall_u = [hee, sa, sc, qr, k1, k2, a1, a2]
    dall_f = [fr, gar, gcr, gqr, gkkr, gkrr, gaar, garr]

    cc = 4.0*p[1]*p[0]**6
    aa = cc*p[0]**6
    qq = p[2]
    kk = p[3]
    r0 = p[4]
    ka = p[5]
    r0a = p[6]

    #uuu = aa*sa - cc*sc + qq*qq*qr + kk*(k2-2.0*k1*r0+float(2*nw)*r0**2) + ka*(a2-2.0*a1*r0a+float(nw)*r0a**2)
    uuu = aa*d_u[1] - cc*d_u[2] + qq*qq*d_u[3]
    uuu = uuu + kk*(d_u[5] - 2.0*d_u[4]*r0  + float(2*nw)*r0**2)
    uuu = uuu + ka*(d_u[7] - 2.0*d_u[6]*r0a + float(nw)*r0a**2)
    #print 'p', p[:], uuu[0], d_u[0][0], aa, cc, qq, kk, r0, ka, r0a
    #print 'u', d_u[1][0], d_u[2][0], d_u[3][0], d_u[4][0], d_u[5][0], d_u[6][0], d_u[7][0]
    #print 'pu', aa*d_u[1][0], cc*d_u[2][0], qq*qq*d_u[3][0], \
    #        kk*(d_u[5][0] - 2.0*d_u[4][0]*r0  + float(2*nw)*r0**2), \
    #        ka*(d_u[7][0] - 2.0*d_u[6][0]*r0a  + float(nw)*r0a**2) 
    #exit()
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
        fff[:] = fff[:] + kk*(d_f[4][i,:] - r0*d_f[5][i,:])
        fff[:] = fff[:] + ka*(d_f[6][i,:] - r0a*d_f[7][i,:])
        #print 'fff', fff[0], d_f[0][i,0]
        #print 'fa', aa*d_f[1][i,0], cc*d_f[2][i,0], qq*qq*d_f[3][i,0]
        #print 'fr', kk*(d_f[4][i,0] - r0*d_f[5][i,0]), ka*(d_f[6][i,0] - r0a*d_f[7][i,0])
        #print 'fs', aa*d_f[1][i,0] - cc*d_f[2][i,0] + qq*qq*d_f[3][i,0] + kk*(d_f[4][i,0] - r0*d_f[5][i,0]) + ka*(d_f[6][i,0] - r0a*d_f[7][i,0])
        fqave = fqave + eee[i]*np.sum(np.exp(betad*fff[:]))*finf
        fhave = fhave + eeh[i]*np.sum(np.exp(0.5*betad*(fff[:]+d_f[0][i,:])))*finf
        fmm = fmm + np.sum((fff[:] - d_f[0][i,:])**2)*finf

    fqave = fqave*fn
    fhave = fhave*fn
    fmm = fmm*fn
    #print fqave, fhave, fmm

    gef = -np.log(fqave/fpave)
    cb = fhave/(fqave*fpave)**0.5
    if cb > 1:
        cb = 1
    ds2f = np.arccos(cb)**2
    #print 'loss', gef, cb, ds2f

    eee = np.exp(0.5*(ge - uuu))
    cb = np.sum(eee)*fn
    ds2 = np.arccos(cb)**2
    
    #dhs = np.arccos(np.sum(eee)*fn)**2

    #print p[:], ds2, ds2f, fmm

    print 'se', se
    return se

def f_dhs(p, d_u, d_f, p0):
    """Free energy perturbation"""
    beta = 1000.0/(8.314472*temp)
    nmx = len(d_u[0][:])
    fn = 1.0/float(nmx)
    #print 'temp', temp, beta

    dall_u = [hee, sa, sc, qr, k1, k2, a1, a2]
    dall_f = [fr, gar, gcr, gqr, gkkr, gkrr, gaar, garr]

    cc = 4.0*p[1]*p[0]**6
    aa = cc*p[0]**6
    qq = p[2]
    #p[3] = p0[3]
    #p[4] = p0[4]
    #p[5] = p0[5]
    #p[6] = p0[6]
    kk = p[3]
    r0 = p[4]
    ka = p[5]
    r0a = p[6]

    #uuu = aa*sa - cc*sc + qq*qq*qr + kk*(k2-2.0*k1*r0+float(2*nw)*r0**2) + ka*(a2-2.0*a1*r0a+float(nw)*r0a**2)
    uuu = aa*d_u[1] - cc*d_u[2] + qq*qq*d_u[3]
    uuu = uuu + kk*(d_u[5] - 2.0*d_u[4]*r0  + float(2*nw)*r0**2)
    uuu = uuu + ka*(d_u[7] - 2.0*d_u[6]*r0a + float(nw)*r0a**2)
    #print 'p', p[:], uuu[0], d_u[0][0], aa, cc, qq, kk, r0, ka, r0a
    #print 'u', d_u[1][0], d_u[2][0], d_u[3][0], d_u[4][0], d_u[5][0], d_u[6][0], d_u[7][0]
    #print 'pu', aa*d_u[1][0], cc*d_u[2][0], qq*qq*d_u[3][0], \
    #        kk*(d_u[5][0] - 2.0*d_u[4][0]*r0  + float(2*nw)*r0**2), \
    #        ka*(d_u[7][0] - 2.0*d_u[6][0]*r0a  + float(nw)*r0a**2) 
    #exit()
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
        fff[:] = fff[:] + kk*(d_f[4][i,:] - r0*d_f[5][i,:])
        fff[:] = fff[:] + ka*(d_f[6][i,:] - r0a*d_f[7][i,:])
        #print 'fff', fff[0], d_f[0][i,0]
        #print 'fa', aa*d_f[1][i,0], cc*d_f[2][i,0], qq*qq*d_f[3][i,0]
        #print 'fr', kk*(d_f[4][i,0] - r0*d_f[5][i,0]), ka*(d_f[6][i,0] - r0a*d_f[7][i,0])
        #print 'fs', aa*d_f[1][i,0] - cc*d_f[2][i,0] + qq*qq*d_f[3][i,0] + kk*(d_f[4][i,0] - r0*d_f[5][i,0]) + ka*(d_f[6][i,0] - r0a*d_f[7][i,0])
        fqave = fqave + eee[i]*np.sum(np.exp(betad*fff[:]))*finf
        fhave = fhave + eeh[i]*np.sum(np.exp(0.5*betad*(fff[:]+d_f[0][i,:])))*finf
        fmm = fmm + np.sum((fff[:] - d_f[0][i,:])**2)*finf

    fqave = fqave*fn
    fhave = fhave*fn
    fmm = fmm*fn
    #print fqave, fhave, fmm

    gef = -np.log(fqave/fpave)
    cb = fhave/(fqave*fpave)**0.5
    if cb > 1:
        cb = 1
    ds2f = np.arccos(cb)**2
    #print 'loss', gef, cb, ds2f

    eee = np.exp(0.5*(ge - uuu))
    cb = np.sum(eee)*fn
    ds2 = np.arccos(cb)**2
    
    #dhs = np.arccos(np.sum(eee)*fn)**2

    #print p[:], ds2, ds2f, fmm

    #return ds2
    #return ds2f
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
    k1 = dat_u[:,6]
    k2 = dat_u[:,7]
    a1 = dat_u[:,8]
    a2 = dat_u[:,9]

    print 'dat_f.shape', dat_f.shape

    fr = np.zeros((nmax,nf), dtype=float)
    gar = np.zeros((nmax,nf), dtype=float)
    gcr = np.zeros((nmax,nf), dtype=float)
    gqr = np.zeros((nmax,nf), dtype=float)
    gkkr = np.zeros((nmax,nf), dtype=float)
    gkrr = np.zeros((nmax,nf), dtype=float)
    gaar = np.zeros((nmax,nf), dtype=float)
    garr = np.zeros((nmax,nf), dtype=float)
    
    for i in range(nmax):
        j = 0
        for ip in range(nat):
            fr[i,j] = dat_f[i,ip,1]
            gar[i,j] = dat_f[i,ip,4]
            gcr[i,j] = dat_f[i,ip,5]
            gqr[i,j] = dat_f[i,ip,10]
            gkkr[i,j] = dat_f[i,ip,13]
            gkrr[i,j] = dat_f[i,ip,14]
            gaar[i,j] = dat_f[i,ip,19]
            garr[i,j] = dat_f[i,ip,20]
            j = j + 1
            fr[i,j] = -dat_f[i,ip,1]
            gar[i,j] = -dat_f[i,ip,4]
            gcr[i,j] = -dat_f[i,ip,5]
            gqr[i,j] = -dat_f[i,ip,10]
            gkkr[i,j] = -dat_f[i,ip,13]
            gkrr[i,j] = -dat_f[i,ip,14]
            gaar[i,j] = -dat_f[i,ip,19]
            garr[i,j] = -dat_f[i,ip,20]
            j = j + 1
            fr[i,j] = dat_f[i,ip,2]
            gar[i,j] = dat_f[i,ip,6]
            gcr[i,j] = dat_f[i,ip,7]
            gqr[i,j] = dat_f[i,ip,11]
            gkkr[i,j] = dat_f[i,ip,15]
            gkrr[i,j] = dat_f[i,ip,16]
            gaar[i,j] = dat_f[i,ip,21]
            garr[i,j] = dat_f[i,ip,22]
            j = j + 1
            fr[i,j] = -dat_f[i,ip,2]
            gar[i,j] = -dat_f[i,ip,6]
            gcr[i,j] = -dat_f[i,ip,7]
            gqr[i,j] = -dat_f[i,ip,11]
            gkkr[i,j] = -dat_f[i,ip,15]
            gkrr[i,j] = -dat_f[i,ip,16]
            gaar[i,j] = -dat_f[i,ip,21]
            garr[i,j] = -dat_f[i,ip,22]
            j = j + 1
            fr[i,j] = dat_f[i,ip,3]
            gar[i,j] = dat_f[i,ip,8]
            gcr[i,j] = dat_f[i,ip,9]
            gqr[i,j] = dat_f[i,ip,12]
            gkkr[i,j] = dat_f[i,ip,17]
            gkrr[i,j] = dat_f[i,ip,18]
            gaar[i,j] = dat_f[i,ip,23]
            garr[i,j] = dat_f[i,ip,24]
            j = j + 1
            fr[i,j] = -dat_f[i,ip,3]
            gar[i,j] = -dat_f[i,ip,8]
            gcr[i,j] = -dat_f[i,ip,9]
            gqr[i,j] = -dat_f[i,ip,12]
            gkkr[i,j] = -dat_f[i,ip,17]
            gkrr[i,j] = -dat_f[i,ip,18]
            gaar[i,j] = -dat_f[i,ip,23]
            garr[i,j] = -dat_f[i,ip,24]
            j = j + 1

        fr[i,j] = 0.0
        gar[i,j] = 0.0
        gcr[i,j] = 0.0
        gqr[i,j] = 0.0
        gkkr[i,j] = 0.0
        gkrr[i,j] = 0.0
        gaar[i,j] = 0.0
        garr[i,j] = 0.0


    # precompute reference system force averages
    fpave = 0.0
    for i in range(nmax):
        fpave = fpave + np.sum(np.exp(betad*fr[i,1:nf]))*finf

    fpave = fpave*finmax

    dall_u = [hee, sa, sc, qr, k1, k2, a1, a2]
    dall_f = [fr, gar, gcr, gqr, gkkr, gkrr, gaar, garr]

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
        kkk = para[j,3]*4.184
        rr0 = para[j,4]
        kka = para[j,5]*4.184
        rr0a = para[j,6]*3.1415926536/180.0
        par_in = [sig, eps, qqq, kkk, rr0, kka, rr0a]
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

