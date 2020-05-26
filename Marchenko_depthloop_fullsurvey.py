# -*- coding: utf-8 -*-
"""
Created on Thu May 23 17:40:55 2019

@author: Claudia Haindl
"""

import sys

#import matplotlib.pyplot as plt
import numpy as np
import MarchenkoFunctions as MF

from scipy.special import hankel2

from pylops.basicoperators import Restriction
from pylops.waveeqprocessing import MDC, MDD
    
def Marchenko_depthloop_fullsurvey(zinvs,zendvs):
    #%%
    toff = 0.045        # direct arrival time shift
    nfmax = 550         # max frequency for MDC (#samples)
    nfft = 2 ** 11
    
    jr = 3              # subsampling in r
    jt = 5
    
    path0='./datasets_new/'
    
    # subsurface array
    xinvs=600         # receiver array initial point in x
    xendvs=2400       # receiver array last point in x
    #zinvs=900         # receiver array initial point in x
    #zendvs=1500       # receiver array last point in x
    dvsx=20           # receiver array sampling in x
    dvsz=5            # receiver array sampling in z
    
    # line of subsurface points for virtual sources
    vsx = np.arange(xinvs,xendvs+dvsx,dvsx)
    nvsx = vsx.shape[0]
    vsz = np.arange(zinvs,zendvs+dvsz,dvsz)
    nvsz = vsz.shape[0]
    
    # geometry
    nz = 401
    oz = 0
    dz = 4
    z = np.arange(oz, oz + nz*dz, dz)
    
    nx=751
    ox=0
    dx=4
    x = np.arange(ox, ox + nx*dx, dx)
    
    # time axis
    ot=0
    nt=1081
    dt=0.0025
    t = np.arange(ot,ot+nt*dt,dt)
    
    # %% loading everything
    
    print('Loading reflection data...')
    
    # Receivers
    r = np.loadtxt(path0 + 'r.dat', delimiter=',')
    nr = r.shape[1]
    
    # Sources
    s = np.loadtxt(path0 + 's.dat', delimiter=',')
    ns = s.shape[1]
    ds = s[0,1]-s[0,0]
     
    # data
    R_1=np.zeros((nt, ns, nr),'f')
    R_2=np.zeros((nt, ns, nr),'f')
    for isrc in range(ns-1):
        is_ = isrc*jr
        R_1[:,:,isrc]=np.loadtxt(path0 + 'dat1_' + str(is_) + '.dat', delimiter=',')
        R_2[:,:,isrc]=np.loadtxt(path0 + 'dat2_' + str(is_) + '.dat', delimiter=',')
    
    R_1 = 2 * np.swapaxes(R_1,0,2)
    R_2 = 2 * np.swapaxes(R_2,0,2)
    
    # %% Convolution operators
    
    print('Creating MDC operators...')
    # Add negative time
    Rtwosided_1 = np.concatenate((np.zeros((nr, ns, nt-1)), R_1), axis=-1)
    R1twosided_1 = np.concatenate((np.flip(R_1, axis=-1), 
                                np.zeros((nr, ns, nt-1))), axis=-1)
    
    Rtwosided_fft_1 = np.fft.rfft(Rtwosided_1, 2*nt-1, axis=-1)/np.sqrt(2*nt-1)
    Rtwosided_fft_1 = Rtwosided_fft_1[...,:nfmax]
    R1twosided_fft_1 = np.fft.rfft(R1twosided_1, 2*nt-1, axis=-1)/np.sqrt(2*nt-1)
    R1twosided_fft_1 = R1twosided_fft_1[...,:nfmax]
    
    Rtwosided_2 = np.concatenate((np.zeros((nr, ns, nt-1)), R_2), axis=-1)
    R1twosided_2 = np.concatenate((np.flip(R_2, axis=-1), 
                                np.zeros((nr, ns, nt-1))), axis=-1)
    
    Rtwosided_fft_2 = np.fft.rfft(Rtwosided_2, 2*nt-1, axis=-1)/np.sqrt(2*nt-1)
    Rtwosided_fft_2 = Rtwosided_fft_2[...,:nfmax]
    R1twosided_fft_2 = np.fft.rfft(R1twosided_2, 2*nt-1, axis=-1)/np.sqrt(2*nt-1)
    R1twosided_fft_2 = R1twosided_fft_2[...,:nfmax]
    
    # Operators
    Rop1 = MDC(Rtwosided_fft_1, nt=2*nt-1, nv=1, dt=dt, dr=ds, twosided=True, dtype='complex64')
    R1op1 = MDC(R1twosided_fft_1, nt=2*nt-1, nv=1, dt=dt, dr=ds, twosided=True, dtype='complex64')
    
    Rop2 = MDC(Rtwosided_fft_2, nt=2*nt-1, nv=1, dt=dt, dr=ds, twosided=True, dtype='complex64')
    R1op2 = MDC(R1twosided_fft_2, nt=2*nt-1, nv=1, dt=dt, dr=ds, twosided=True, dtype='complex64')
    
    # %% wavelet and traveltime
    
    print('Loading direct wave...')
    wav = np.loadtxt(path0 + 'wav.dat', delimiter=',')
    
    trav_eik = np.loadtxt(path0 + 'trav.dat', delimiter=',')
    trav_eik = np.reshape(trav_eik,(nz,ns,nx))
    
    # %% the loop
    I1=np.zeros((nvsx,nvsz))
    I2=np.zeros((nvsx,nvsz))
    
    print('Entering loop...')
    for iz in range(nvsz):
        #%%
        FUP1=np.zeros(shape=(nr,nvsx,nt*2-1))
        FDOWN1=np.zeros(shape=(nr,nvsx,nt*2-1))
        PUP1=np.zeros(shape=(nr,nvsx,nt))
        PDOWN1=np.zeros(shape=(nr,nvsx,nt))
        PUP2=np.zeros(shape=(nr,nvsx,nt))
        PDOWN2=np.zeros(shape=(nr,nvsx,nt))
        
        for ix in range(nvsx):
            
            s = '############ Point ' + str(ix+1) + ' of ' + str(nvsx) + ', line ' + str(iz+1) + ' of ' + str(nvsz) + ' (z = ' + str(vsz[iz]) + ', x = ' + str(vsx[ix]) + ') ... ' + str(100 * (iz*nvsx + ix + 1)/(nvsx*nvsz)) + '%'
            print(s)
            direct = trav_eik[MF.find_closest(vsz[iz],z),:,MF.find_closest(vsx[ix],x)]
            W = np.abs(np.fft.rfft(wav, nfft)) * dt
            f = 2 * np.pi * np.arange(nfft) / (dt * nfft)
            g0VS = np.zeros((nfft, nr), dtype=np.complex128)
            for it in range(len(W)):
                g0VS[it] = W[it] * 1j * f[it] * (-1j) * hankel2(0, f[it] * direct + 1e-10) / 4
            g0VS = np.fft.irfft(g0VS, nfft, axis=0) / dt
            g0VS = np.real(g0VS[:nt])
            
            f1_1_minus, f1_1_plus, f1_2_minus, f1_2_plus, g_1_minus, g_1_plus, g_2_minus, g_2_plus = MF.machenkoFocSourceLoop_JointLsqr_fullsurvey(direct,toff,g0VS,Rop1,R1op1,Rop2,R1op2,t)
    
            FUP1[:,ix,:]=f1_1_minus
            FDOWN1[:,ix,:]=f1_1_plus
            PUP1[:,ix,:]=g_1_minus[:,nt-1:]
            PDOWN1[:,ix,:]=g_1_plus[:,nt-1:]
            PUP2[:,ix,:]=g_2_minus[:,nt-1:]
            PDOWN2[:,ix,:]=g_2_plus[:,nt-1:]
            
        jt=2
        redatumed1 = MDD(PDOWN1[:,:,::jt], PUP1[:,:,::jt], dt = jt*dt, dr = dvsx, twosided=True, adjoint=False, psf=False, dtype='complex64', dottest=False, **dict(iter_lim=20, show=0))
        redatumed2 = MDD(PDOWN2[:,:,::jt], PUP2[:,:,::jt], dt = jt*dt, dr = dvsx, twosided=True, adjoint=False, psf=False, dtype='complex64', dottest=False, **dict(iter_lim=20, show=0))
        
        #np.savetxt('datasets_new/Line1_' + str(vsz[iz]) + '.dat', np.diag(redatumed1[:,:,(nt+1)/jt-1]), delimiter=',')
        #np.savetxt('datasets_new/Line2_' + str(vsz[iz]) + '.dat', np.diag(redatumed2[:,:,(nt+1)/jt-1]), delimiter=',')
        #%%
        #I1[:,iz]=np.diag(redatumed1[:,:,(nt+1)/jt-1])
        #I2[:,iz]=np.diag(redatumed2[:,:,(nt+1)/jt-1])
     
    #np.savetxt('datasets_new/Image1' + str(zinvs) + '-' + str(zendvs) + '.dat', I1, delimiter=',')
    #np.savetxt('datasets_new/Image2' + str(zinvs) + '-' + str(zendvs) + '.dat', I2, delimiter=',')
    
    return FUP1, FDOWN1, PUP1, PDOWN1, PUP2, PDOWN2, redatumed1, redatumed2

if __name__ == "__main__":
    a = int(sys.argv[1])
    b = int(sys.argv[2])
    Marchenko_depthloop_fullsurvey(a, b)
