# -*- coding: utf-8 -*-
"""
Created on Thu May 23 17:40:55 2019

@author: Claudia Haindl
"""

import sys

import numpy as np
import MarchenkoFunctions as MF

from scipy.special import hankel2

from pylops.basicoperators        import Restriction
from pylops.waveeqprocessing      import MDC, MDD
from pylops.signalprocessing      import Radon2D, Sliding2D
from pylops.basicoperators        import Diagonal, BlockDiag, VStack, HStack, Zero, Transpose, Identity
from pylops.optimization.sparsity import SPGL1, FISTA

#import Operators as OP
import multiprocessing as MP


def Marchenko_depthloop_JointRadon(zinvs, zendvs, nproc):
    #%%
    
    toff = 0.045        # direct arrival time shift
    nfmax = 550         # max frequency for MDC (#samples)
    nfft = 2 ** 11
    
    jr = 3              # subsampling in r
    
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
    vsz = np.arange(zinvs,zendvs+dvsz,dvsz) 
    
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
    
    # restriction operators
    iava1 = np.loadtxt(path0 + 'temp_select_rfrac70.dat', delimiter=',', dtype=int) - 1
    iava2 = np.loadtxt(path0 + 'temp_select_rfrac50.dat', delimiter=',', dtype=int) - 1
    Restrop1 = Restriction(ns*(2*nt-1), iava1, dims=(ns, 2*nt-1), dir=0, dtype='float64')
    Restrop2 = Restriction(ns*(2*nt-1), iava2, dims=(ns, 2*nt-1), dir=0, dtype='float64')
    
    # data
    R_1=np.zeros((nt, ns, nr),'f')
    R_2=np.zeros((nt, ns, nr),'f')
    for isrc in range(ns-1):
        is_ = isrc*jr
        R_1[:,:,isrc]=np.loadtxt(path0 + 'dat1_' + str(is_) + '.dat', delimiter=',')
        R_2[:,:,isrc]=np.loadtxt(path0 + 'dat2_' + str(is_) + '.dat', delimiter=',')
    
    R_1 = 2 * np.swapaxes(R_1,0,2)
    R_2 = 2 * np.swapaxes(R_2,0,2)
    
    wav = np.loadtxt(path0 + 'wav.dat', delimiter=',')
    W = np.abs(np.fft.rfft(wav, nfft)) * dt
    
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
    Rop1 = MDC(Rtwosided_fft_1[iava1], nt=2*nt-1, nv=1, dt=dt, dr=ds, twosided=True, dtype='complex64')
    R1op1 = MDC(R1twosided_fft_1[iava1], nt=2*nt-1, nv=1, dt=dt, dr=ds, twosided=True, dtype='complex64')
    
    Rop2 = MDC(Rtwosided_fft_2[iava2], nt=2*nt-1, nv=1, dt=dt, dr=ds, twosided=True, dtype='complex64')
    R1op2 = MDC(R1twosided_fft_2[iava2], nt=2*nt-1, nv=1, dt=dt, dr=ds, twosided=True, dtype='complex64')
    
    del wav, R_1, R_2, Rtwosided_1, Rtwosided_2, R1twosided_1, R1twosided_2, Rtwosided_fft_1, Rtwosided_fft_2, R1twosided_fft_1, R1twosided_fft_2
    
    # Radon transform
    nwin=35
    nwins=7
    nover=10
    npx=101
    pxmax = 1e-3
    px = np.linspace(-pxmax, pxmax, npx)
    
    t2=np.concatenate([-t[::-1], t[1:]])
    nt2=t2.shape[0]
    
    dimsd = (nr, nt2)
    dimss = (nwins*npx, dimsd[1])
    
    # tranpose operator
    #Top = Transpose((nt2, nr), axes=(1, 0), dtype=np.float64)
    
    # sliding window radon with overlap
    RadOp = Radon2D(t2, np.linspace(-ds*nwin//2, ds*nwin//2, nwin), px, centeredh=True, 
                    kind='linear', engine='numba')
    Slidop = Sliding2D(RadOp, dimss, dimsd, nwin, nover, tapertype='cosine', design=False)
    #Sparseop = BlockDiag([Top.H*Slidop, Top.H*Slidop, Top.H*Slidop, Top.H*Slidop, Top.H*Slidop, Top.H*Slidop])
    Sparseop = BlockDiag([Slidop, Slidop, Slidop, Slidop, Slidop, Slidop])
    
    #del nwin, nwins, nover, npx, pxmax, px, t2, nt2, dimsd, dimss, Top, RadOp, Slidop
    # %% the loop
    
    trav_eik = np.loadtxt(path0 + 'trav.dat', delimiter=',')
    trav_eik = np.reshape(trav_eik,(nz,ns,nx))
   
    print('Entering loop...')
    z_steps = np.arange(zinvs, zendvs + 1, dvsz * nproc)
    for z_step in z_steps:
        for i in range(nproc):
            z_current = z_step + dvsz * i
            if (z_current <= zendvs):
                FUP1, FDOWN1, PUP1, PDOWN1, PUP2, PDOWN2, redatumed1, redatumed2 = focusing_wrapper(toff, W, iava1, iava2, Rop1, Rop2, R1op1, R1op2, Restrop1, Restrop2, Sparseop, vsx, vsz, x, z, z_current, nt, dt, nfft, nr, ds, dvsx, trav_eik)
              
        
    return FUP1, FDOWN1, PUP1, PDOWN1, PUP2, PDOWN2, redatumed1, redatumed2
     
    
def focusing_wrapper(toff, W, iava1, iava2, Rop1, Rop2, R1op1, R1op2, Restrop1, Restrop2, Sparseop, vsx, vsz, x, z, z_current, nt, dt, nfft, nr, ds, dvsx, trav_eik):
    
    from scipy.signal import filtfilt
    
    nava1=iava1.shape[0]
    nava2=iava2.shape[0]
    nvsx = vsx.shape[0]
    
    FUP1=np.zeros(shape=(nr,nvsx,nt*2-1))
    FDOWN1=np.zeros(shape=(nr,nvsx,nt*2-1))
    PUP1=np.zeros(shape=(nava1,nvsx,nt))
    PDOWN1=np.zeros(shape=(nava1,nvsx,nt))
    PUP2=np.zeros(shape=(nava2,nvsx,nt))
    PDOWN2=np.zeros(shape=(nava2,nvsx,nt))
         
    for ix in range(nvsx):
        #%%
        s = '####### Point ' + str(ix+1) + ' of ' + str(nvsx) + ' of current line (z = ' + str(z_current) + ', x = ' + str(vsx[ix]) + ')'
        print(s)
        #direct = np.loadtxt('datasets_new/Traveltimes/trav_x' + str(vsx[ix]) + '_z' + str(z[MF.find_closest(z_current,z)]) + '.dat', delimiter=',')
        direct = trav_eik[MF.find_closest(z_current,z),:,MF.find_closest(vsx[ix],x)]

        f = 2 * np.pi * np.arange(nfft) / (dt * nfft)
        g0VS = np.zeros((nfft, nr), dtype=np.complex128)
        for it in range(len(W)):
            g0VS[it] = W[it] * f[it] * hankel2(0, f[it] * direct + 1e-10) / 4
        g0VS = np.fft.irfft(g0VS, nfft, axis=0) / dt
        g0VS = np.real(g0VS[:nt])
        
        # window
        nsmooth=10
        nr=direct.shape[0]
        nsava1=iava1.shape[0]
        nsava2=iava2.shape[0]
        
        directVS_off = direct - toff
        idirectVS_off = np.round(directVS_off/dt).astype(np.int)
        w = np.zeros((nr, nt))
        wi = np.ones((nr, nt))
        for ir in range(nr-1):
            w[ir, :idirectVS_off[ir]]=1   
        wi = wi - w
             
        w = np.hstack((np.fliplr(w), w[:, 1:]))
        wi = np.hstack((np.fliplr(wi), wi[:, 1:]))
        
        if nsmooth>0:
            smooth=np.ones(nsmooth)/nsmooth
            w  = filtfilt(smooth, 1, w)
            wi  = filtfilt(smooth, 1, wi)
            
        # Input focusing function
        fd_plus =  np.concatenate((np.fliplr(g0VS.T), np.zeros((nr, nt-1))), axis=-1)
        
        Wop = Diagonal(w.flatten())
        WSop1 = Diagonal(w[iava1].flatten())
        WSop2 = Diagonal(w[iava2].flatten())
        WiSop1 = Diagonal(wi[iava1].flatten())
        WiSop2 = Diagonal(wi[iava2].flatten())
        
        
        Mop1 = VStack([HStack([Restrop1, -1*WSop1*Rop1]),
                       HStack([-1*WSop1*R1op1, Restrop1])])*BlockDiag([Wop, Wop])
        Mop2 = VStack([HStack([Restrop2, -1*WSop2*Rop2]),
                       HStack([-1*WSop2*R1op2, Restrop2])])*BlockDiag([Wop, Wop])
        Mop = VStack([HStack([Identity(2*(2*nt-1)*nr), Identity(2*(2*nt-1)*nr), Zero(2*(2*nt-1)*nr)]),
                      HStack([Identity(2*(2*nt-1)*nr), Zero(2*(2*nt-1)*nr), Identity(2*(2*nt-1)*nr)])])
        Mop = BlockDiag([Mop1, Mop2]) * Mop

        Mop_radon = Mop * Sparseop
        
        Gop1 = VStack([HStack([Restrop1, -1*Rop1]),
                       HStack([-1*R1op1, Restrop1])])
        Gop2 = VStack([HStack([Restrop2, -1*Rop2]),
                       HStack([-1*R1op2, Restrop2])])
        
        # data
        d1 = WSop1*Rop1*fd_plus.flatten()
        d1 = np.concatenate((d1.reshape(nsava1, 2*nt-1), np.zeros((nsava1, 2*nt-1))))
        d2 = WSop2*Rop2*fd_plus.flatten()
        d2 = np.concatenate((d2.reshape(nsava2, 2*nt-1), np.zeros((nsava2, 2*nt-1))))
        
        d = np.concatenate((d1, d2))
        
        # inversion
        #comb_f = SPGL1(Mop_radon, d.flatten(), sigma=1e-1, iter_lim=20, verbosity=0)[0]
        comb_f = FISTA(Mop_radon, d.flatten(), eps=5e-2, niter=50, eigsiter=4, eigstol=1e-3, 
                       tol=1e-2, returninfo=False, show=True)[0]
        comb_f = Sparseop * comb_f
        comb_f = comb_f.reshape(6*nr, (2*nt-1))
        comb_f_tot = comb_f + np.concatenate((np.zeros((nr, 2*nt-1)),
                                              fd_plus, np.zeros((4*nr, 2*nt-1))))
        
        f1_1 = comb_f_tot[:2*nr] + comb_f_tot[2*nr:4*nr]
        f1_2 = comb_f_tot[:2*nr] + comb_f_tot[4*nr:]
        
        g_1 = BlockDiag([WiSop1,WiSop1])*Gop1*f1_1.flatten()
        g_1 = g_1.reshape(2*nsava1, (2*nt-1))
        g_2 = BlockDiag([WiSop2,WiSop2])*Gop2*f1_2.flatten()
        g_2 = g_2.reshape(2*nsava2, (2*nt-1))
        
        f1_1_minus, f1_1_plus =  f1_1[:nr], f1_1[nr:]
        f1_2_minus, f1_2_plus =  f1_2[:nr], f1_2[nr:]
        g_1_minus, g_1_plus =  -g_1[:nsava1], np.fliplr(g_1[nsava1:])
        g_2_minus, g_2_plus =  -g_2[:nsava2], np.fliplr(g_2[nsava2:])
        
        FUP1[:,ix,:]=f1_1_minus 
        FDOWN1[:,ix,:]=f1_1_plus
        PUP1[:,ix,:]=g_1_minus[:,nt-1:]
        PDOWN1[:,ix,:]=g_1_plus[:,nt-1:]
        PUP2[:,ix,:]=g_2_minus[:,nt-1:]
        PDOWN2[:,ix,:]=g_2_plus[:,nt-1:]

    jt=2
    redatumed1 = MDD(PDOWN1[:,:,::jt], PUP1[:,:,::jt], dt = jt*dt, dr = dvsx, twosided=True, adjoint=False, psf=False, dtype='complex64', dottest=False, **dict(iter_lim=20, show=0))
    redatumed2 = MDD(PDOWN2[:,:,::jt], PUP2[:,:,::jt], dt = jt*dt, dr = dvsx, twosided=True, adjoint=False, psf=False, dtype='complex64', dottest=False, **dict(iter_lim=20, show=0))
    
    #np.savetxt('datasets_radon/Line1_' + str(z_current) + '.dat', np.diag(redatumed1[:,:,(nt+1)/jt-1]), delimiter=',')
    #np.savetxt('datasets_radon/Line2_' + str(z_current) + '.dat', np.diag(redatumed2[:,:,(nt+1)/jt-1]), delimiter=',')
    return FUP1, FDOWN1, PUP1, PDOWN1, PUP2, PDOWN2, redatumed1, redatumed2


if __name__ == "__main__":
    zinvs = int(sys.argv[1])
    zendvs = int(sys.argv[2])
    nproc = int(sys.argv[3])

    Marchenko_depthloop_JointSparse(zinvs, zendvs, nproc)

