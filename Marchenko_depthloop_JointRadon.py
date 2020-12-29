# -*- coding: utf-8 -*-
"""
Created on Thu May 23 17:40:55 2019

@author: Claudia Haindl
"""

import sys

import numpy as np

from scipy.special import hankel2

from pylops.basicoperators        import Restriction
from pylops.waveeqprocessing      import MDC, MDD
from pylops.signalprocessing      import Radon2D, Sliding2D
from pylops.basicoperators        import Diagonal, BlockDiag, VStack, HStack, Zero
from pylops.optimization.sparsity import SPGL1, FISTA

from AngleGather import AngleGather
from find_closest import find_closest

# set paths:
path0='./datasets/'
path_save='./results_JointRadon/'


def Marchenko_depthloop_JointRadon(zinvs, zendvs):
    
    toff = 0.045        # direct arrival time shift
    nfmax = 550         # max frequency for MDC (#samples)
    nfft = 2 ** 11
    
    jr = 3              # subsampling in r
    
    # subsurface array
    xinvs=600         # receiver array initial point in x
    xendvs=2400       # receiver array last point in x

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
    
    # Receivers
    r = np.loadtxt(path0 + 'r.dat', delimiter=',')
    nr = r.shape[1]
    
    # Sources
    s = np.loadtxt(path0 + 's.dat', delimiter=',')
    ns = s.shape[1]
    ds = s[0,1]-s[0,0]
    
    # restriction operators
    iava1 = np.loadtxt(path0 + 'select_rfrac70.dat', delimiter=',', dtype=int) - 1
    iava2 = np.loadtxt(path0 + 'select_rfrac50.dat', delimiter=',', dtype=int) - 1
    Restrop1 = Restriction(ns*(2*nt-1), iava1, dims=(ns, 2*nt-1), dir=0, dtype='float64')
    Restrop2 = Restriction(ns*(2*nt-1), iava2, dims=(ns, 2*nt-1), dir=0, dtype='float64')
    
    # data
    print('Loading reflection data...')
    R_1=np.zeros((nt, ns, nr),'f')
    R_2=np.zeros((nt, ns, nr),'f')
    for isrc in range(ns-1):
        is_ = isrc*jr
        R_1[:,:,isrc]=np.loadtxt(path0 + 'R/dat1_' + str(is_) + '.dat', delimiter=',')
        R_2[:,:,isrc]=np.loadtxt(path0 + 'R/dat2_' + str(is_) + '.dat', delimiter=',')
    
    R_1 = 2 * np.swapaxes(R_1,0,2)
    R_2 = 2 * np.swapaxes(R_2,0,2)
    
    # wavelet
    wav = np.loadtxt(path0 + 'wav.dat', delimiter=',')
    wav_c = wav[np.argmax(wav)-60:np.argmax(wav)+60]
    W = np.abs(np.fft.rfft(wav, nfft)) * dt
    
    # Convolution operators
    print('Creating MDC operators...')
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
    
    # configuring radon transform
    nwin=21
    nwins=13
    nover=6
    npx=101
    pxmax = 0.006
    px = np.linspace(-pxmax, pxmax, npx)
    
    t2=np.concatenate([-t[::-1], t[1:]])
    nt2=t2.shape[0]
    
    dimsd = (nr, nt2) 
    dimss = (nwins*npx, dimsd[1])
    
    # sliding window radon with overlap
    RadOp = Radon2D(t2, np.linspace(-ds*nwin//2, ds*nwin//2, nwin), px, centeredh=True,
                    kind='linear', engine='numba')
    Slidop = Sliding2D(RadOp, dimss, dimsd, nwin, nover, tapertype='cosine', design=True)
    Sparseop = BlockDiag([Slidop,Slidop,Slidop,Slidop,Slidop,Slidop])

    del nwin, nwins, nover, npx, pxmax, px, t2, nt2, dimsd, dimss, RadOp, Slidop
    
    # the loop
    print('Entering loop...')
    z_steps = np.arange(zinvs, zendvs + 1, dvsz)
    for z_current in z_steps:
        redatuming_wrapper(toff, W, wav_c, iava1, iava2, Rop1, Rop2, R1op1, R1op2, Restrop1, Restrop2, Sparseop, vsx, vsz, x, z, z_current, nt, dt, nfft, nr, ds, dvsx)
            
    
def redatuming_wrapper(toff, W, wav, iava1, iava2, Rop1, Rop2, R1op1, R1op2, Restrop1, Restrop2, Sparseop, vsx, vsz, x, z, z_current, nt, dt, nfft, nr, ds, dvsx):

    from scipy.signal import filtfilt
    
    nava1=iava1.shape[0]
    nava2=iava2.shape[0]
    nvsx = vsx.shape[0]
    
    PUP1=np.zeros(shape=(nava1,nvsx,nt))
    PDOWN1=np.zeros(shape=(nava1,nvsx,nt))
    PUP2=np.zeros(shape=(nava2,nvsx,nt))
    PDOWN2=np.zeros(shape=(nava2,nvsx,nt))
     
    for ix in range(nvsx):
        s = '####### Point ' + str(ix+1) + ' of ' + str(nvsx) + ' of current line (z = ' + str(z_current) + ', x = ' + str(vsx[ix]) + ')'
        print(s)
        
        # direct wave
        direct = np.loadtxt(path0 + 'Traveltimes/trav_x' + str(vsx[ix]) + '_z' + str(z_current) + '.dat', delimiter=',')
        f = 2 * np.pi * np.arange(nfft) / (dt * nfft)
        g0VS = np.zeros((nfft, nr), dtype=np.complex128)
        for it in range(len(W)):
            g0VS[it] = W[it] * f[it] * hankel2(0, f[it] * direct + 1e-10) / 4
        g0VS = np.fft.irfft(g0VS, nfft, axis=0) / dt
        g0VS = np.real(g0VS[:nt])
        
        nr=direct.shape[0]
        nsava1=iava1.shape[0]
        nsava2=iava2.shape[0]
        
        # window
        directVS_off = direct - toff
        idirectVS_off = np.round(directVS_off/dt).astype(np.int)
        w = np.zeros((nr, nt))
        wi = np.ones((nr, nt))
        for ir in range(nr-1):
            w[ir, :idirectVS_off[ir]]=1   
        wi = wi - w
             
        w = np.hstack((np.fliplr(w), w[:, 1:]))
        wi = np.hstack((np.fliplr(wi), wi[:, 1:]))
        
        # smoothing
        nsmooth=10
        if nsmooth>0:
            smooth=np.ones(nsmooth)/nsmooth
            w  = filtfilt(smooth, 1, w)
            wi  = filtfilt(smooth, 1, wi)
            
        # Input focusing function
        fd_plus =  np.concatenate((np.fliplr(g0VS.T), np.zeros((nr, nt-1))), axis=-1)
        
        # operators
        Wop = Diagonal(w.flatten())
        WSop1 = Diagonal(w[iava1].flatten())
        WSop2 = Diagonal(w[iava2].flatten())
        WiSop1 = Diagonal(wi[iava1].flatten())
        WiSop2 = Diagonal(wi[iava2].flatten())
        
        
        Mop1 = VStack([HStack([Restrop1, -1*WSop1*Rop1]),
                       HStack([-1*WSop1*R1op1, Restrop1])])*BlockDiag([Wop, Wop])
        Mop2 = VStack([HStack([Restrop2, -1*WSop2*Rop2]),
                       HStack([-1*WSop2*R1op2, Restrop2])])*BlockDiag([Wop, Wop])
        Mop = VStack([HStack([Mop1, Mop1, Zero(Mop1.shape[0],Mop1.shape[1])]),
                      HStack([Mop2, Zero(Mop2.shape[0],Mop2.shape[1]), Mop2])])
        
        Mop_radon =  Mop * Sparseop
        
        Gop1 = VStack([HStack([Restrop1, -1*Rop1]),
                       HStack([-1*R1op1, Restrop1])])
        Gop2 = VStack([HStack([Restrop2, -1*Rop2]),
                       HStack([-1*R1op2, Restrop2])])
        
        d1 = WSop1*Rop1*fd_plus.flatten()
        d1 = np.concatenate((d1.reshape(nsava1, 2*nt-1), np.zeros((nsava1, 2*nt-1))))
        d2 = WSop2*Rop2*fd_plus.flatten()
        d2 = np.concatenate((d2.reshape(nsava2, 2*nt-1), np.zeros((nsava2, 2*nt-1))))
        
        d = np.concatenate((d1, d2))
        
        # solve with SPGL1
        comb_f = SPGL1(Mop_radon, d.flatten(), sigma=1e-5, iter_lim=30, opt_tol=0.05, dec_tol=0.05, verbosity=1)[0]

        # alternatively solve with FISTA
        #comb_f = FISTA(Mop_radon, d.flatten(), eps=1e-1, niter=200,
        #           alpha=2.129944e-04, eigsiter=4, eigstol=1e-3, 
        #           tol=1e-2, returninfo=False, show=True)[0]
        
        # alternatively solve with LSQR
        #comb_f = lsqr(Mop_radon, d.flatten(), iter_lim=100, show=True)[0]

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
        
        #f1_1_minus, f1_1_plus =  f1_1[:nr], f1_1[nr:]
        #f1_2_minus, f1_2_plus =  f1_2[:nr], f1_2[nr:]
        g_1_minus, g_1_plus = -g_1[:nsava1], np.fliplr(g_1[nsava1:])
        g_2_minus, g_2_plus = -g_2[:nsava2], np.fliplr(g_2[nsava2:])
        
        PUP1[:,ix,:]=g_1_minus[:,nt-1:]
        PDOWN1[:,ix,:]=g_1_plus[:,nt-1:]
        PUP2[:,ix,:]=g_2_minus[:,nt-1:]
        PDOWN2[:,ix,:]=g_2_plus[:,nt-1:]
    
    # calculate and save redatumed wavefields (line-by-line)
    jt=2
    redatumed1 = MDD(PDOWN1[:,:,::jt], PUP1[:,:,::jt], dt = jt*dt, dr = dvsx, wav = wav[::jt], twosided=True, adjoint=False, psf=False, dtype='complex64', dottest=False, **dict(iter_lim=20, show=0))
    redatumed2 = MDD(PDOWN2[:,:,::jt], PUP2[:,:,::jt], dt = jt*dt, dr = dvsx, wav = wav[::jt], twosided=True, adjoint=False, psf=False, dtype='complex64', dottest=False, **dict(iter_lim=20, show=0))
    
    np.savetxt(path_save + 'Line1_' + str(z_current) + '.dat', np.diag(redatumed1[:,:,(nt+1)//jt-1]), delimiter=',')
    np.savetxt(path_save + 'Line2_' + str(z_current) + '.dat', np.diag(redatumed2[:,:,(nt+1)//jt-1]), delimiter=',')
    
    # calculate and save angle gathers (line-by-line)
    vel_sm = np.loadtxt(path0 + 'vel_sm.dat', delimiter=',')
    cp=vel_sm[find_closest(z_current,z),751//2];
    
    irA=np.asarray([7,15,24,35])
    nalpha=201
    A1=np.zeros((nalpha,len(irA)))
    A2=np.zeros((nalpha,len(irA)))
    
    for i in np.arange(0,len(irA)):
        ir=irA[i]
        anglegath1, alpha =AngleGather(np.swapaxes(redatumed1,0,2),nvsx,nalpha,dt*jt,ds,ir,cp);
        anglegath2, alpha =AngleGather(np.swapaxes(redatumed2,0,2),nvsx,nalpha,dt*jt,ds,ir,cp);
        A1[:,i]=anglegath1;
        A2[:,i]=anglegath2;
        
    np.savetxt(path_save + 'AngleGather1_' + str(z_current) + '.dat', A1, delimiter=',')
    np.savetxt(path_save + 'AngleGather2_' + str(z_current) + '.dat', A2, delimiter=',')

if __name__ == "__main__":
    zinvs = int(sys.argv[1])
    zendvs = int(sys.argv[2])

    Marchenko_depthloop_JointRadon(zinvs, zendvs)
