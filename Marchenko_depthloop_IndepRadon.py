# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 18:57:43 2020

@author: Claudia Haindl
"""

import sys

import numpy as np

from scipy.special import hankel2

from pylops.basicoperators        import Restriction
from pylops.waveeqprocessing      import MDC, MDD
from pylops.signalprocessing      import Radon2D, Sliding2D
from pylops.basicoperators        import Diagonal, BlockDiag, VStack, HStack
from pylops.optimization.sparsity import SPGL1, FISTA

from AngleGather import AngleGather
from find_closest import find_closest

# set paths:
path0='./datasets/'
path_save='./results_IndepRadon/'

def Marchenko_depthloop_IndepRadon(zinvs, zendvs):
    
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
    iava = np.loadtxt(path0 + 'select_rfrac70.dat', delimiter=',', dtype=int) - 1
    Restrop = Restriction(ns*(2*nt-1), iava, dims=(ns, 2*nt-1), dir=0, dtype='float64')
    
    # data
    print('Loading reflection data...')
    R=np.zeros((nt, ns, nr),'f')
    for isrc in range(ns-1):
        is_ = isrc*jr
        R[:,:,isrc]=np.loadtxt(path0 + 'R/dat1_' + str(is_) + '.dat', delimiter=',')
    
    R = 2 * np.swapaxes(R,0,2)
    
    # wavelet
    wav = np.loadtxt(path0 + 'wav.dat', delimiter=',')
    wav_c = wav[np.argmax(wav)-60:np.argmax(wav)+60]
    W = np.abs(np.fft.rfft(wav, nfft)) * dt
    
    # Convolution operators
    print('Creating MDC operators...')
    Rtwosided = np.concatenate((np.zeros((nr, ns, nt-1)), R), axis=-1)
    R1twosided = np.concatenate((np.flip(R, axis=-1), 
                                np.zeros((nr, ns, nt-1))), axis=-1)
    
    Rtwosided_fft = np.fft.rfft(Rtwosided, 2*nt-1, axis=-1)/np.sqrt(2*nt-1)
    Rtwosided_fft = Rtwosided_fft[...,:nfmax]
    R1twosided_fft = np.fft.rfft(R1twosided, 2*nt-1, axis=-1)/np.sqrt(2*nt-1)
    R1twosided_fft = R1twosided_fft[...,:nfmax]
    
    Rop = MDC(Rtwosided_fft[iava], nt=2*nt-1, nv=1, dt=dt, dr=ds, twosided=True, dtype='complex64')
    R1op = MDC(R1twosided_fft[iava], nt=2*nt-1, nv=1, dt=dt, dr=ds, twosided=True, dtype='complex64')
   
    del wav, R, Rtwosided, R1twosided, Rtwosided_fft, R1twosided_fft
    
    # configuring radon transform
    nwin=23
    nwins=14
    nover=10
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
    Sparseop = BlockDiag([Slidop,Slidop])

    del nwin, nwins, nover, npx, pxmax, px, t2, nt2, dimsd, dimss, RadOp, Slidop
    
    # the loop
    print('Entering loop...')
    z_steps = np.arange(zinvs, zendvs + 1, dvsz)
    for z_current in z_steps:
        redatuming_wrapper(toff, W, wav_c, iava, Rop, R1op, Restrop, Sparseop, vsx, vsz, x, z, z_current, nt, dt, nfft, nr, ds, dvsx)

            
    
def redatuming_wrapper(toff, W, wav, iava, Rop, R1op, Restrop, Sparseop, vsx, vsz, x, z, z_current, nt, dt, nfft, nr, ds, dvsx):

    from scipy.signal import filtfilt
    
    nava=iava.shape[0]
    nvsx = vsx.shape[0]
    
    PUP=np.zeros(shape=(nava,nvsx,nt))
    PDOWN=np.zeros(shape=(nava,nvsx,nt))
     
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
        nsava=iava.shape[0]
        
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
        
        # create operators
        Wop = Diagonal(w.flatten())
        WSop = Diagonal(w[iava].flatten())
        WiSop = Diagonal(wi[iava].flatten())
        
        Mop = VStack([HStack([Restrop, -1*WSop*Rop]),
                       HStack([-1*WSop*R1op, Restrop])])*BlockDiag([Wop, Wop])
        
        Mop_radon =  Mop * Sparseop
        
        Gop = VStack([HStack([Restrop, -1*Rop]),
                       HStack([-1*R1op, Restrop])])
        
        d = WSop*Rop*fd_plus.flatten()
        d = np.concatenate((d.reshape(nsava, 2*nt-1), np.zeros((nsava, 2*nt-1))))
        
        # solve with SPGL1
        f = SPGL1(Mop_radon, d.flatten(), sigma=1e-5, iter_lim=35, opt_tol=0.05, dec_tol=0.05, verbosity=1)[0]
        
        # alternatively solve with FISTA
        #f = FISTA(Mop_radon, d.flatten(), eps=1e-1, niter=200,
        #           alpha=2.129944e-04, eigsiter=4, eigstol=1e-3, 
        #           tol=1e-2, returninfo=False, show=True)[0]
        
        # alternatively solve with LSQR
        #f = lsqr(Mop_radon, d.flatten(), iter_lim=100, show=True)[0]
        
        f = Sparseop * f
        f = f.reshape(2*nr, (2*nt-1))
        f_tot = f + np.concatenate((np.zeros((nr, 2*nt-1)), fd_plus))
        
        g_1 = BlockDiag([WiSop,WiSop])*Gop*f_tot.flatten()
        g_1 = g_1.reshape(2*nsava, (2*nt-1))
        
        #f1_minus, f1_plus =  f_tot[:nr], f_tot[nr:]
        g_minus, g_plus = -g_1[:nsava], np.fliplr(g_1[nsava:])
        
        # 
        PUP[:,ix,:]=g_minus[:,nt-1:]
        PDOWN[:,ix,:]=g_plus[:,nt-1:]
    
    # save redatumed wavefield (line-by-line)    
    jt=2
    redatumed = MDD(PDOWN[:,:,::jt], PUP[:,:,::jt], dt = jt*dt, dr = dvsx, wav=wav[::jt], twosided=True, adjoint=False, psf=False, dtype='complex64', dottest=False, **dict(iter_lim=20, show=0))
    
    np.savetxt(path_save + 'Line1_' + str(z_current) + '.dat', np.diag(redatumed[:,:,(nt+1)//jt-1]), delimiter=',')
    
    # calculate and save angle gathers (line-by-line)
    vel_sm = np.loadtxt(path0 + 'vel_sm.dat', delimiter=',')
    cp=vel_sm[find_closest(z_current,z),751//2];
        
    irA=np.asarray([7,15,24,35])
    nalpha=201
    A=np.zeros((nalpha,len(irA)))
    
    for i in np.arange(0,len(irA)):
        ir=irA[i]
        anglegath, alpha =AngleGather(np.swapaxes(redatumed,0,2),nvsx,nalpha,dt*jt,ds,ir,cp);
        A[:,i]=anglegath;
        
    np.savetxt(path_save + 'AngleGather1_' + str(z_current) + '.dat', A, delimiter=',')

if __name__ == "__main__":
    zinvs = int(sys.argv[1])
    zendvs = int(sys.argv[2])

    Marchenko_depthloop_IndepRadon(zinvs, zendvs)
