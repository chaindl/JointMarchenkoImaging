# -*- coding: utf-8 -*-
"""
Created on Thu May 23 17:40:55 2019

@author: Claudia Haindl
"""

import sys

import numpy as np

from scipy.special import hankel2
from scipy.sparse.linalg import lsqr
from scipy.signal import filtfilt

from pylops.basicoperators import Restriction, Diagonal, BlockDiag, VStack, HStack
from pylops.waveeqprocessing import MDC, MDD

from AngleGather import AngleGather
from find_closest import find_closest

# set paths:
path0='./datasets/'
path_save='./results_indepLSQR/'

def Marchenko_depthloop_IndepLsqr(zinvs,zendvs):
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
    nava1 = iava1.shape[0]
    nava2 = iava2.shape[0]
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
    
    Rop1 = MDC(Rtwosided_fft_1[iava1], nt=2*nt-1, nv=1, dt=dt, dr=ds, twosided=True, dtype='complex64')
    R1op1 = MDC(R1twosided_fft_1[iava1], nt=2*nt-1, nv=1, dt=dt, dr=ds, twosided=True, dtype='complex64')
    
    Rop2 = MDC(Rtwosided_fft_2[iava2], nt=2*nt-1, nv=1, dt=dt, dr=ds, twosided=True, dtype='complex64')
    R1op2 = MDC(R1twosided_fft_2[iava2], nt=2*nt-1, nv=1, dt=dt, dr=ds, twosided=True, dtype='complex64')
    
    # wavelet and traveltime
    print('Loading direct wave...')
    wav = np.loadtxt(path0 + 'wav.dat', delimiter=',')
    wav_c = wav[np.argmax(wav)-60:np.argmax(wav)+60]
    
    trav_eik = np.loadtxt(path0 + 'trav.dat', delimiter=',')
    trav_eik = np.reshape(trav_eik,(nz,ns,nx))
    
    # the loop
    print('Entering loop...')
    for iz in range(nvsz):

        z_current=vsz[iz]
        
        PDIR1=np.zeros(shape=(nava1,nvsx,nt))
        PDIR2=np.zeros(shape=(nava2,nvsx,nt))
        PRTM1=np.zeros(shape=(nava1,nvsx,nt))
        PRTM2=np.zeros(shape=(nava2,nvsx,nt))
        PUP1=np.zeros(shape=(nava1,nvsx,nt))
        PDOWN1=np.zeros(shape=(nava1,nvsx,nt))
        PUP2=np.zeros(shape=(nava2,nvsx,nt))
        PDOWN2=np.zeros(shape=(nava2,nvsx,nt))
        
        for ix in range(nvsx):
            
            s = '############ Point ' + str(ix+1) + ' of ' + str(nvsx) + ', line ' + str(iz+1) + ' of ' + str(nvsz) + ' (z = ' + str(vsz[iz]) + ', x = ' + str(vsx[ix]) + ') ... ' + str(100 * (iz*nvsx + ix + 1)/(nvsx*nvsz)) + '%'
            print(s)
            
            # direct wave
            direct = trav_eik[find_closest(z_current,z),:,find_closest(vsx[ix],x)]
            W = np.abs(np.fft.rfft(wav, nfft)) * dt
            f = 2 * np.pi * np.arange(nfft) / (dt * nfft)
            g0VS = np.zeros((nfft, nr), dtype=np.complex128)
            for it in range(len(W)):
                g0VS[it] = W[it] * f[it] * hankel2(0, f[it] * direct + 1e-10) / 4
            g0VS = np.fft.irfft(g0VS, nfft, axis=0) / dt
            g0VS = np.real(g0VS[:nt])
            
            # Marchenko focusing
            f1_1_minus, f1_1_plus, g_1_minus, g_1_plus, p0_1_minus = focusing_wrapper(direct,toff,g0VS,iava1,Rop1,R1op1,Restrop1,t)
            f1_2_minus, f1_2_plus, g_2_minus, g_2_plus, p0_2_minus = focusing_wrapper(direct,toff,g0VS,iava2,Rop2,R1op2,Restrop2,t)
            
            # assembling wavefields
            PRTM1[:,ix,:]=p0_1_minus[:,nt-1:]
            PRTM2[:,ix,:]=p0_2_minus[:,nt-1:]
            PDIR1[:,ix,:]=g0VS[:,iava1].T
            PDIR2[:,ix,:]=g0VS[:,iava2].T
            PUP1[:,ix,:]=g_1_minus[:,nt-1:]
            PUP2[:,ix,:]=g_2_minus[:,nt-1:]
            PDOWN1[:,ix,:]=g_1_plus[:,nt-1:]
            PDOWN2[:,ix,:]=g_2_plus[:,nt-1:]
        
        # calculate and save redatumed wavefields (line-by-line)  
        jt=2
        redatumed1 = MDD(PDOWN1[:,:,::jt], PUP1[:,:,::jt], dt = jt*dt, dr = dvsx, wav=wav_c[::jt], twosided=True, adjoint=False, psf=False, dtype='complex64', dottest=False, **dict(iter_lim=20, show=0))
        redatumed2 = MDD(PDOWN2[:,:,::jt], PUP2[:,:,::jt], dt = jt*dt, dr = dvsx, wav=wav_c[::jt], twosided=True, adjoint=False, psf=False, dtype='complex64', dottest=False, **dict(iter_lim=20, show=0))
        
        _,rtm1 = MDD(PDIR1[:,:,::jt], PRTM1[:,:,::jt], dt = jt*dt, dr = dvsx, wav=wav_c[::jt], twosided=True, adjoint=True, psf=False, dtype='complex64', dottest=False, **dict(iter_lim=20, show=0))
        _,rtm2 = MDD(PDIR2[:,:,::jt], PRTM2[:,:,::jt], dt = jt*dt, dr = dvsx, wav=wav_c[::jt], twosided=True, adjoint=True, psf=False, dtype='complex64', dottest=False, **dict(iter_lim=20, show=0))

        np.savetxt(path_save + 'Line1_rtm_' + str(z_current) + '.dat', np.diag(rtm1[:,:,(nt+1)//jt-1]), delimiter=',')
        np.savetxt(path_save + 'Line2_rtm_' + str(z_current) + '.dat', np.diag(rtm2[:,:,(nt+1)//jt-1]), delimiter=',')
 
        np.savetxt(path_save + 'Line1_' + str(z_current) + '.dat', np.diag(redatumed1[:,:,(nt+1)//jt-1]), delimiter=',')
        np.savetxt(path_save + 'Line2_' + str(z_current) + '.dat', np.diag(redatumed2[:,:,(nt+1)//jt-1]), delimiter=',')
        
        vel_sm = np.loadtxt(path0 + 'vel_sm.dat', delimiter=',')
        cp=vel_sm[find_closest(z_current,z),751//2];
            
        irA=np.asarray([7,15,24,35])
        nalpha=201
        A1=np.zeros((nalpha,len(irA)))
        A2=np.zeros((nalpha,len(irA)))
        A1rtm=np.zeros((nalpha,len(irA)))
        A2rtm=np.zeros((nalpha,len(irA)))
        
        # calculate and save angle gathers (line-by-line)
        for i in np.arange(0,len(irA)):
            ir=irA[i]
            anglegath, alpha =AngleGather(np.swapaxes(redatumed1,0,2),nvsx,nalpha,dt*jt,ds,ir,cp);
            A1[:,i]=anglegath;
            anglegath, alpha =AngleGather(np.swapaxes(redatumed2,0,2),nvsx,nalpha,dt*jt,ds,ir,cp);
            A2[:,i]=anglegath;
            anglegath, alpha =AngleGather(np.swapaxes(rtm1,0,2),nvsx,nalpha,dt*jt,ds,ir,cp);
            A1rtm[:,i]=anglegath;
            anglegath, alpha =AngleGather(np.swapaxes(rtm2,0,2),nvsx,nalpha,dt*jt,ds,ir,cp);
            A2rtm[:,i]=anglegath;
            
        np.savetxt(path_save + 'AngleGather1_' + str(z_current) + '.dat', A1, delimiter=',')
        np.savetxt(path_save + 'AngleGather2_' + str(z_current) + '.dat', A2, delimiter=',')
        np.savetxt(path_save + 'AngleGather1_rtm_' + str(z_current) + '.dat', A1rtm, delimiter=',')
        np.savetxt(path_save + 'AngleGather2_rtm_' + str(z_current) + '.dat', A2rtm, delimiter=',')


def focusing_wrapper(direct,toff,g0VS,iava,Rop,R1op,Restrop,t):
    nr=direct.shape[0]
    nsava=iava.shape[0]
    
    nt=t.shape[0]
    dt=t[1]-t[0]
    
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
    WSop = Diagonal(w[iava].flatten())
    WiSop = Diagonal(wi[iava].flatten())
    
    Mop = VStack([HStack([Restrop, -1*WSop*Rop]),
                   HStack([-1*WSop*R1op, Restrop])])*BlockDiag([Wop, Wop])
    
    Gop = VStack([HStack([Restrop, -1*Rop]),
                   HStack([-1*R1op, Restrop])])
    
    p0_minus = Rop*fd_plus.flatten()
    d = WSop*p0_minus
    
    p0_minus = p0_minus.reshape(nsava, 2*nt-1)
    d = np.concatenate((d.reshape(nsava, 2*nt-1), np.zeros((nsava, 2*nt-1))))
    
    # solve
    f1 = lsqr(Mop, d.flatten(), iter_lim=10, show=False)[0]
    f1 = f1.reshape(2*nr, (2*nt-1))
    f1_tot = f1 + np.concatenate((np.zeros((nr, 2*nt-1)), fd_plus))
    
    g = BlockDiag([WiSop,WiSop])*Gop*f1_tot.flatten()
    g = g.reshape(2*nsava, (2*nt-1))
    
    f1_minus, f1_plus =  f1_tot[:nr], f1_tot[nr:]
    g_minus, g_plus =  -g[:nsava], np.fliplr(g[nsava:])

    return f1_minus, f1_plus, g_minus, g_plus, p0_minus


if __name__ == "__main__":
    a = int(sys.argv[1])
    b = int(sys.argv[2])
    Marchenko_depthloop_indepLsqr(a, b)
