# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

from scipy.sparse.linalg import lsqr
from scipy.signal import filtfilt

from pylops.basicoperators        import Diagonal, BlockDiag, VStack, HStack, Zero
from pylops.signalprocessing      import Radon2D, Sliding2D

import Operators as OP
 
def machenkoFocSourceLoop_JointLsqr(direct,toff,g0VS,iava1,Rop1,R1op1,Restrop1,iava2,Rop2,R1op2,Restrop2,t):
    #%%
    nsmooth=10
    nr=direct.shape[0]
    nsava1=iava1.shape[0]
    nsava2=iava2.shape[0]
    
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
    Mop = VStack([HStack([Mop1, Mop1, Zero(Mop1.shape[0],Mop1.shape[1])]),
                  HStack([Mop2, Zero(Mop2.shape[0],Mop2.shape[1]), Mop2])])
    
    Gop1 = VStack([HStack([Restrop1, -1*Rop1]),
                   HStack([-1*R1op1, Restrop1])])
    Gop2 = VStack([HStack([Restrop2, -1*Rop2]),
                   HStack([-1*R1op2, Restrop2])])
    
    #%%
    d1 = WSop1*Rop1*fd_plus.flatten()
    d1 = np.concatenate((d1.reshape(nsava1, 2*nt-1), np.zeros((nsava1, 2*nt-1))))
    d2 = WSop2*Rop2*fd_plus.flatten()
    d2 = np.concatenate((d2.reshape(nsava2, 2*nt-1), np.zeros((nsava2, 2*nt-1))))
    
    d = np.concatenate((d1, d2))
    
    comb_f = lsqr(Mop, d.flatten(), iter_lim=10, show=False)[0]
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
    #%%
    return f1_1_minus, f1_1_plus, f1_2_minus, f1_2_plus, g_1_minus, g_1_plus, g_2_minus, g_2_plus

def machenkoFocSourceLoop_JointSparse(direct,toff,g0VS,iava1,Rop1,R1op1,Restrop1,iava2,Rop2,R1op2,Restrop2,t,nr,dr):
    #%%
    nwin=25
    nwins=12
    nover=10
    npx=101
    pxmax = 1e-3
    px = np.linspace(-pxmax, pxmax, npx)
    
    nt=t.shape[0]
    dt=t[1]-t[0]
    t2=np.concatenate([-t[::-1], t[1:]])
    nt2=t2.shape[0]
    
    dimsd = (nr, nt2) 
    dimss = (nwins*npx, dimsd[1])
    
    # tranpose operator
    Top = OP.Transpose((nt2, nr), axes=(1, 0), dtype=np.float64)
    
    # sliding window radon with overlap
    RadOp = Radon2D(t2, np.linspace(-dr*nwin//2, dr*nwin//2, nwin), px, centeredh=True,
                    kind='linear', engine='numba')
    Slidop = Sliding2D(RadOp, dimss, dimsd, nwin, nover, tapertype='cosine', design=True)
    Sparseop = BlockDiag([Top.H*Slidop, Top.H*Slidop, Top.H*Slidop, Top.H*Slidop, Top.H*Slidop, Top.H*Slidop])
    
    nsmooth=10
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
    Mop = VStack([HStack([Mop1, Mop1, Zero(Mop1.shape[0],Mop1.shape[1])]),
                  HStack([Mop2, Zero(Mop2.shape[0],Mop2.shape[1]), Mop2])])
    
    Mop_radon =  Mop * Sparseop
    
    Gop1 = VStack([HStack([Restrop1, -1*Rop1]),
                   HStack([-1*R1op1, Restrop1])])
    Gop2 = VStack([HStack([Restrop2, -1*Rop2]),
                   HStack([-1*R1op2, Restrop2])])
    
    #%%
    d1 = WSop1*Rop1*fd_plus.flatten()
    d1 = np.concatenate((d1.reshape(nsava1, 2*nt-1), np.zeros((nsava1, 2*nt-1))))
    d2 = WSop2*Rop2*fd_plus.flatten()
    d2 = np.concatenate((d2.reshape(nsava2, 2*nt-1), np.zeros((nsava2, 2*nt-1))))
    
    d = np.concatenate((d1, d2))
    
    comb_f = OP.SPGL1(Mop_radon, d.flatten(), sigma=1e-1, iter_lim=10, verbosity=0)[0]
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
    #%%
    return f1_1_minus, f1_1_plus, f1_2_minus, f1_2_plus, g_1_minus, g_1_plus, g_2_minus, g_2_plus

def machenkoFocSourceLoop_Lsqr(direct,toff,g0VS,iava,Rop,R1op,Restrop,t):
    #%%
    nsmooth=10
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
    
    if nsmooth>0:
        smooth=np.ones(nsmooth)/nsmooth
        w  = filtfilt(smooth, 1, w)
        wi  = filtfilt(smooth, 1, wi)
        
    # Input focusing function
    fd_plus =  np.concatenate((np.fliplr(g0VS.T), np.zeros((nr, nt-1))), axis=-1)
    
    Wop = Diagonal(w.flatten())
    WSop = Diagonal(w[iava].flatten())
    WiSop = Diagonal(wi[iava].flatten())
    
    Mop = VStack([HStack([Restrop, -1*WSop*Rop]),
                   HStack([-1*WSop*R1op, Restrop])])*BlockDiag([Wop, Wop])
    
    Gop = VStack([HStack([Restrop, -1*Rop]),
                   HStack([-1*R1op, Restrop])])
    
    #%%
    d = WSop*Rop*fd_plus.flatten()
    d = np.concatenate((d.reshape(nsava, 2*nt-1), np.zeros((nsava, 2*nt-1))))
    
    f1 = lsqr(Mop, d.flatten(), iter_lim=10, show=False)[0]
    f1 = f1.reshape(2*nr, (2*nt-1))
    f1_tot = f1 + np.concatenate((np.zeros((nr, 2*nt-1)), fd_plus))
    
    g = BlockDiag([WiSop,WiSop])*Gop*f1_tot.flatten()
    g = g.reshape(2*nsava, (2*nt-1))
    
    f1_minus, f1_plus =  f1_tot[:nr], f1_tot[nr:]
    g_minus, g_plus =  -g[:nsava], np.fliplr(g[nsava:])

    #%%
    return f1_minus, f1_plus, g_minus, g_plus

def find_closest(nr,a):
    i0 = np.nonzero(a <= nr)[-1][-1]
    i1 = np.nonzero(nr < a)[0][0]

    if ((a[i0]-nr)**2 <= (a[i1]-nr)**2):
        return i0
    else:
        return i1