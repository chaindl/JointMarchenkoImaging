# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

from scipy.sparse.linalg import lsqr
from scipy.signal import filtfilt

from pylops.basicoperators import Diagonal, BlockDiag, VStack, HStack, Zero, Identity


def machenkoFocSourceLoop_JointLsqr_fullsurvey(direct,toff,g0VS,Rop1,R1op1,Rop2,R1op2,t):
    #%%
    nsmooth=10
    nr=direct.shape[0]
    ns = nr

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
    WSop1 = Diagonal(w.flatten())
    WSop2 = Diagonal(w.flatten())
    WiSop1 = Diagonal(wi.flatten())
    WiSop2 = Diagonal(wi.flatten())
    
    Iop = Identity(nr * (2*nt-1))
    
    Mop1 = VStack([HStack([Iop, -1*WSop1*Rop1]),
                   HStack([-1*WSop1*R1op1, Iop])])*BlockDiag([Wop, Wop])
    Mop2 = VStack([HStack([Iop, -1*WSop2*Rop2]),
                   HStack([-1*WSop2*R1op2, Iop])])*BlockDiag([Wop, Wop])
    Mop = VStack([HStack([Mop1, Mop1, Zero(Mop1.shape[0],Mop1.shape[1])]),
                  HStack([Mop2, Zero(Mop2.shape[0],Mop2.shape[1]), Mop2])])
    
    Gop1 = VStack([HStack([Iop, -1*Rop1]),
                   HStack([-1*R1op1, Iop])])
    Gop2 = VStack([HStack([Iop, -1*Rop2]),
                   HStack([-1*R1op2, Iop])])
    
    #%%
    d1 = WSop1*Rop1*fd_plus.flatten()
    d1 = np.concatenate((d1.reshape(ns, 2*nt-1), np.zeros((ns, 2*nt-1))))
    d2 = WSop2*Rop2*fd_plus.flatten()
    d2 = np.concatenate((d2.reshape(ns, 2*nt-1), np.zeros((ns, 2*nt-1))))
    
    d = np.concatenate((d1, d2))
    
    comb_f = lsqr(Mop, d.flatten(), iter_lim=10, show=False)[0]
    comb_f = comb_f.reshape(6*nr, (2*nt-1))
    comb_f_tot = comb_f + np.concatenate((np.zeros((nr, 2*nt-1)),
                                          fd_plus, np.zeros((4*nr, 2*nt-1))))
    
    f1_1 = comb_f_tot[:2*nr] + comb_f_tot[2*nr:4*nr]
    f1_2 = comb_f_tot[:2*nr] + comb_f_tot[4*nr:]
    
    g_1 = BlockDiag([WiSop1,WiSop1])*Gop1*f1_1.flatten()
    g_1 = g_1.reshape(2*ns, (2*nt-1))
    g_2 = BlockDiag([WiSop2,WiSop2])*Gop2*f1_2.flatten()
    g_2 = g_2.reshape(2*ns, (2*nt-1))
    
    f1_1_minus, f1_1_plus =  f1_1[:nr], f1_1[nr:]
    f1_2_minus, f1_2_plus =  f1_2[:nr], f1_2[nr:]
    g_1_minus, g_1_plus =  -g_1[:ns], np.fliplr(g_1[ns:])
    g_2_minus, g_2_plus =  -g_2[:ns], np.fliplr(g_2[ns:])
    #%%
    return f1_1_minus, f1_1_plus, f1_2_minus, f1_2_plus, g_1_minus, g_1_plus, g_2_minus, g_2_plus
    


def machenkoFocSourceLoop_JointLsqr(direct,toff,g0VS,iava1,Rop1,R1op1,Restrop1,iava2,Rop2,R1op2,Restrop2,t):
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
    
def find_closest(nr,a):
    i0 = np.nonzero(a <= nr)[-1][-1]
    i1 = np.nonzero(nr < a)[0][0]

    if ((a[i0]-nr)^2 >= (a[i1]-nr)^2):
        return i0
    else:
        return i1
