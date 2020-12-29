import numpy as np

def AngleGather(d,nfft_k,nalpha,dt,ds,ir,cp):
    nfft_f=1024
    ifin=10

    # param
    f=np.arange(0,nfft_f/2+1)/(nfft_f*dt)
    k=np.arange((-nfft_k/2+1),nfft_k/2+1)/(nfft_k*ds)

    # extract single virtual dipole source
    d_tmp=np.fft.fftshift(np.squeeze(d[:,ir,:]),0)
    d=np.zeros((nfft_f,nfft_k))
    d[-nfft_f//2-1:,:] = d_tmp[-nfft_f//2-1:,:]
    d[:nfft_f//2,:] = d_tmp[:nfft_f//2,:]
   
    # convert from t-x to f-x domain
    D_r=np.fft.fft(d,nfft_f,0)
    D_r=D_r[:nfft_f//2+1,:]
    D_r=np.hstack((D_r[:,ir:], D_r[:,:ir]))
    D_fk=np.fft.fftshift(np.fft.fft(D_r,nfft_k,1),1)
 
    # convert from f-kx to f-angle
    alpha=np.linspace(-90,90,nalpha)
    sinalpha=np.sin(alpha*np.pi/180)
    Alpha_sampled=np.zeros((nfft_f//2+1,nfft_k))

    D_alpha=np.zeros((nfft_f//2+1,nalpha), dtype=np.complex)
    for iif in np.arange(ifin,nfft_f//2+1):
        sinalpha_sampled=cp*k/f[iif]
        Alpha_sampled[iif,:]=sinalpha_sampled
        D_alpha[iif,:]= np.interp(sinalpha,sinalpha_sampled,np.real(D_fk[iif,:])) \
                          + 1j*np.interp(sinalpha,sinalpha_sampled,np.imag(D_fk[iif,:]))
    D_alpha[np.isnan(D_alpha)]=0

    # create angle gather
    R_alpha=np.fft.ifftshift(np.fft.ifft(D_alpha,nfft_f,0),0)
    R=np.sum(D_alpha[0:nfft_f//2-1,:],axis=0)

    return R, alpha, R_alpha
