"""
This script is used to extract features vector from Room Impulse Response.
STFT - function calculating STFT (power spectrum)
iSTFT - calculates iSTFT
optimal_synth_window - calculates synthesis window based on analysis window
MelFilters - function calculating Mel Filter Bank
features - function calculating LMSC and MFCC coefficients of RIR

Copyright to:
    Katarzyna Augustyn
    Dominika Godzisz
    Bartłomiej Piekarz
"""

import numpy as np
from scipy.fftpack import dct
def STFT(x, W, L, N=None):
    """
    This function computes Short-Time Fourier Transfor of the signal
    Parameters
    x - vector that contains signal
    W - Window
    L - Overlap between consecutive frames
    N - Number of discrete frequencies in output STFT

    Returns
    X: Power spectrum of each frame
    """
    xLen = x.size
    WLen = W.size
    if N is None:
        N = WLen
    K=int((xLen-L)/(WLen-L))
    X=np.zeros((K,N))
    for i in range(K):
        st=int(i*(WLen-L))
        end=int(st+WLen)
        h=np.fft.fft(x[st:end]*W,n=N)
        h=np.abs(h)
        power=(h**2)/(N/2+1)
        X[i,:]=power
    return X

def iSTFT(X, W, L):
    """
    Parameters:
    X - STFT
    W - synthesis window
    L - Overlap between consecutive frames (samples)
    
    Returns:
    x - reconstructed signal
    """
    # Number of frames and Number of samples
    K, N = X.shape
    
    # Number of samples in synthesis window
    WLen = W.size
    # Częstotliwość próbkowania sygnału oknem
    hop = WLen - L
    xlen=(K*hop)+L
    x=np.zeros(xlen)
    
    for i in range(K):
        s = np.real(np.fft.ifft(X[i,:]))
        Ws=np.zeros(N)
        Ws[0:WLen]=W
        swin=s*Ws
        trim=x[i*hop:(i*hop)+N].size
        x[i*hop:(i*hop)+N]=x[i*hop:(i*hop)+N] + swin[0:trim]

    return x

def optimal_synth_window(window, L):
    """
    Parameters:
    window - analysis window
    L - Overlap between consecutive frames (samples)
    
    Returns:
    s_win - synthesis window
    """
    WL=window.size
    S=window.size-L
    app_0=np.zeros(S)
    it_win=window
    sw=window**2
    for shift in range(S,WL,S):
        it_win=np.append(app_0,it_win)
        it_win=np.append(it_win,app_0)
        sw=sw+(it_win[0:WL]**2)
        sw=sw+(it_win[it_win.size-WL:]**2)
    s_win=window/sw
    return s_win

def MelFilters(fs,nfilt,nfft):
    """
    This function returns Mel Filters Bank
    Parameters
    fs - sampling rate
    nfilt - number of filters
    nfft - fft size
    Returns
    X: Matrix of mel filters
    """
    l_mel=0
    h_mel=2595*np.log10(1+((fs/2)/700))
    mel_f=np.linspace(l_mel,h_mel,nfilt+2)
    hz_f=(700 * (10**(mel_f / 2595) - 1))
    bin=np.floor((nfft+1)*hz_f/fs) # indexes in fft of hz_f points
    fbank=np.zeros((nfilt,int(np.floor(nfft/2+1))))
    for i in range(1,nfilt+1):
        f_l=int(bin[i-1])
        f_m=int(bin[i])
        f_h=int(bin[i+1])
        for f in range(f_l,f_m):
            fbank[i-1,f]=(f-f_l)/(f_m-f_l)
        for f in range(f_m,f_h):
            fbank[i-1,f]=(f_h-f)/(f_h-f_m)
    return fbank

def features(RIR, fs,dt=0.025):
    """
    This function calucates 48 features for each signal frame
    Parameters
    RIR - room impulse response 1D array
    fs - sampling rate
    dt - length of an analysis window in seconds
    Returns
    features_matrix - n_framesx48 matrix of features
    """
    win=np.hamming(int(dt*fs)) #Hamming window
    step=int(fs*dt*0.5) #50% overlapping
    nfft=512 #fft size
    RIR_STFT=STFT(x=RIR,W=win,L=step,N=nfft)
    MEL=MelFilters(fs,26,nfft)
    MSC=np.dot(RIR_STFT[:,:int(np.floor(nfft/2)+1)],MEL.T)
    MSC = np.where(MSC == 0, np.finfo(float).eps, MSC)
    LMSC=20*np.log10(MSC)
    MFCC=dct(LMSC, type=2, axis=1, norm='ortho')
    LMSC=LMSC[:,:24]
    MFCC=MFCC[:,:24]
    (n, ncoeff) = MFCC.shape
    n = np.arange(ncoeff)
    lift = 1 + (22 / 2) * np.sin(np.pi * n / 22)
    MFCC *= lift
    features_matrix=np.concatenate((LMSC,MFCC),axis=1)

    return features_matrix
