import numpy as np
from spd import *
from scipy.linalg import expm, logm


def sampleAlloc(B, c, var, corr):
    """
    Compute optimal coefficient vector a and sample size vector m
    """
    a = corr[1:]*np.sqrt(var[0]/var[1:])  # coefficient alpha
    corr = np.append(corr, 0.0)

    r = np.ones(len(var))

    for i in range(1, len(r)):
        r[i] = np.sqrt((c[0]*(corr[i]**2 - corr[i+1]**2))/(c[i]*(1.0 - corr[1]**2)))

    m = r*(B/(np.dot(c, r)))
    return (a,m)


def computeVarCorr(Sigma, noise_lvls):
    """
    Compute sample variances and correlations
    """
    var = np.zeros(len(noise_lvls) + 1)
    corr = np.zeros(len(noise_lvls) + 1)

    var[0] = np.trace(np.dot(Sigma, Sigma)) + np.trace(Sigma)**2
    corr[0] = 1.0
    for i in range(1, len(var)):
        Gamma = Sigma + noise_lvls[i-1]*np.eye(Sigma.shape[0])
        var[i] = np.trace(np.dot(Gamma, Gamma)) + np.trace(Gamma)**2
        corr[i] = np.sqrt(var[0]/var[i])

    return (var, corr)


def EMF(samps, a, n):
    S = np.cov(samps[0].T)
    for i in range(len(a)):
        S = S + a[i]*( np.cov(samps[i+1].T) - np.cov(samps[i+1][:n[i]].T)  )
    return S


def LEMF(samps, a, n, d):
    S = logm(np.cov(samps[0].T))
    for i in range(len(a)):
        S = S + a[i]*( logm(np.cov(samps[i+1].T)) - logm(np.cov(samps[i+1][:n[i]].T))  )
    return expm(S)


def truncated(samps, a, n, threshold):
    Y = EMF(samps, a, n)
    D, V = np.linalg.eig(Y)
    D1 = np.diag(np.maximum(D, threshold))
    if np.any(D <= 0.):
        negative = 1
    else:
        negative = 0
    Y1 = np.matmul(V, np.matmul(D1, V.T))
    return (Y1, negative)
