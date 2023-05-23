import numpy as np
from scipy.stats import multivariate_normal as mvnrnd

import pandas as pd
from tqdm import tqdm

from spd import *
from est import *


def drawSamples(n, Sigma, noise_lvls):
    """
    n should be a vector of increasing ints
    noise_lvls vector of increasing floats

    Returns a list of samples
    y, y^(1), ... , y^(L)  # y^(L) is lowest fidelity
    """
    assert len(noise_lvls)+1 == len(n)
    d = Sigma.shape[0]
    y = mvnrnd.rvs(size=np.max(n), mean=np.zeros(d), cov=Sigma)
    samps = [ y[:n[0]] ]  #High fidelity samples
    for i in range(len(noise_lvls)):
        eps = mvnrnd.rvs(size=n[i+1], mean=np.zeros(d), cov=noise_lvls[i]*np.eye(d))
        samps.append( y[:n[i+1]] + eps )
    return samps


# Sample a d-by-d SPD matrix
np.random.seed(1000) # reproducible
def sampleSPD(d):
    S = np.random.randn(d*d).reshape(d,d)
    return np.dot(S.T, S)

d = 4
Sigma = sampleSPD(d)


print('True covariance = \n')
print('\\begin{bmatrix}')
for i in range(d):
    str = ''
    for j in range(d):
        str = str + f"{Sigma[i,j]:.2f} &"
    str = str[:-2]
    print(str + '\\\\')
print("\\end{bmatrix}")


ntrials = 100

estimator_list = ['HF', 'LF', 'EMF', 'LEMF', 'TruncatedMF']

mse_frob = np.zeros((ntrials, len(estimator_list)))
mse_aff = np.zeros((ntrials, len(estimator_list)))
mse_logE = np.zeros((ntrials, len(estimator_list)))
neg_mf = np.zeros(ntrials)

B = 15

noise_lvls = np.array([0.1, 0.5, 1.])
costs = np.array([1, 1e-2, 1e-3, 1e-4])

var, corr = computeVarCorr(Sigma, noise_lvls)
print('\n')
print("sample variances", var)
print("sample correlations", corr)

a, n = sampleAlloc(B, costs, var, corr)
n = np.floor(n).astype(int)
print("sample vector", n)

n_high = np.floor(B/costs[0]).astype(int)
n_low = np.floor(B/costs[-1]).astype(int)
print("number of high-fidelity only", n_high)
print("number of low-fidelity only", n_low)

for i in tqdm(range(ntrials)):

    # Draw samples for HF
    y = mvnrnd.rvs(size=n_high, mean=np.zeros(d), cov=Sigma)
    S = np.cov(y.T)

    mse_frob[i,0] = frobDist(S, Sigma)**2
    mse_aff[i,0] = affineInvDist(S, Sigma)**2
    mse_logE[i,0] = logEuclideanDist(S, Sigma)**2

    # Draw samples for LF
    y = mvnrnd.rvs(size=n_low, mean=np.zeros(d), cov=Sigma)
    eps = mvnrnd.rvs(size=n_low, mean=np.zeros(d), cov=noise_lvls[-1]*np.eye(d))
    y = y + eps
    S = np.cov(y.T)

    mse_frob[i,1] = frobDist(S, Sigma)**2
    mse_aff[i,1] = affineInvDist(S, Sigma)**2
    mse_logE[i,1] = logEuclideanDist(S, Sigma)**2

    # Draw samples for MF estimators
    samps = drawSamples(n, Sigma, noise_lvls)

    S, neg = truncated(samps, a, n, 1e-16)
    S = makePositive(S)
    neg_mf[i] = neg

    mse_frob[i,4] = frobDist(S, Sigma)**2
    mse_aff[i,4] = affineInvDist(S, Sigma)**2
    mse_logE[i,4] = logEuclideanDist(S, Sigma)**2


    # Compute Euclidean
    S = EMF(samps, a, n)

    mse_frob[i,2] = frobDist(S, Sigma)**2
    if not neg:
        mse_aff[i,2] = affineInvDist(S, Sigma)**2
        mse_logE[i,2] = logEuclideanDist(S, Sigma)**2
    else:
        mse_aff[i,2] = np.nan
        mse_logE[i,2] = np.nan


    S = LEMF(samps, a, n, d)

    mse_frob[i,3] = frobDist(S, Sigma)**2
    mse_aff[i,3] = affineInvDist(S, Sigma)**2
    mse_logE[i,3] = logEuclideanDist(S, Sigma)**2


m1 = np.mean(mse_frob, axis=0)
m2 = np.mean(mse_logE, axis=0)
m3 = np.mean(mse_aff, axis=0)
print("percent indefinite {:0.2f}".format(100*np.mean(neg_mf)))

print("")
print("\\begin{tabular}{|c|r|r|r|r|r|}")
print("\hline")
print("MSE & high-fidelity only & surrogate only & Euclidean MF & truncated MF & LEMF (ours)\\\\")
print("\hline")
print("log-Euclidean & " + f"{m2[0]:.2f} & " + f"{m2[1]:.2f} &" + f"{m2[2]:.2f} & " + f"{m2[4]:.2f} &" + f"\\bf {m2[3]:.2f} \\\\")
print("affine-invariant & " + f"{m3[0]:.2f} & " + f"{m3[1]:.2f} &" + f"{m3[2]:.2f} & " + f"{m3[4]:.2f} &" + f"\\bf {m3[3]:.2f} \\\\")
print("Euclidean & " + f"{m1[0]:.2f} & " + f"{m1[1]:.2f} &" + f"{m1[2]:.2f} & " + f"{m1[4]:.2f} &" + f"\\bf {m1[3]:.2f} \\\\")
print("\hline")
print('\end{tabular}')
print("")

print(pd.DataFrame(columns=estimator_list, data=np.vstack([m2, m3, m1]), index=['logE', 'aff', 'Frob']))
