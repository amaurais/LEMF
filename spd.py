# Define basic functions
import numpy as np
from scipy.linalg import expm, logm


def Log(X, A):
    logAX, _ = logm( np.linalg.solve(A, X), disp=False )
    S = np.matmul( A, np.real(logAX) )
    return 0.5*(S + np.transpose(S))


def Log1d(x, a):
    return a*np.log(x/a)


def Exp(X, A):
    S = np.matmul( A, np.real(expm( np.linalg.solve(A, X) )) )
    return 0.5*(S + np.transpose(S))


def Exp1d(x, a):
    return a*np.exp(x/a)


def affineInvDist(X, Y):
    """
    Independent of the location of the tangent space.
    """
    S, _ = logm(np.linalg.solve(Y, X), disp=False)
    return np.real( np.sqrt( np.trace( np.matmul(np.transpose(S), S)) ) )


def affineInvDist1d(x, y):
    return np.abs(np.log(x) - np.log(y))


def frobDist1d(x, y):
    return np.abs(x - y)


def frobDist(A, B):
    return np.linalg.norm(A - B, ord = 'fro')


def logEuclideanDist(X, Y):
    logX, _ = logm(X, disp=False)
    logY, _ = logm(Y, disp=False)
    return frobDist(logX, logY)


def makePositive(Y):
    D, V = np.linalg.eig(Y)
    D1 = np.diag(abs(D))
    Y1 = np.matmul(V, np.matmul(D1, V.T))
    return Y1
