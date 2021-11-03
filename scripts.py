import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import pandas as pd
import astropy.units as u
from astropy.wcs import wcs
from astropy.io import fits
from scipy import ndimage
from matplotlib.gridspec import GridSpec
from tempfile import TemporaryFile
from scipy.ndimage import gaussian_filter as norm_kde


#This function creates a uniform gaussian 
def Gauss_dist(x, mu, sigma):
    G = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    return G / np.trapz(G, x)

def quantile(x, q, weights=None):
    """
    Compute (weighted) quantiles from an input set of samples.
    Parameters
    ----------
    x : `~numpy.ndarray` with shape (nsamps,)
        Input samples.
    q : `~numpy.ndarray` with shape (nquantiles,)
       The list of quantiles to compute from `[0., 1.]`.
    weights : `~numpy.ndarray` with shape (nsamps,), optional
        The associated weight from each sample.
    Returns
    -------
    quantiles : `~numpy.ndarray` with shape (nquantiles,)
        The weighted sample quantiles computed at `q`.
    """

    # Initial check.
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    # Quantile check.
    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("Quantiles must be between 0. and 1.")

    if weights is None:
        # If no weights provided, this simply calls `np.percentile`.
        return np.percentile(x, list(100.0 * q))
    else:
        # If weights are provided, compute the weighted quantiles.
        weights = np.atleast_1d(weights)
        if len(x) != len(weights):
            raise ValueError("Dimension mismatch: len(weights) != len(x).")
        idx = np.argsort(x)  # sort samples
        sw = weights[idx]  # sort weights
        cdf = np.cumsum(sw)[:-1]  # compute CDF
        cdf /= cdf[-1]  # normalize CDF
        cdf = np.append(0, cdf)  # ensure proper span
        quantiles = np.interp(q, cdf, x[idx]).tolist()
        return quantiles

def Get_posterior(sample, bins=300):
    weight = np.ones_like(sample)

    q = [0.5 - 0.5 * 0.999999426697, 0.5 + 0.5 * 0.999999426697]
    span = quantile(sample.T, q, weights=weight)

    s = 0.02
    n, b = np.histogram(sample, bins=bins, weights=weight,
                        range=np.sort(span))
    n = norm_kde(n, 10.)
    x0 = 0.5 * (b[1:] + b[:-1])
    y0 = n
    
    return x0, y0 / np.trapz(y0,x0)

def Scaling_factor(X, m, M):              #X = Counts, m = Vmin, M = Vmax
    """
    Our filter function
    ----------
    X : `~numpy.ndarray` The array we are trying to filter
    m : Our lower bound
    M : Our Upper bound
    -------
    """
    m_mask = np.zeros_like(X)
    M_mask = np.zeros_like(X)

    for i in range(len(X)):
        for ii in range(len(X[0])):
            if X[i][ii] <= m:
                m_mask[i][ii] = 1 
            
            if X[i][ii] >= M:
                M_mask[i][ii] = 1 
             
    scl_img =  np.arcsinh(X - m)/np.arcsinh(M - m)
            
    for i in range(len(X)):
        for ii in range(len(X[0])):
            if m_mask[i][ii] == 1:
                scl_img[i][ii] = 0
            if M_mask[i][ii] == 1:
                scl_img[i][ii] = 1
            
    return scl_img