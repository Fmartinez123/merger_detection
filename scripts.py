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
import os


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

def scaling_factor(X, m, M):              #X = Counts, m = Vmin, M = Vmax
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

def plot_fig(Total,Halpha,Orig,SegMap,Zoom,Table,Title):
    #Plotting the Figures
    if Total == 2:
        Labels=['Brightest Pixel','x','y','Max Value']
    
        plt.figure(figsize=[17,7])
        gs = GridSpec(2,4, wspace = -.499, hspace=0)
    
        plt.subplot(gs[0,1])    #Halpha Image
        plt.title('H alpha (W/ Median Filter)',fontsize = 13)
        plt.imshow(Halpha)

        plt.subplot(gs[0,3])    #Original Image
        plt.title('Original',fontsize = 13)
        plt.imshow(Orig,cmap='binary')
    
        ax = plt.subplot(gs[1,1:])   #Tabel Below
        table = ax.table(Table,colLabels=Labels,loc='center')
        table.set_fontsize(14)
        table.scale(1,2)
        ax.axis('off')
    
        plt.suptitle(Title,x=.57,y=.97,fontsize=15)

    if Total == 3:
        Labels=['Brightest Pixel','x','y','Max Value']
    
        plt.figure(figsize=[17,7])
        gs = GridSpec(2,6, wspace = -.499, hspace=0)
    
        plt.subplot(gs[0,1])    #Halpha Image
        plt.title('H alpha (W/ Median Filter)',fontsize = 13)
        plt.imshow(Halpha)

        plt.subplot(gs[0,3])    #Original Image
        plt.title('Original',fontsize = 13)
        plt.imshow(Orig,cmap='binary')
    
        plt.subplot(gs[0,5])    #Seg Map Image
        plt.title('Segmentation Mask',fontsize = 13)
        plt.imshow(SegMap)
    
        ax = plt.subplot(gs[1,1:])   #Tabel Below
        table = ax.table(Table,colLabels=Labels,loc='center')
        table.set_fontsize(14)
        table.scale(1,2)
        ax.axis('off')
    
        plt.suptitle(Title,x=.57,y=.97,fontsize=15)
    if Total == 4:
        Labels=['Brightest Pixel','x','y','Max Value']
    
        plt.figure(figsize=[17,7])
        gs = GridSpec(2,8, wspace = -.499, hspace=0)
    
        plt.subplot(gs[0,1])    #Halpha Image
        plt.title('H alpha (W/ Median Filter)',fontsize = 13)
        plt.imshow(Halpha)

        plt.subplot(gs[0,3])    #Original Image
        plt.title('Original',fontsize = 13)
        plt.imshow(Orig,cmap='binary')

        plt.subplot(gs[0,5])    #Seg Map Image
        plt.title('Segmentation Mask',fontsize = 13)
        plt.imshow(SegMap)

        plt.subplot(gs[0,7])    #Zoomed Out Image
        plt.title('Zoomed Out',fontsize = 13)
        plt.imshow(Zoom)
    
        ax = plt.subplot(gs[1,1:])   #Tabel Below
        table = ax.table(Table,colLabels=Labels,loc='center')
        table.set_fontsize(14)
        table.scale(1,2)
        ax.axis('off')
    
        plt.suptitle(Title,x=.57,y=.97,fontsize=15)
        
def sig_1(Normalized_data):
    """
    Doing a 1 sigma Test
    ----------
    Normalized_data : The normalized curve of our data after subtracting out random noise.
    (see Randomized H-alpha Median Filter for more context)
    
    Returns: [low,high]
    -------
    """
    xrange = np.linspace(-30,30, 1000)
    
    summed_x = np.cumsum(Normalized_data / np.trapz(Normalized_data,xrange))/max(np.cumsum(Normalized_data / np.trapz(Normalized_data,xrange)))
    
    low_bound = .16
    high_bound = .84

    #finding the intersections between the lines
    low_3 = np.argwhere(np.diff(np.sign(summed_x - low_bound))).flatten()
    high_3 = np.argwhere(np.diff(np.sign(summed_x - high_bound))).flatten()
    
    intercept = [round(xrange[low_3[0]],1),round(xrange[high_3[0]],1)]

    #print("x intercepts for the offset are: " + str(round(xrange[low_3[0]],1)) + " for the lower bound and: "
    #  + str(round(xrange[high_3[0]],1)) + " for the upper bound for a 1 sigma test.")

    
    return(intercept)


def sig_3(Normalized_data):
    """
    Doing a 1 sigma Test
    ----------
    Normalized_data : The normalized curve of our data after subtracting out random noise.
    (see Randomized H-alpha Median Filter for more context)
    
    Returns: [low,high]
    -------
    """
    xrange = np.linspace(-30,30, 1000)
    
    summed_x = np.cumsum(Normalized_data / np.trapz(Normalized_data,xrange))/max(np.cumsum(Normalized_data / np.trapz(Normalized_data,xrange)))
    
    low_bound = .0015
    high_bound = .9985

    #finding the intersections between the lines
    low_3 = np.argwhere(np.diff(np.sign(summed_x - low_bound))).flatten()
    high_3 = np.argwhere(np.diff(np.sign(summed_x - high_bound))).flatten()
    
    intercept = [round(xrange[low_3[0]],1),round(xrange[high_3[0]],1)]

    #print("x intercepts for the offset are: " + str(round(xrange[low_3[0]],1)) + " for the lower bound and: "
    #  + str(round(xrange[high_3[0]],1)) + " for the upper bound for a 3 sigma test.")

    
    return(intercept)

def Median_filter(Data,Data_string,sigma,median_size):
    """
    Doing a Median filter test:
    ----------
    This test will take H-alpha data from our list of galaxies [Data] (Downloaded separately) and find the highest value pixel from this data set and subtract the x and y positions from the highest value pixel of our Original image data. The goal is to find the offset of H-alpha from the original image, and see if this points us towards any interesting nearby objects. We will need the sementation fits file from the field we will be observing to conduct this. We will also need to run our data through the normalization test before we conduct this function.
    ----------
    Data : A numpy array with the id's of our galaxies we want to iterate over separated up by fields.
    Data_string: A numpy array with the field listed as a string (ex. ['GN1','ERSPRIME'] etc.)
    sigma: What sigma test we wish to use, type in the whole number; 1 for a 1 sigma test etc
    median_size: The nxn size of the median filter we wish to use.
    
    Returns: [x_offset,y_offset]
    -------
    """
    
    # setting up the pathing
    HOME_PATH = '/Users/felix/Research/merger_detection'
    os.chdir(HOME_PATH)
    
    img_N = fits.open(HOME_PATH+'/CANDLES_data/goodsn_3dhst_v4.0_f160w/goodsn_3dhst.v4.0.F160W_orig_sci.fits')[0].data
    seg_N = fits.open(HOME_PATH+'/CANDLES_data/goodsn_3dhst_v4.0_f160w/goodsn_3dhst.v4.0.F160W_seg.fits')[0].data
    segN_fits = HOME_PATH+'/CANDLES_data/goodsn_3dhst_v4.0_f160w/goodsn_3dhst.v4.0.F160W_seg.fits'
    w = wcs.WCS(segN_fits)

    corrected_x = np.load(HOME_PATH+'/Rand_Ha/corrected_x_median.npy')
    corrected_y = np.load(HOME_PATH+'/Rand_Ha/corrected_y_median.npy')
    
    if sigma == 1:
        low_x, high_x = sig_1(corrected_x)
        low_y, high_y = sig_1(corrected_y)
        print("x intercepts for the X offset are: " + str(low_x) + " for the lower bound and: " + str(high_x) + " for the upper bound for a 1 sigma test.")
        print("x intercepts for the Y offset are: " + str(low_y) + " for the lower bound and: " + str(high_y) + " for the upper bound for a 1 sigma test.")
        pass
    elif sigma == 3:
        low_x, high_x = sig_3(corrected_x)
        low_y, high_y = sig_3(corrected_y)
        print("x intercepts for the X offset are: " + str(low_x) + " for the lower bound and: " + str(high_x) + " for the upper bound for a 3 sigma test.")
        print("x intercepts for the Y offset are: " + str(low_y) + " for the lower bound and: " + str(high_y) + " for the upper bound for a 3 sigma test.")
        pass
    else:
        return(print('Incorrect sigma, please choose sigma 1 or sigma 3'))

    x_off,y_off = [],[]

    #Listing all the Arrays for Each Section i.e. GN1
    RealData = []
    Mask = []
    im_med = []
    Halph = []
    Orig = []
    Title_fancy = []
    Title_save = []
    Ha_x = []
    Ha_y = []
    Orig_x = []
    Orig_y = []
    
    for j in range(len(Data)):
        GID = Data[j]
        field = Data_string[0]
        
    #Opening all the Fits Files
        H_a = fits.open(HOME_PATH+'/merger_candidates/{}/{}_{}.full.fits'.format(field,field,GID))
        Ha_fits = HOME_PATH+'/merger_candidates/{}/{}_{}.full.fits'.format(field,field,GID)
        Line = H_a['Line','Ha'].data     #Halpha data
        Continuum = H_a['Continuum','Ha'].data
        Contam = H_a['Contam','Ha'].data
        Raw = H_a['DSCI','F105W'].data
        
    #Cropping and Correcting the data
        Halph.append(Line[59:100,59:100] - Continuum[59:100,59:100] - Contam[59:100,59:100])  #Fixing the Errors on Halpha
        Orig.append(Raw[59:100,59:100])
        Title_fancy.append(field + ' ' + str(GID))
        Title_save.append(field + '_'+str(GID))
        
    #X and Y Posistions of Galaxy on Seg Map
        Y,X = np.where(seg_N == GID)
        Z = np.array([Y,X]).T     #Gives the X and Y positions of Object in Y,X
        #world = w.wcs_pix2world(Z,1)  #Gives RA and Dec Values of Object (May not be needed)
        
    #Center Pixel of Ha on Seg Map
        RA = H_a[0].header['RA']
        DEC = H_a[0].header['DEC']

        center = w.wcs_world2pix([[RA,DEC]],1)   #Finds the center pixels of the central RA and Dec of the Ha cutout
        hy = np.round(center[0][0])
        hx = np.round(center[0][1])
        center = np.array([int(hx),int(hy)])

    #Making a Mask that is the Galaxy on Seg Map
        seg_mask = np.array(seg_N[center[0]-20:center[0]+21,center[1]-20:center[1]+21])
        seg_mask[seg_mask != GID] = 0
        seg_mask[seg_mask == GID] = 1
    
        Mask.append(np.array(seg_mask))
        
        Ha_img = Halph[j]
        Orig_img = Orig[j]
        
    #Applying the Segmentation Map Mask
        Ha_img[Mask[j] == 0]=0
        Orig_img[Mask[j] == 0]=0
        
    #Applying the Filter
        m=np.percentile(Ha_img,1.5)   #High Pass
        M=np.percentile(Ha_img,98.5)  #Low Pass
        if M < np.abs(m)*6:
            M = np.abs(m)*6
        RealData.append(scaling_factor(Ha_img, m, M))
        im_med.append(ndimage.median_filter(RealData[j], size = median_size))  #Median Smoothing
        
    #Finding the Position of the Brightest Pixel and its value
        Ha_position = np.argwhere(im_med[j] == im_med[j].max()) * u.pixel
        Ha_max_value = np.amax(im_med[j])

        Orig_position = np.argwhere(Orig_img == Orig_img.max()) * u.pixel
        Orig_max_value = np.amax(Orig_img)
        
    #Appending the x and y for the Histogram
        Ha_x.append(Ha_position[0][1])
        Ha_y.append(Ha_position[0][0])
        Orig_x.append(Orig_position[0][1])
        Orig_y.append(Orig_position[0][0])
        
    #Filtering out Galaxies that are not within the inner 68%
        diff_x = Ha_x[j].value - Orig_x[j].value
        diff_y = Ha_y[j].value - Orig_y[j].value
        
        if((low_x <= diff_x <= high_x) and (low_y <= diff_y <= high_y)):
                
    #Appending the x and y offset Separetly by Section
            x_off.append(Ha_x[j].value - Orig_x[j].value)
            y_off.append(Ha_y[j].value - Orig_y[j].value)

        else:
            pass
    
    print(field + ' is complete.')
    return(x_off,y_off)
