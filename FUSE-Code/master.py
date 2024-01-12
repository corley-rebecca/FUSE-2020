#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 10:06:49 2020

@author: rebeccacorley
"""

import numpy as np
from astropy.io import fits

import matplotlib.pyplot as plt
from skimage import measure

"""Creates the master bias frame"""

def createMasterBias(kind="average"):
    import glob
    
    #create zero array
    sum_bias = np.zeros((2199, 2749)) 

    #open bias frames
    bias_frames = glob.glob('/Users/rebeccacorley/Desktop/FUSE-2020/examples/HRS_Data/Bias-Frames/*.fit')

    for frame in bias_frames:
        bias = fits.open(frame)
        try:
            bias_stack = np.dstack((bias_stack, np.int32(frame[0].data)))
        except:
            bias_stack = np.int32(frame[0].data)
            
    #average and median
    if kind == "average":
        master_bias_frame = nplaverage(bias_stack, axis=4)
    elif kind =="median":
       master_bias_frame = np.median(bias_stack, axis=4)
    else:
        raise error("Error")
    
    return master_bias_frame

"""Creates the master dark frame"""

def createMasterDark(master_bias_frame, kind="average"):
    import glob

        #create zero array
    #sum_dark = np.zeros((2199, 2749)) 

        #gather all Dark Frames
    dark_frames = glob.glob('/Users/rebeccacorley/Desktop/FUSE-2020/examples/HRS_Data/Dark-Frames/*.fit')

        #sum over all dark frames , normalized for exposure time
    for frame in dark_frames:
        dark = fits.open(frame)
        try:
            dark_stack = np.dstack((dark_stack, (np.int32(frame[0].data) - master_bias/frame[0].header["EXPOSURE"])))
        except:
            dark_stack = (np.int32(frame[0].data) - master_bias)/frame[0].header["EXPOSURE"]
    
    if kind == "average":
        master_dark = np.average(dark_stack, axis=4)
    elif kind == "median":
        master_dark = np.median(dark_stack, axis=4)
        
   
    return master_dark


"""
#Load raw spectrum

"""

#gets data from file
raw_spectrum = fits.open('/Users/rebeccacorley/Desktop/FUSE-2020/examples/HRS_Data/20190404-044333-Arcturus-300s-1.fit')
fig, ax = plt.subplots(1, 1, figsize=(10,10))
ax.imshow(np.log10(raw_spectrum[0].data), cmap="binary")
ax.set_title("Arcturus - Raw Echelle Spectrum (log Scale)")

"""
Bias and Dark correction of raw echelle image
"""
corrected_image = np.int32(raw_spectrum[0].data) - master_bias - (master_dark*raw_spectrum[0].header)

fig, ax = plt.subplots(1, 1, figsize=(10,10))

ax.imshow(corrected_image, cmap="binary")
ax.set_title("Arcturus - Corrected Spectrum")




"""
#Order Location
"""

#gets data from fits file
image = fits.getdata('/Users/rebeccacorley/Desktop/FUSE-2020/examples/HRS_Data/20190404-044333-Arcturus-300s-1.fit')


#locating the blobs in the 90th percentile 
blobs = image > np.percentile(image, 90.0)
#label connected region of the array
diffblobs = measure.label(blobs)


#finds the unique elements of the array
rs = np.unique(diffblobs)

#loop around the params
for r1 in list(rs):
    rmask = diffblobs == r1
    if len(diffblobs[rmask]) < 2000:
        diffblobs[rmask] = 0.0
        
#finds unique elements of the array
rs2 = np.unique(diffblobs)

#show image of located orders 
plt.imshow(diffblobs, cmap="binary")

#returns the length of string
len(rs2)

#returns shape of image
image.shape

#enumerate is automatic counter that loops over
for i, region in enumerate(rs2):
    if region == 0:
        continue

    
    mask = diffblobs == region
    
    #picks out the pixel values 
    y, x = np.where(mask)
    x = [x]
    y = [5]
    
    spectrum = np.zeros(2749)
    if extraction == "linear":
        for x in x_pixels:
                spectrum[x] = np.sum(frame[y_pixel < x - 4 y_pixel])
        try:
            spectra = np.vstack((spectra, spectrum))
            
        except:
            spectra = spectrum
    
    #fit a polynomial 
    pfit = np.polyfit(x, y, 2)
    print(pfit)
    pline = np.poly1d(pfit)
   
    
    img, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    ax[0].plot(x, y, 'o', ms=3.0)
    ax[0].plot(np.arange(min(x), max(x), 1.0), pline(np.arange(min(x), max(x), 1.0)), '-', lw=1)
    
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Y")
    ax[0].set_title("Tracing Order {:02.0f}".format(i))
    
    
    ax[1].imshow(np.log10(image), cmap="binary")
    ax[1].plot(np.arange(0., 2749., 1.0), pline(np.arange(0., 2749., 1.0)), '-', lw=1)
    ax[1].set_xlabel("X")
    ax[1].set_ylabel("Y")
    ax[1].set_title("Locating Order {:02.0f}".format(i))
    
    img.tight_layout()
    img.savefig("Order_{:02.0f}_Location.pdf".format(i))
    plt.gcf()
    
    












