#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 11:57:00 2020

@author: rebeccacorley
"""

#Required libraries
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from skimage import measure

#gets data from fits file
image = fits.getdata('/Users/rebeccacorley/Desktop/FUSE-2020/examples/HRS_Data/20190404-044333-Arcturus-300s-1.fit')

#plots raw echelle data on logscale
plt.imshow(np.log10(image), cmap = "binary")

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
    
    
    
    
    
    
    

