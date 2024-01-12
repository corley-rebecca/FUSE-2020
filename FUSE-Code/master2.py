#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 13:01:08 2020

@author: rebeccacorley
"""

import numpy as np
from astropy.io import fits as fits
import matplotlib.pyplot as plt
from skimage import measure
import glob

"""
Step 1: Creates the master bias frame
"""

def createMasterBias(kind="average"):
   
    
    #create zero array
    #sum_bias = np.zeros((2199, 2749)) 

    #open bias frames
    bias_frames = glob.glob('/Users/rebeccacorley/Desktop/FUSE-2020/examples/HRS_Data/Bias-Frames/*.fit')   
    print (bias_frames)
    
    for frame in bias_frames:
        bias = fits.open(frame)
        bias = bias[0].data.astype(np.int32)
        try:
            bias_stack = np.dstack((bias_stack, bias))
        except:
            bias_stack = bias
            
    #average and median
    if kind == "average":
        master_bias_frame = np.average(bias_stack, axis=2)
    elif kind == "median":
        master_bias_frame = np.median(bias_stack, axis=2)
    else:
        raise ValueError("Error")
        
    fig, ax = plt.subplots(1, 1, figsize=(10,10))

    ax.imshow(np.log10(master_bias_frame), cmap="binary")
    ax.set_title("Master Bias Frame")
    
    plt.savefig("bfix.pdf")
    plt.show()
    
    return master_bias_frame

#createMasterBias()
masterbias = createMasterBias()

"""
Step 2: Creates the master dark frame
"""

def createMasterDark(master_bias_frame, kind="average"):
   

        #create zero array
    #sum_dark = np.zeros((2199, 2749)) 

        #gather all Dark Frames
    dark_frames = glob.glob('/Users/rebeccacorley/Desktop/FUSE-2020/examples/HRS_Data/Dark-Frames/*.fit')
    print (dark_frames)
        #sum over all dark frames , normalized for exposure time
    for frame in dark_frames:
        dark = fits.open(frame)
        #dark = dark[0].data.astype(np.int32)
        try:
            dark_stack = np.dstack((dark_stack, (np.int32(dark[0].data) - master_bias_frame)/dark[0].header["EXPOSURE"]))
        except:
            dark_stack = (np.int32(dark[0].data) - master_bias_frame)/dark[0].header["EXPOSURE"]
    
    if kind == "average":
        master_dark = np.average(dark_stack, axis=2)
    elif kind == "median":
        master_dark = np.median(dark_stack, axis=2)
    else:
        raise ValueError("Error")
        
    fig, ax = plt.subplots(1, 1, figsize=(10,10))

    ax.imshow(np.log10(master_dark), cmap="binary")
    ax.set_title("Master Dark Frame")
    
    plt.savefig("dfix.pdf")
    plt.show()
        
   
    return master_dark

createMasterDark(masterbias)


"""
Step 3: Load raw data
"""

raw_spectrum = fits.open('/Users/rebeccacorley/Desktop/FUSE-2020/examples/HRS_Data/20190404-044333-Arcturus-300s-1.fit')

fig, ax = plt.subplots(1, 1, figsize=(10,10))

ax.imshow(np.log10(raw_spectrum[0].data), cmap="binary")
ax.set_title("Raw Sepectrum - Arcturus (log scale)")




"""
Step 4: Correct raw spectrum for Bias and Dark
"""
def createCorrectedImg(master_bias_frame, master_dark, raw_spectrum):
    
    
    #gets data from file
    #bdfix = glob.glob('/Users/rebeccacorley/Desktop/FUSE-2020/examples/HRS_Data/20190404-044333-Arcturus-300s-1.fit')
    
    bdfix = np.int32(raw_spectrum[0].data) - master_bias_frame - (master_dark*raw_spectrum[0].header["EXPOSURE"])

    fig, ax = plt.subplots(1, 1, figsize=(10,10))

    ax.imshow(np.log10(bdfix), cmap="binary")
    ax.set_title("Arcturus - Corrected Spectrum")
    
    plt.savefig("bdfix.pdf")
    plt.show()
    
    return bdfix 

masterbias = createMasterBias()
masterdark = createMasterDark(masterbias)
correctedimg = createCorrectedImg(masterbias, masterdark, raw_spectrum)

"""
Step 5: Order Location

def createorderLocation():

    #gets data from fits file
    image = fits.getdata('/Users/rebeccacorley/Desktop/FUSE-2020/examples/HRS_Data/20190404-044333-Arcturus-300s-1.fit')

    #plots raw echelle data on logscale
    plt.imshow(np.log10(image), cmap = "binary")

    #locating the blobs in the 90th percentile 
    blobs = image > np.percentile(image, 90.0)
    #label connected region of the array
    dblobs = measure.label(blobs)


    #finds the unique elements of the array
    rs = np.unique(dblobs)

    #loop around the params
    for r1 in list(rs):
        rmask = dblobs == r1
        if len(dblobs[rmask]) < 2000:
            dblobs[rmask] = 0.0
        
        #finds unique elements of the array
        rs2 = np.unique(dblobs)

    #show image of located orders 
    plt.imshow(dblobs, cmap="binary")

    #returns the length of string
    len(rs2)

    #returns shape of image
    image.shape


    #enumerate is automatic counter that loops over
    for i, region in enumerate(rs2):
        if region == 0:
            continue

    
        mask = dblobs == region
    
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
        

orderlocation = createorderLocation()

"""

"""
Step 6: Create master Flat Field image
""" 

def createMasterFlat(bdfix, master_bias_frame, master_dark, kind="average"):
    
    flat_frame = glob.glob('/Users/rebeccacorley/Desktop/FUSE-2020/examples/20200715/Flats/*.fits')
    print (flat_frame)
    
    for flat in flat_frame:
        frame = fits.open(flat)
        
        exp = frame[0].header["EXPOSURE"]
        fixed = np.int32(frame[0].data) - master_bias_frame - (master_dark*exp)
        
        try:
            flat_stack = np.dstack((flat_stack, fixed))
        except:
            flat_stack = fixed
            
    if kind == "average":
        master_flat = np.average(flat_stack, axis = 2)
    elif kind == "median":
        master_flat = np.median(flat_stack, axis = 2)
    
    else:
        raise ValueError("Error")
        
    fig, ax = plt.subplots(1, 1, figsize=(10,10))

    ax.imshow(np.log10(master_flat), cmap="binary")
    ax.set_title("Master Flat Frame")
    
    plt.savefig("ffix.pdf")
    plt.show()
        
    return master_flat

    fig, ax = plt.subplots(1, 1, cmap="binary")    

masterflat = createMasterFlat(masterbias, masterdark, correctedimg)    
    


"""
Step 7: Extract Orders from Flat

def createExtractOrders(masterflat, masterbias, masterdark, correctedimg, kind = "median"):
    
    plt.imshow(np.log10(masterflat), cmap = "binary")
     
    #locating the blobs in the 90th percentile 
    blobs = masterflat > np.percentile(masterflat, 90.0)
    #label connected region of the array
    dblobs = measure.label(blobs)


    #finds the unique elements of the array
    rs = np.unique(dblobs)

    #loop around the params
    for r1 in list(rs):
        rmask = dblobs == r1
        if len(dblobs[rmask]) < 2000:
            dblobs[rmask] = 0.0
        
        #finds unique elements of the array
        rs2 = np.unique(dblobs)
        
        #show image of located orders 
    plt.imshow(dblobs, cmap="binary")

    #returns the length of string
    len(rs2)

    #returns shape of image
    masterflat.shape
        
"""
    
  





"""
Step 8: Blaze Fit
"""
#def blazefit():  





"""
Step 9: Remove Blaze (from flat) and Normalize 
"""






"""
Step 10: Flat Field And Bias Correction of Extracted Orders
""" 





"""
Step 11: Wavelength Calibration 
"""

wavelength_cal = fits.open('/Users/rebeccacorley/Desktop/FUSE-2020/examples/20200715/20200715-195940-THAR-1s-2.fits')

fig, ax = plt.subplots(1, 1, figsize=(10,10))

ax.imshow(np.log10(wavelength_cal[0].data), cmap="binary")
ax.set_title("Thorium-Argon Spectrum")


    
    
