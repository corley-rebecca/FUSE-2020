#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 13:50:32 2020

@author: rebeccacorley
"""
 
import numpy as np
from astropy.io import fits
import glob
import matplotlib.pyplot as plt


#dark_frame = fits.open('/Users/rebeccacorley/Desktop/FUSE-2020/examples/HRS_Data/Dark-Frames/20190404-034616-DARK-600s-1.fit')

#dark_data = dark_frame[0].header

#exp = dark_data['EXPOSURE']

def createMasterBias():
    #create zero array

    sum_bias = np.zeros((2199, 2749)) 


    bias_frames = glob.glob('/Users/rebeccacorley/Desktop/FUSE-2020/examples/HRS_Data/Bias-Frames/*.fit')

    for frame in bias_frames:
        bias = fits.open(frame)
        bias_data = bias[0]
        bias_intensity = bias_data.data[:,:]
    
        sum_bias = sum_bias + bias_intensity
    
    avg_frame = sum_bias/len(bias_frames)
    #write master bias to fits file
    #print(avg_frame)

    #plt.imshow( np.log10(avg_frame), cmap="binary" )
    #plt.show()
    
    return avg_frame
    
def createMasterDark(master_bias_frame):
    """
    create a master dark frame from individual dark exposures 
    
    """
        #create zero array/image
    sum_dark = np.zeros((2199, 2749)) 

        #gather all Dark Frames
    dark_frames = glob.glob('/Users/rebeccacorley/Desktop/FUSE-2020/examples/HRS_Data/Dark-Frames/*.fit')

        #sum over all dark frames , normalized for exposure time
    for frame in dark_frames:
            #open image
        dark = fits.open(frame)
    
            #get header value for exposure time 
        exp_time = dark[0].header['EXPOSURE']
    
            #puts data into new array
        dark_data = dark[0]
    
    
            #add normalized data to sum array
        sum_dark += (dark_data.data - master_bias_frame)/exp_time
    
    avg_frame = sum_dark/(len(dark_frames))
        #print(avg_frame)

        
        
        ###Running the code####
  

#save to fits
        
    return avg_frame
    
master_bias = createMasterBias()
master_dark = createMasterDark(master_bias_frame = master_bias)

#print(master_dark)
plt.imshow(np.log10(1000*master_dark), cmap="binary")






