#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 13:05:14 2020

@author: rebeccacorley
"""

#This is to practice using astropy data to appear as an image

#Required libraries
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


#opens file
hdu_list = fits.open('/Users/rebeccacorley/Desktop/FUSE-2020/examples/HRS_Data/20190404-044333-Arcturus-300s-1.fit')


hdu_list.info() #summarizes content in file

# hdu_list is an array with N elements, where N is the number of HDUs.
#Here N = 1
hdu = hdu_list[0]

hdu.header


#display image. Note: it is in log scale 
plt.imshow( np.log10(hdu.data), cmap="binary" )

#plt.plot(hdu.data[:, 1500])




#Changes title, x and y axis labels 
plt.title("eShel Spectrum -- Arcturus -- log-scale")
plt.xlabel("X")
plt.ylabel("Y")














    
    