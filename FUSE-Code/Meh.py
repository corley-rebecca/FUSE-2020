#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 17:49:32 2020

@author: rebeccacorley
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


#from astropy.utils.data import get_pkg_data_filename


hdu_list = fits.open('/Users/rebeccacorley/Desktop/FUSE-2020/examples/HRS_Data/20190404-044333-Arcturus-300s-1.fit')

hdu_list.info() #summarizes content in file

# hdu_list is an array with N elements, where N is the number of HDUs.
#Here N = 1
hdu = hdu_list[0]

hdu.header


#print(hdu_list)
#plt.imshow(hdu.data, cmap= "binary")
plt.imshow( np.log10(hdu.data), cmap="binary" )

plt.title("Echelle Spectrum ")
#plt.xlabel("X")
#plt.ylabel("Y")
#cmap = "binary"