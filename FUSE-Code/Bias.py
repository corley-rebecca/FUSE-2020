#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 12:33:50 2020

@author: rebeccacorley
"""

#This function is to make a "big" bias frame  

#import astropy
import numpy as np
from astropy.io import fits
import glob
import matplotlib.pyplot as plt
"""
bias_1 = fits.open('/Users/rebeccacorley/Desktop/FUSE-2020/examples/HRS_Data/Bias-Frames/20190404-034337-BIAS-1.fit')

bias_data = bias_1[0]

bias1 = bias_data.data[:,:]

print((bias1.shape))


#bias_1_mean = np.mean(bias_1)
"""

#create zero array

sum_bias = np.zeros((2199, 2749)) 


bias_frames = glob.glob('/Users/rebeccacorley/Desktop/FUSE-2020/examples/HRS_Data/Bias-Frames/*.fit')

for frame in bias_frames:
    bias = fits.open(frame)
    bias_data = bias[0]
    bias_intensity = bias_data.data[:,:]
    
    sum_bias = sum_bias + bias_intensity
    
avg_frame = sum_bias/len(bias_frames)
print(avg_frame)

plt.imshow( np.log10(avg_frame), cmap="binary" )
plt.show()









        #take pixel 0,0 from each image use mean not median
        #take all bias images and add to find average value of each pixel
        #we care about variation across detector...certain pixels have dif zero points that flux around some mean
        #should look like original bias frame
        #overscan is differnt columns that are never exposed...treat as some estimate of bias
        #dstack for median which is a big array
        