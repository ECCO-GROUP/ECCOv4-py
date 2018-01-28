#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 11:06:48 2018

@author: ifenty
"""
from __future__ import division
import numpy as np
import matplotlib.pylab as plt
import xarray as xr
import time
from copy import deepcopy
import glob

#%%
def load_llc_mds(fdir, fname, llc, **kwargs):

    #%%
    # construct the netcdf file name based on the variable name and the
    # the tile index
    datafile = fdir + '/' + fname
    
    print 'loading ' + fname
    
    
    # check to see if file exists.    
    file = glob.glob(fname)
    if len(file) == 0:
        raise IOError(fname + ' not found ')


    f = open(datafile, 'rb')
    arr = np.fromfile(f, dtype='>f', count=-1)#.reshape(recshape)[levinds]
    f.close()
    
    arr = np.reshape(arr,(llc*13, llc))
    
    f1 = arr[:3*llc,:]
    f2 = arr[3*llc:6*llc,:]
    f3 = arr[6*llc:7*llc,:]
    
    f4 = np.zeros((llc, 3*llc))

    for f in range(8,11):
        i1 = np.arange(0, llc)+(f-8)*llc
        i2 = np.arange(0,3*llc,3) + 7*llc + f -8
        f4[:,i1] = arr[i2,:]

    f5 = np.zeros((llc, 3*llc))

    for f in range(11,14):
        i1 = np.arange(0, llc)+(f-11)*llc
        i2 = np.arange(0,3*llc,3) + 10*llc + f -11
        f5[:,i1] = arr[i2,:]
    
    if 1 == 0:
        plt.close('all')
        plt.imshow(f1, origin='lower')
        plt.figure()
        plt.imshow(f2, origin='lower')   
        plt.figure()
        plt.imshow(f3, origin='lower')
        
        plt.figure()
        plt.imshow(f4, origin='lower')
        
        plt.figure()
        plt.imshow(f5, origin='lower')
    
    arr_tiles = np.zeros((13, 90, 90))
    arr_tiles[0,:] = f1[llc*0:llc*1,:]
    arr_tiles[1,:] = f1[llc*1:llc*2,:]
    arr_tiles[2,:] = f1[llc*2:,:]

    arr_tiles[3,:] = f2[llc*0:llc*1,:]
    arr_tiles[4,:] = f2[llc*1:llc*2,:]
    arr_tiles[5,:] = f2[llc*2:,:]
    
    arr_tiles[6,:] = f3

    arr_tiles[7,:] = f4[:,llc*0:llc*1]
    arr_tiles[8,:] = f4[:,llc*1:llc*2]
    arr_tiles[9,:] = f4[:,llc*2:]
    
    arr_tiles[10,:] = f5[:,llc*0:llc*1]
    arr_tiles[11,:] = f5[:,llc*1:llc*2]
    arr_tiles[12,:] = f5[:,llc*2:]
    
    if 1 == 0:
        plt.figure()
        plt.imshow(arr_tiles[0], origin='lower')
        plt.figure()
        plt.imshow(arr_tiles[1], origin='lower')
        plt.figure()
        plt.imshow(arr_tiles[2], origin='lower')
    
    return arr_tiles