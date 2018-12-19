#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""ECCO v4 Python: mds_io

This module includes utility routines for loading binary files in the llc 13-tile native flat binary layout.  This layout is the default for MITgcm input and output for global setups using lat-lon-cap (llc) layout.  The llc layout is used for ECCO v4. 

.. _ecco_v4_py Documentation :
   https://github.com/ECCO-GROUP/ECCOv4-py
"""

from __future__ import division
import numpy as np
import matplotlib.pylab as plt
import xarray as xr
import time
from copy import deepcopy
import glob


#%%
def load_llc_compact_binary_to_tiles(fdir, fname, llc, skip=0, nk=1, filetype = '>f', 
                 less_output = False ):
    """

    This routine loads a field from a MITgcm mds binary file in the llc 13 tie layout

    Parameters
    ----------
    fdir : string
        A string with the directory of the binary file to open
    fname : string
        A string with the name of the binary file to open
    llc : int
        the size of the llc grid.  For ECCO v4, we use the llc90 domain so `llc` would be `90`
    skip : int
        the number of 2D slices (or records) to skip.  Records could be vertical levels of a 3D field, or different 2D fields, or both.
    nk : int
        number of 2D slices (or records) to load.  
    filetype: string
        the file type, default is big endian (>) 32 bit float (f)
        alternatively, ('<d') would be little endian (<) 64 bit float (d)
    less_output : boolean
        a debug flag.  True means print more to the screen, False means be
        quieter.  Default False
        
    Returns
    -------
    data_tiles_k
        the binary file contents organized into a 13 x nk x llc x llc`data_tiles_k`
        one llc x llc array for each of the 13 tiles and depth levels

    Raises
    ------
    IOError
        If the file is not found

    """
    
    data_compact = load_llc_compact_binary(fdir, fname, llc, skip, nk, filetype, less_output)

    # define a blank array
    if nk > 1:
        data_tiles_k = np.zeros((13, nk, llc, llc))
    else:
        data_tiles_k = np.zeros((13, llc, llc))

    
    len_rec = 13*llc*llc

    # go through each 2D slice (or record)
    for k in range(nk):

        #tmp = arr_k[len_rec*(k):len_rec*(k+1)]
        #arr = np.reshape(tmp,(13*llc, llc

        if nk == 1:
            tmp = data_compact
        else:
            tmp = data_compact[:,:,k]

        f1, f2, f3, f4, f5 = llc_compact_to_faces(tmp)

        if 1 == 0:
            plt.close('all')
            plt.imshow(f1, origin='lower')
            plt.figure()
            plt.imshow(f2, origin='lower')   
            plt.show()
            
            plt.figure()
            plt.imshow(f3, origin='lower')
            plt.show()
            
            plt.figure()
            plt.imshow(f4, origin='lower')
            plt.show()
            
            plt.figure()
            plt.imshow(f5, origin='lower')
            plt.show()
        

        data_tiles = llc_faces_to_tiles(f1, f2, f3, f4, f5, llc)

        
        if 1 == 0:
            plt.figure()
            plt.imshow(data_tiles[0], origin='lower')
            plt.figure()
            plt.imshow(data_tiles[1], origin='lower')
            plt.figure()
            plt.imshow(data_tiles[2], origin='lower')
      
        if nk == 1:
            #print ('nk = 1')
            #print np.max(data_tiles)
            data_tiles_k = data_tiles[:,:,:]
            
        else:
            #print ('k' , k)
            for tile in range(0,13):
                #print ('tile',tile)
                data_tiles_k[tile,k,:,:] = data_tiles[tile,:,:]
        
        
    # return the array
    return data_tiles_k


def load_llc_compact_binary(fdir, fname, llc=90, skip=0, nk=1, filetype = '>f', 
             less_output = False ):
    """

    This routine loads an MITgcm binary file in the lat-lon-cap 'compact' format
    The compact format has dimension 13*llc x llc x nk
    

    Parameters
    ----------
    fdir : string
        A string with the directory of the binary file to open
    fname : string
        A string with the name of the binary file to open
    llc : int
        the size of the llc grid.  For ECCO v4, we use the llc90 domain so `llc` would be `90`
    skip : int
        the number of 2D slices (or records) to skip.  Records could be vertical levels of a 3D field, or different 2D fields, or both.
    nk : int
        number of 2D slices (or records) to load.  
    filetype: string
        the file type, default is big endian (>) 32 bit float (f)
        alternatively, ('<d') would be little endian (<) 64 bit float (d)
    less_output : boolean
        a debug flag.  True means print more to the screen, False means be
        quieter.  Default False
        
    Returns
    -------
    data_compact
        the binary file contents organized into a 13*llc x llc x nk `arr_k`
        one 13*llc x llc array for each depth level

    Raises
    ------
    IOError
        If the file is not found

    """
    
    datafile = fdir + '/' + fname
    
    print 'loading ' + fname
    
    # check to see if file exists.    
    file = glob.glob(datafile)
    if len(file) == 0:
        raise IOError(fname + ' not found ')

    f = open(datafile, 'rb')
    dt = np.dtype(filetype)

    # skip ahead 'skip' number of 2D slices
    f.seek(llc*llc*13*skip*dt.itemsize)

    # read in 'nk' 2D slices (or records) from the mds file
    arr_k = np.fromfile(f, dtype=filetype, count=llc*llc*13*nk)
    
    f.close()

    if nk > 1:
        data_compact = np.reshape(arr_k,(13*llc, llc, nk))
    else:
        data_compact = np.reshape(arr_k,(13*llc, llc))

    # return the array
    return data_compact


#%%
def llc_compact_to_faces(data_compact):
    """
    This routine takes an array of size 13*llc x llc and splits into the 5 'faces'
    of the llc grid.  4 lat-lon approximately pieces and one Arctic 'cap'

    Parameters
    ----------
    data_compact : 
        An array of dimension  13*llc x llc 
        
    Returns
    -------
    f1, f2, f3, f4, f5
        the 'data_compact'  array split into five faces 

    """
    # The length of the second dimension equals 'llc'
    llc = data_compact.shape[1]

    f1 = data_compact[:3*llc,:]
    f2 = data_compact[3*llc:6*llc,:]
    f3 = data_compact[6*llc:7*llc,:]
    
    f4 = np.zeros((llc, 3*llc))

    for f in range(8,11):
        i1 = np.arange(0, llc)+(f-8)*llc
        i2 = np.arange(0,3*llc,3) + 7*llc + f -8
        f4[:,i1] = data_compact[i2,:]

    f5 = np.zeros((llc, 3*llc))

    for f in range(11,14):
        i1 = np.arange(0, llc)+(f-11)*llc
        i2 = np.arange(0,3*llc,3) + 10*llc + f -11
        f5[:,i1] = data_compact[i2,:]

    return f1, f2, f3, f4, f5


#%%
def llc_faces_to_tiles(f1, f2, f3, f4, f5):
    """

    This routine takes an array of size 13*llc x llc and splits into the 13 'tiles'
    of the llc grid.  The result is 12 lat-lon approximately pieces and one Arctic 'cap'

    Parameters
    ----------
    'f1','f2' ... 'f5' : faces of the llc
        
        
    Returns
    -------
    data_tiles
         an array of dimension [13 x llc x llc]

    """
    llc = len(f3)

    data_tiles = np.zeros((13, llc, llc))
    
    data_tiles[0,:] = f1[llc*0:llc*1,:]

    data_tiles[1,:] = f1[llc*1:llc*2,:]
    data_tiles[2,:] = f1[llc*2:,:]

    data_tiles[3,:] = f2[llc*0:llc*1,:]
    data_tiles[4,:] = f2[llc*1:llc*2,:]
    data_tiles[5,:] = f2[llc*2:,:]
    
    data_tiles[6,:] = f3

    data_tiles[7,:] = f4[:,llc*0:llc*1]
    data_tiles[8,:] = f4[:,llc*1:llc*2]
    data_tiles[9,:] = f4[:,llc*2:]
    
    data_tiles[10,:] = f5[:,llc*0:llc*1]
    data_tiles[11,:] = f5[:,llc*1:llc*2]
    data_tiles[12,:] = f5[:,llc*2:]

    return data_tiles



def llc_tiles_to_faces(data_tiles):
    """

    This routine takes an array of the 13 'tiles' from the lat-lon-cap grid
    and rearranges them to 5 faces, 4 of which are approximately lat-lon 
    and the fifth is the Arctic 'cap' 

    Parameters
    ----------
    data_tiles : the array with 13 tiles
        An array of dimension 13xllcNxllcN
        
    Returns
    -------
    [f1,f2,f3,f4,f5] = five faces

    - dimensions:
        f1,f2: 3*llc x llc
           f3: 3*llc x llc
        f4,f5: llc x 3*llc  
    
    """

    llc = len(data_tiles[1,:])

    f1 = np.zeros((3*llc, llc))
    f2 = np.zeros((3*llc, llc))
    f3 = np.zeros((llc, llc))
    f4 = np.zeros((llc, 3*llc))
    f5 = np.zeros((llc, 3*llc))

    
    f1[llc*0:llc*1,:] = data_tiles[0,:]

    f1[llc*1:llc*2,:] = data_tiles[1,:]
    f1[llc*2:,:] = data_tiles[2,:]

    f2[llc*0:llc*1,:] = data_tiles[3,:]
    f2[llc*1:llc*2,:] = data_tiles[4,:]
    f2[llc*2:,:]      = data_tiles[5,:]
    
    f3 = data_tiles[6,:]

    f4[:,llc*0:llc*1] = data_tiles[7,:]
    f4[:,llc*1:llc*2] = data_tiles[8,:]
    f4[:,llc*2:] = data_tiles[9,:]
    
    f5[:,llc*0:llc*1] = data_tiles[10,:]
    f5[:,llc*1:llc*2] = data_tiles[11,:]
    f5[:,llc*2:] = data_tiles[12,:]

    return f1, f2, f3, f4, f5


def llc_faces_to_compact(f1,f2, f3, f4, f5):

    llc = len(f3)
    data_compact = np.zeros((13*llc, llc))

    data_compact[:3*llc,:] = f1
    data_compact[3*llc:6*llc,:] = f2
    data_compact[6*llc:7*llc,:] = f3
    
    for f in range(8,11):
        i1 = np.arange(0, llc)+(f-8)*llc
        i2 = np.arange(0,3*llc,3) + 7*llc + f -8
        data_compact[i2,:] = f4[:,i1]

    for f in range(11,14):
        i1 = np.arange(0, llc)+(f-11)*llc
        i2 = np.arange(0,3*llc,3) + 10*llc + f -11
        data_compact[i2,:] = f5[:,i1]

    return data_compact