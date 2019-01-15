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
def load_binary_array(fdir, fname, ni, nj, nk=1, skip=0, filetype = '>f', 
                      less_output = False ):
    """

    Loads a binary array from a file.  The first two dimensions
    of the array have length ni and nj, respectively.  The array is comprised
    of one or more 2D 'slices' of dimension ni x nj.  The number of 2D slices
    to read is 'nk'.  nk is not necessarily the length of the third dimension
    of the file.  The 'skip' parameter specifies the number of 2D slices
    to skip over before reading the nk number of 2D slices.  nk can be 1.

    Parameters
    ----------
    fdir : string
        A string with the directory of the binary file to open
    fname : string
        A string with the name of the binary file to open
    ni,nj : int
        the length of each array dimension.  must all be > 1
    skip : int
        the number of 2D (nj x ni) slices to skip.
        Default: 0
    nk   : int
        the number of 2D (nj x ni) slides to read  (not necessarily
        the length of the third dimension, just the length that you
        want to read).  
        Default 1
    filetype: string
        the file type, default is big endian (>) 32 bit float (f)
        alternatively, ('<d') would be little endian (<) 64 bit float (d)
        Default '>f'
    less_output : boolean
        a debug flag.  True means print more to the screen, False means be
        quieter.  
        Default: False
        
    Returns
    -------
    data
        a numpy array with dimensions nk x nj x ni
        
    Raises
    ------
    IOError
        If the file is not found

    """
    datafile = fdir + '/' + fname
    
    if less_output == False:
        print 'loading ' + fname
    
    # check to see if file exists.    
    file = glob.glob(datafile)
    if len(file) == 0:
        raise IOError(fname + ' not found ')

    f = open(datafile, 'rb')
    dt = np.dtype(filetype)

    if skip > 0:
        # skip ahead 'skip' number of 2D slices
        f.seek(ni*nj*skip*dt.itemsize)

    if ni*nj > 0:
        # read in 'nk' 2D slices (or records) from the mds file
        arr_k = np.fromfile(f, dtype=filetype, count=ni*nj*nk)

        f.close()

        if nk > 1:
            data = np.reshape(arr_k,(nk, nj, ni))
        else:
            data = np.reshape(arr_k,(nj, ni))
        
    # return the array
    else :
        if less_output == False:
            print "ni and nj must be > 0"

        data = []

    if less_output == False:
        print ('data shape ', data.shape)

    return data




#%%
def load_llc_compact(fdir, fname, llc=90, skip=0, nk=1, filetype = '>f', 
            less_output = False ):
    """

    Loads an MITgcm binary file in the 'compact' format of the 
    lat-lon-cap (LLC) grids.  Data in the compact format have
    dimensions 13*llc x llc x nk.

    Parameters
    ----------
    fdir : string
        A string with the directory of the binary file to open
    fname : string
        A string with the name of the binary file to open
    llc : int
        the size of the llc grid.  For ECCO v4, we use the llc90 domain 
        so `llc` would be `90`.  
        Default: 90
    skip : int
        the number of 2D slices (or records) to skip.  Records could 
        be vertical levels of a 3D field, or different 2D fields, or both.
        Default: 0
    nk : int
        number of 2D slices (or records) to load.  
        Default: 1
    filetype: string
        the file type, default is big endian (>) 32 bit float (f)
        alternatively, ('<d') would be little endian (<) 64 bit float (d)
        Deafult '>f'
    less_output : boolean
        a debug flag.  True means print more to the screen, False means be
        quieter.  Default False
        
    Returns
    -------
    data_compact
        a numpy array of dimension 13*llc x llc x nk

    """

    data_compact = load_binary_array(fdir, fname, llc, 13*llc, nk=nk, 
                    skip=skip, filetype = filetype, less_output = less_output)
    
    # return the array
    return data_compact



#%%
def load_llc_compact_to_faces(fdir, fname, llc=90, skip=0, nk=1, 
        filetype = '>f', less_output = False ):
    """

    Loads an MITgcm binary file in the 'compact' format of the 
    lat-lon-cap (LLC) grids and converts it to the '5 faces' format
    of the LLC grids.  

    Can load 2D and 3D arrays.

    Parameters
    ----------
    fdir : string
        A string with the directory of the binary file to open
    fname : string
        A string with the name of the binary file to open
    llc : int
        the size of the llc grid.  For ECCO v4, we use the llc90 domain 
        so `llc` would be `90`.  
        Default: 90
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
    F : a dictionary containing the five lat-lon-cap faces
        F['1'] is a numpy array of face 1.
        F['2'] is a numpy array of face 2.
        ...
        F['5'] is a numpy array of face 5.

        If the faces are 3D, the first dimension is k
        
    """
    
    data_compact = load_llc_compact(fdir, fname, llc, skip, nk, 
        filetype, less_output)

    # initialize arrays
    if nk > 1:
        f1_k = np.zeros((nk, 3*llc, llc))        
        f2_k = np.zeros((nk, 3*llc, llc))        
        f3_k = np.zeros((nk, llc, llc))                
        f4_k = np.zeros((nk, llc, 3*llc))
        f5_k = np.zeros((nk, llc, 3*llc))        
        
    else:
        f1_k = np.zeros((3*llc, llc))        
        f2_k = np.zeros((3*llc, llc))        
        f3_k = np.zeros((llc, llc))        
        f4_k = np.zeros((llc, 3*llc))
        f5_k = np.zeros((llc, 3*llc))      
    
    len_rec = 13*llc*llc

    # go through each 2D slice of the 3D field (or 2D record)
    for k in range(nk):

        if nk == 1:
            tmp = data_compact
        else:
            tmp = data_compact[k,:,:]

        print type(tmp)
        print tmp.shape

        F = llc_compact_to_faces(tmp)
        f1 = F['1']
        f2 = F['2']
        f3 = F['3']
        f4 = F['4']
        f5 = F['5']                

        if less_output == False:
            print 'F shapes'
            print f1.shape
            print f2.shape
            print f3.shape
            print f4.shape
            print f5.shape

        if nk == 1:
            f1_k = f1
            f2_k = f2
            f3_k = f3
            f4_k = f4
            f5_k = f5
            
        else:
            for face in range(0,5):
                f1_k[k,:,:] = f1
                f2_k[k,:,:] = f2
                f3_k[k,:,:] = f3
                f4_k[k,:,:] = f4
                f5_k[k,:,:] = f5
        
        
    # put these faces in a dictionary
    F = {}    
    F['1'] = f1_k
    F['2'] = f2_k
    F['3'] = f3_k
    F['4'] = f4_k
    F['5'] = f5_k
    
    # return a dictionary of faces
    return F


#%%
def load_llc_compact_to_tiles(fdir, fname, llc=90, skip=0, nk=1, filetype = '>f', 
                 less_output = False ):
    """

    Loads an MITgcm binary file in the 'compact' format of the 
    lat-lon-cap (LLC) grids and converts it to the '13 tiles' format
    of the LLC grids.  

    Parameters
    ----------
    fdir : string
        A string with the directory of the binary file to open
    fname : string
        A string with the name of the binary file to open
    llc : int
        the size of the llc grid.  For ECCO v4, we use the llc90 domain 
        so `llc` would be `90`.  
        Default: 90
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
    data_tiles
        a numpy array of dimension 13 x nk x llc x llc, one llc x llc array 
        for each of the 13 tiles and nk levels

    """
    
    data_compact = load_llc_compact(fdir, fname, llc, skip, nk, 
            filetype, less_output)

    data_tiles   = llc_compact_to_tiles(data_compact, less_output)
        
    # return the array
    return data_tiles

#%%
def llc_compact_to_tiles(data_compact, less_output = False):
    """

    Converts a numpy binary array in the 'compact' format of the 
    lat-lon-cap (LLC) grids and converts it to the '13 tiles' format
    of the LLC grids.  

    Parameters
    ----------
    data_compact
        a numpy array of dimension 13*llc x llc x nk
    less_output : boolean
        a debug flag.  True means print more to the screen, False means be
        quieter.  
        Default: False
        
    Returns
    -------
    data_tiles_k
        a numpy array of dimension 13 x nk x llc x llc, one llc x llc array 
        for each of the 13 tiles and nk levels

    """   
    if less_output == False:
        print ('data compact shape ', data_compact.shape)
        print (type(data_compact))

    if data_compact.ndim == 3:
        nk = np.size(data_compact,0)
    else:
        nk = 1

    # the last dimension should have length llc
    llc = np.size(data_compact,-1)

    if less_output == False:
        print ('nk = ' , nk)
        print ('llc = ', llc)


    # define a blank array
    if nk > 1:
        data_tiles_k = np.zeros((13, nk, llc, llc))
    else:
        data_tiles_k = np.zeros((13, llc, llc))
    
    if less_output == False:
        print ('data_tiles_k shape = ' , data_tiles_k.shape)
    

    len_rec = 13*llc*llc

    # go through each 2D slice (or record)
    for k in range(nk):

        if nk == 1:
            tmp = data_compact
        else:
            tmp = data_compact[k,:,:]

        # operates on a 2D compact array
        F = llc_compact_to_faces(tmp)
        f1 = F['1']
        f2 = F['2']
        f3 = F['3']
        f4 = F['4']
        f5 = F['5']                

        if less_output == False:
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
        

        data_tiles = llc_faces_to_tiles(F)

        
        if less_output == False:
            plt.figure()
            plt.imshow(data_tiles[0], origin='lower')
            plt.figure()
            plt.imshow(data_tiles[1], origin='lower')
            plt.figure()
            plt.imshow(data_tiles[2], origin='lower')
      
        if nk == 1:
            if less_output == False:
                print ('nk = 1')
                print np.max(data_tiles)
            
            data_tiles_k = data_tiles
            
        else:
            if less_output == False:
                print ('k' , k)
            
            for tile in range(0,13):
                if less_output == False:
                    print ('tile',tile)
                
                data_tiles_k[tile,k,:,:] = data_tiles[tile,:,:]
        
    return data_tiles_k

#%%
def llc_compact_to_faces(data_compact):
    """
    Converts a numpy binary array in the 'compact' format of the 
    lat-lon-cap (LLC) grids and converts it into the 5 'faces'
    of the llc grid.  4 lat-lon approximately pieces and one Arctic 'cap'

    Only works with 2D compact arrays

    Parameters
    ----------
    data_compact : 
        An 2D array of dimension  13*llc x llc 
        
    Returns
    -------
    F : a dictionary containing the five lat-lon-cap faces
        F['1'] is a numpy array of face 1.
        F['2'] is a numpy array of face 2.
        ...
        F['5'] is a numpy array of face 5.

    """

    # llc is the length of the second dimension
    print type(data_compact)
    print data_compact.shape


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
    F = {}
    F['1'] = f1
    F['2'] = f2
    F['3'] = f3
    F['4'] = f4
    F['5'] = f5

    return F


#%%
def llc_faces_to_tiles(F):
    """

    Converts a dictionary containing 5 lat-lon-cap faces into 13 tiles
    of dimension llc x llc x nk.  tiles 1-6 and 8-13 are approximately
    lat-lon in orientation and tile 7 is the Arctic 'cap'
    
    Only works for 2D faces

    Parameters
    ----------
    F : a dictionary containing the five lat-lon-cap faces
        F['1'] is a numpy array of face 1.
        F['2'] is a numpy array of face 2.
        ...
        F['5'] is a numpy array of face 5.
        
        
    Returns
    -------
    data_tiles :
        a numpy array of dimension 13 x nk x llc x llc, one llc x llc array 
        for each of the 13 tiles and nk levels

    """
    f1 = F['1']
    f2 = F['2']
    f3 = F['3']
    f4 = F['4']
    f5 = F['5']                

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

    Converts an array of 13 'tiles' from the lat-lon-cap grid
    and rearranges them to 5 faces.  Faces 1,2,4, and 5 are approximately 
    lat-lon while face 3 is the Arctic 'cap' 

    Only works for 2D data tiles [so far].


    Parameters
    ----------
    data_tiles : the 2D array with 13 tiles
        An array of dimension 13xllcxllc
        
    Returns
    -------
    F : an array of five faces
        F['1'] is face 1

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

    F = {}
    F['1'] = f1
    F['2'] = f2
    F['3'] = f3
    F['4'] = f4
    F['5'] = f5
    
    return F


def llc_faces_to_compact(F):
    """
    
    Converts a dictionary containing five 'faces' of the lat-lon-cap grid
    and rearranges it to the 'compact' llc format.

    Only works for 2D faces [so far].


    Parameters
    ----------
    data_tiles : the 2D array with 13 tiles
        An array of dimension 13xllcxllc
        
    Returns
    -------
    F : an array of five faces
        F['1'] is face 1

    - dimensions:
        f1,f2: 3*llc x llc
           f3: 3*llc x llc
        f4,f5: llc x 3*llc  
    
    """

    f1 = F['1']
    f2 = F['2']
    f3 = F['3']
    f4 = F['4']
    f5 = F['5']

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