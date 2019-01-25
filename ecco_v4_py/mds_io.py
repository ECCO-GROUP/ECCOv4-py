#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""ECCO v4 Python: mds_io

This module includes utility routines for loading binary files in the llc 13-tile native flat binary layout.  This layout is the default for MITgcm input and output for global setups using lat-lon-cap (llc) layout.  The llc layout is used for ECCO v4. 

.. _ecco_v4_py Documentation :
   https://github.com/ECCO-GROUP/ECCOv4-py
"""

from __future__ import division
import numpy as np
import glob

from llc_array_conversion  import llc_compact_to_tiles
from llc_array_conversion  import llc_compact_to_faces
from llc_array_conversion  import llc_faces_to_tiles
from llc_array_conversion  import llc_faces_to_compact
from llc_array_conversion  import llc_tiles_to_faces
from llc_array_conversion  import llc_tiles_to_compact

#%%
def load_binary_array(fdir, fname, ni, nj, nk=1, nl=1, skip=0, 
                      filetype = '>f', less_output = False ):
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
        the length of each array dimension.  ni, nj must be > 0
    skip : int
        the number of 2D (nj x ni) slices to skip.
        Default: 0
    nk : int
        number of 2D slices (or records) to load in the third dimension.  
        if nk = -1, load all 2D slices
        Default: 1 [singleton]
    nl : int
        number of 2D slices (or records) to load in the fourth dimension.  
        Default: 1 [singleton]
    filetype: string
        the file type, default is big endian (>) 32 bit float (f)
        alternatively, ('<d') would be little endian (<) 64 bit float (d)
        Default '>f'
    less_output : boolean
        A debugging flag.  False = less debugging output
        Default: False
        
    Returns
    -------
    data
        a numpy array with dimensions nl x nk x nj x ni
        
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

    if (ni <= 0) or (nj <= 0):
        print ('ni and nj must be > 1')
        return []

    # load all 2D records
    if nk == -1:
        # nl can only be 1 if we use nk = -1
        nl = 1
        # read all 2D records
        arr_k = np.fromfile(f, dtype=filetype, count=-1)
        # find the number of 2D records (nk)
        length_arr_k = len(arr_k)

        # length of each 2D slice is ni * nj
        nk = int(length_arr_k / (ni*nj))

        if less_output == False:
            print ('loading all 2D records.  nk =',nk)
        
        # reshape the array to 2D records
        if nk > 1: # we have more than one 2D record, make 3D field
            data = np.reshape(arr_k,(nk, nj, ni))
        
        else: # nk = 1, just make 2D field
            data = np.reshape(arr_k,(nj, ni))

    # read a specific number of records (nk*nl)
    else:
        if (nk <= 0) or (nl <= 0):
            print('nk and nl must be > 0.  If they are singleton dimensions, use 1')
            return []

        # read in nk*nl 2D records
        arr_k = np.fromfile(f, dtype=filetype, count=ni*nj*nk*nl)

        # put data into 2D records
        #  - if we have a fourth dimension
        if nl > 1:
            data = np.reshape(arr_k,(nl, nk, nj, ni))
        
        #  - if we have a third dimension
        elif nk > 1:
            data = np.reshape(arr_k,(nk, nj, ni))
        
        #  - if we only have two dimensions
        else:
            data = np.reshape(arr_k,(nj, ni))
    
    f.close()
       
    
    if less_output == False:
        print ('data shape ', data.shape)

    return data




#%%
def load_llc_compact(fdir, fname, llc=90, skip=0, nk=1, nl=1, 
            filetype = '>f', less_output = False ):
    """

    Loads an MITgcm binary file in the 'compact' format of the 
    lat-lon-cap (LLC) grids.  

    Data in the compact format should have dimensions:
    nl x nk x 13*llc x llc

    If dimensions nl or nk are singular, they are not included 
    as dimensions in the compact array

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
        number of 2D slices (or records) to load in the third dimension.  
        if nk = -1, load all 2D slices
        Default: 1 [singleton]
    nl : int
        number of 2D slices (or records) to load in the fourth dimension.  
        Default: 1 [singleton]
    filetype: string
        the file type, default is big endian (>) 32 bit float (f)
        alternatively, ('<d') would be little endian (<) 64 bit float (d)
        Deafult '>f'
    less_output : boolean
        A debugging flag.  False = less debugging output
        Default: False
        
    Returns
    -------
    data_compact
        a numpy array of dimension nl x nk x llc x llc 

    """

    data_compact = load_binary_array(fdir, fname, llc, 13*llc, nk=nk, nl=nl, 
                    skip=skip, filetype = filetype, less_output = less_output)
    
    # return the array
    return data_compact



#%%
def load_llc_compact_to_faces(fdir, fname, llc=90, skip=0, nk=1, nl=1,
        filetype = '>f', less_output = False):
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
        the number of 2D slices (or records) to skip.  Records could be 
        vertical levels of a 3D field, or different 2D fields, or both.
    nk : int
        number of 2D slices (or records) to load in the third dimension.  
        if nk = -1, load all 2D slices
        Default: 1 [singleton]
    nl : int
        number of 2D slices (or records) to load in the fourth dimension.  
        Default: 1 [singleton]
    filetype: string
        the file type, default is big endian (>) 32 bit float (f)
        alternatively, ('<d') would be little endian (<) 64 bit float (d)
    less_output : boolean
        A debugging flag.  False = less debugging output
        Default: False
        
    Returns
    -------
    F : a dictionary containing the five lat-lon-cap faces
        F[n] is a numpy array of face n, n in [1..5]

    - dimensions of each 2D slice of F
        f1,f2: 3*llc x llc
           f3: llc x llc
        f4,f5: llc x 3*llc  
        
    """
    
    data_compact = load_llc_compact(fdir, fname, llc=llc, skip=skip, nk=nk, nl=nl, 
        filetype = filetype, less_output=less_output)

    F = llc_compact_to_faces(data_compact, less_output = less_output)

    return F


#%%
def load_llc_compact_to_tiles(fdir, fname, llc=90, skip=0, nk=1, nl=1, 
                filetype = '>f', less_output = False, 
                third_dimension = 'time', fourth_dimension = 'depth'):
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
        number of 2D slices (or records) to load in the third dimension.  
        if nk = -1, load all 2D slices
        Default: 1 [singleton]
    nl : int
        number of 2D slices (or records) to load in the fourth dimension.  
        Default: 1 [singleton] 
    filetype: string
        the file type, default is big endian (>) 32 bit float (f)
        alternatively, ('<d') would be little endian (<) 64 bit float (d)
    less_output : boolean
        A debugging flag.  False = less debugging output
        Default: False
        
    Returns
    -------
    data_tiles
        a numpy array of dimension 13 x nl x nk x llc x llc, one llc x llc array 
        for each of the 13 tiles and nl and nk levels.  

    """
    
    data_compact = load_llc_compact(fdir, fname, llc=llc, skip=skip, nk=nk, nl=nl, 
        filetype = filetype, less_output=less_output)

    data_tiles   = llc_compact_to_tiles(data_compact, less_output=less_output)

        
    # return the array
    return data_tiles
