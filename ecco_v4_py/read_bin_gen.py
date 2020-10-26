"""ECCO v4 Python: read_bin_gen

This module includes utility routines for loading general binary files,
not necessarily in the lat-lon-cap layout. For LLC type data, see read_bin_llc.

.. _ecco_v4_py Documentation :
   https://github.com/ECCO-GROUP/ECCOv4-py
"""

from __future__ import division,print_function
import numpy as np
import glob
import os
from pathlib import Path

#%%
def load_binary_array(fdir, fname, ni, nj, nk=1, nl=1, skip=0,
                      filetype = '>f', less_output = False ):
    """
    Note: This function is for reading a general binary file. To read data in
    the llc structure, see read_bin_llc.

    Loads a binary array from a file into memory, which is not necessarily in
    llc format.  The first two dimensions
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
    #datafile = fdir + '/' + fname
    data_folder = Path(fdir)
    datafile = data_folder / fname

    #datafile = os.path.join(fdir, fname)
    if less_output == False:
        print('load_binary_array: loading file', datafile)

    # check to see if file exists.
    if datafile.exists() == False:
        raise IOError(fname + ' not found ')

    f = open(datafile, 'rb')
    dt = np.dtype(filetype)

    if skip > 0:
        # skip ahead 'skip' number of 2D slices
        f.seek(ni*nj*skip*dt.itemsize)

    if (ni <= 0) or (nj <= 0):
        raise TypeError('ni and nj must be > 1')

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

        if not less_output:
            print('load_binary_array: loading all 2D records.  nk =',nk)

        # reshape the array to 2D records
        if nk > 1: # we have more than one 2D record, make 3D field
            data = np.reshape(arr_k,(nk, nj, ni))

        else: # nk = 1, just make 2D field
            data = np.reshape(arr_k,(nj, ni))

    # read a specific number of records (nk*nl)
    else:
        if (nk <= 0) or (nl <= 0):
            raise TypeError('nk and nl must be > 0.  If they are singleton dimensions, use 1')

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

    if not less_output:
        print('load_binary_array: data array shape ', data.shape)
        print('load_binary_array: data array type ', data.dtype)

    return data
