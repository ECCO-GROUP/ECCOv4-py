"""
ECCO v4 Python: llc_array_conversion

This module includes routines for converting arrays between the model
'compact' format, the 13 tile llc format, and the 5 'face' llc format.

.. _ecco_v4_py Documentation :
   https://github.com/ECCO-GROUP/ECCOv4-py

"""

from __future__ import division,print_function
import numpy as np
import xarray as xr
from xmitgcm.variables import dimensions

# xarray>=0.12.0 compatibility
try:
    from xarray.core.pycompat import OrderedDict
except ImportError:
    from collections import OrderedDict


#%%
def llc_compact_to_tiles(data_compact, less_output = False):
    """

    Converts a numpy binary array in the 'compact' format of the
    lat-lon-cap (LLC) grids and converts it to the '13 tiles' format
    of the LLC grids.

    Parameters
    ----------
    data_compact : ndarray
        a numpy array of dimension nl x nk x 13*llc x llc

    less_output : boolean, optional, default False
        A debugging flag.  False = less debugging output


    Returns
    -------
    data_tiles : ndarray
        a numpy array organized by, at most,
        13 tiles x nl x nk x llc x llc

    Note
    ----
    If dimensions nl or nk are singular, they are not included
    as dimensions in data_tiles

    """

    data_tiles =  llc_faces_to_tiles(
                    llc_compact_to_faces(data_compact,
                                         less_output=less_output),
                    less_output=less_output)

    return data_tiles

# %%
def llc_tiles_to_compact(data_tiles, less_output = False):
    """

    Converts a numpy binary array in the 'compact' format of the
    lat-lon-cap (LLC) grids and converts it to the '13 tiles' format
    of the LLC grids.

    Parameters
    ----------
    data_tiles : ndarray
        a numpy array organized by, at most,
        13 tiles x nl x nk x llc x llc

        where dimensions 'nl' and 'nk' are optional.

    less_output : boolean, optional, default False
        A debugging flag.  False = less debugging output

    Returns
    -------
    data_compact : ndarray
        a numpy array of dimension nl x nk x 13*llc x llc

    Note
    ----
    If dimensions nl or nk are singular, they are not included
    as dimensions in data_compact

    """

    data_faces   = llc_tiles_to_faces(data_tiles, less_output=less_output)
    data_compact = llc_faces_to_compact(data_faces, less_output=less_output)

    return data_compact



#%%
def llc_compact_to_faces(data_compact, less_output = False):
    """
    Converts a numpy binary array in the 'compact' format of the
    lat-lon-cap (LLC) grids and converts it into the 5 'faces'
    of the llc grid.

    The five faces are 4 approximately lat-lon oriented and one Arctic 'cap'

    Parameters
    ----------
    data_compact : ndarray
        An 2D array of dimension  nl x nk x 13*llc x llc

    less_output : boolean, optional, default False
        A debugging flag.  False = less debugging output


    Returns
    -------
    F : dict
        a dictionary containing the five lat-lon-cap faces

        F[n] is a numpy array of face n, n in [1..5]

        dimensions of each 2D slice of F

        - f1,f2: 3*llc x llc
        -    f3: llc x llc
        - f4,f5: llc x 3*llc

    Note
    ----
    If dimensions nl or nk are singular, they are not included
    as dimensions of data_compact

    """

    dims = data_compact.shape
    num_dims = len(dims)

    # final dimension is always of length llc
    llc = dims[-1]

    # dtype of compact array
    arr_dtype = data_compact.dtype

    if not less_output:
        print('llc_compact_to_faces: dims, llc ', dims, llc)
        print('llc_compact_to_faces: data_compact array type ', data_compact.dtype)

    if num_dims == 2: # we have a single 2D slices (y, x)
        f1 = np.zeros((3*llc, llc), dtype=arr_dtype)
        f2 = np.zeros((3*llc, llc), dtype=arr_dtype)
        f3 = np.zeros((llc, llc), dtype=arr_dtype)
        f4 = np.zeros((llc, 3*llc), dtype=arr_dtype)
        f5 = np.zeros((llc, 3*llc), dtype=arr_dtype)

    elif num_dims == 3: # we have 3D slices (time or depth, y, x)
        nk = dims[0]

        f1 = np.zeros((nk, 3*llc, llc), dtype=arr_dtype)
        f2 = np.zeros((nk, 3*llc, llc), dtype=arr_dtype)
        f3 = np.zeros((nk, llc, llc), dtype=arr_dtype)
        f4 = np.zeros((nk, llc, 3*llc), dtype=arr_dtype)
        f5 = np.zeros((nk, llc, 3*llc), dtype=arr_dtype)

    elif num_dims == 4: # we have a 4D slice (time or depth, time or depth, y, x)
        nl = dims[0]
        nk = dims[1]

        f1 = np.zeros((nl, nk, 3*llc, llc), dtype=arr_dtype)
        f2 = np.zeros((nl, nk, 3*llc, llc), dtype=arr_dtype)
        f3 = np.zeros((nl, nk, llc, llc), dtype=arr_dtype)
        f4 = np.zeros((nl, nk, llc, 3*llc), dtype=arr_dtype)
        f5 = np.zeros((nl, nk, llc, 3*llc), dtype=arr_dtype)

    else:
        print ('llc_compact_to_faces: can only handle compact arrays of 2, 3, or 4 dimensions!')
        return []

    # map the data from the compact format to the five face arrays

    # -- 2D case
    if num_dims == 2:

        f1 = data_compact[:3*llc,:]
        f2 = data_compact[3*llc:6*llc,:]
        f3 = data_compact[6*llc:7*llc,:]

        #f4 = np.zeros((llc, 3*llc))

        for f in range(8,11):
            i1 = np.arange(0, llc)+(f-8)*llc
            i2 = np.arange(0,3*llc,3) + 7*llc + f -8
            f4[:,i1] = data_compact[i2,:]

        #f5 = np.zeros((llc, 3*llc))

        for f in range(11,14):
            i1 = np.arange(0, llc)+(f-11)*llc
            i2 = np.arange(0,3*llc,3) + 10*llc + f -11
            #print ('f, i1, i2 ', f, i1[0], i2[0])

            f5[:,i1] = data_compact[i2,:]

    # -- 3D case
    elif num_dims == 3:
        # loop over k

        for k in range(nk):
            f1[k,:] = data_compact[k,:3*llc,:]
            f2[k,:] = data_compact[k,3*llc:6*llc,:]
            f3[k,:] = data_compact[k,6*llc:7*llc,:]

            # if someone could explain why I have to make
            # dummy arrays of f4_tmp and f5_tmp instead of just using
            # f5 directly I would be so grateful!
            f4_tmp = np.zeros((llc, 3*llc))
            f5_tmp = np.zeros((llc, 3*llc))

            for f in range(8,11):
                i1 = np.arange(0, llc)+(f-8)*llc
                i2 = np.arange(0,3*llc,3) + 7*llc + f -8
                f4_tmp[:,i1] = data_compact[k,i2,:]


            for f in range(11,14):
                i1 = np.arange(0,  llc)   +(f-11)*llc
                i2 = np.arange(0,3*llc,3) + 10*llc + f -11
                f5_tmp[:,i1] = data_compact[k,i2,:]

            f4[k,:] = f4_tmp
            f5[k,:] = f5_tmp



    # -- 4D case
    elif num_dims == 4:
        # loop over l and k
        for l in range(nl):
            for k in range(nk):

                f1[l,k,:] = data_compact[l,k,:3*llc,:]
                f2[l,k,:] = data_compact[l,k, 3*llc:6*llc,:]
                f3[l,k,:] = data_compact[l,k, 6*llc:7*llc,:]

                # if someone could explain why I have to make
                # dummy arrays of f4_tmp and f5_tmp instead of just using
                # f5 directly I would be so grateful!
                f4_tmp = np.zeros((llc, 3*llc))
                f5_tmp = np.zeros((llc, 3*llc))

                for f in range(8,11):
                    i1 = np.arange(0, llc)+(f-8)*llc
                    i2 = np.arange(0,3*llc,3) + 7*llc + f -8
                    f4_tmp[:,i1] = data_compact[l,k,i2,:]

                for f in range(11,14):
                    i1 = np.arange(0, llc)+(f-11)*llc
                    i2 = np.arange(0,3*llc,3) + 10*llc + f -11
                    f5_tmp[:,i1] = data_compact[l,k,i2,:]

                f4[l,k,:,:] = f4_tmp
                f5[l,k,:,:] = f5_tmp


    # put the 5 faces in the dictionary.
    F = {}
    F[1] = f1
    F[2] = f2
    F[3] = f3
    F[4] = f4
    F[5] = f5

    return F


#%%
def llc_faces_to_tiles(F, less_output=False):
    """

    Converts a dictionary, F, containing 5 lat-lon-cap faces into 13 tiles
    of dimension nl x nk x llc x llc x nk.

    Tiles 1-6 and 8-13 are oriented approximately lat-lon
    while tile 7 is the Arctic 'cap'

    Parameters
    ----------
    F : dict
        a dictionary containing the five lat-lon-cap faces

        F[n] is a numpy array of face n, n in [1..5]

    less_output : boolean, optional, default False
        A debugging flag.  False = less debugging output

    Returns
    -------
    data_tiles : ndarray
        an array of dimension 13 x nl x nk x llc x llc,

        Each 2D slice is dimension 13 x llc x llc

    Note
    ----
    If dimensions nl or nk are singular, they are not included
    as dimensions of data_tiles


    """

    # pull out the five face arrays
    f1 = F[1]
    f2 = F[2]
    f3 = F[3]
    f4 = F[4]
    f5 = F[5]

    dims = f3.shape
    num_dims = len(dims)

    # dtype of compact array
    arr_dtype = f1.dtype

    # final dimension of face 1 is always of length llc
    ni_3 = f3.shape[-1]
    nj_3 = f3.shape[-2]

    llc = ni_3 # default
    #

    if num_dims == 2: # we have a single 2D slices (y, x)
        data_tiles = np.zeros((13, llc, llc), dtype=arr_dtype)


    elif num_dims == 3: # we have 3D slices (time or depth, y, x)
        nk = dims[0]
        data_tiles = np.zeros((nk, 13, llc, llc), dtype=arr_dtype)


    elif num_dims == 4: # we have a 4D slice (time or depth, time or depth, y, x)
        nl = dims[0]
        nk = dims[1]

        data_tiles = np.zeros((nl, nk, 13, llc, llc), dtype=arr_dtype)

    else:
        print ('llc_faces_to_tiles: can only handle face arrays that have 2, 3, or 4 dimensions!')
        return []

    # llc is the length of the second dimension
    if not less_output:
        print ('llc_faces_to_tiles: data_tiles shape ', data_tiles.shape)
        print ('llc_faces_to_tiles: data_tiles dtype ', data_tiles.dtype)


    # map the data from the faces format to the 13 tile arrays

    # -- 2D case
    if num_dims == 2:
        data_tiles[0,:]  = f1[llc*0:llc*1,:]
        data_tiles[1,:]  = f1[llc*1:llc*2,:]
        data_tiles[2,:]  = f1[llc*2:,:]
        data_tiles[3,:]  = f2[llc*0:llc*1,:]
        data_tiles[4,:]  = f2[llc*1:llc*2,:]
        data_tiles[5,:]  = f2[llc*2:,:]
        data_tiles[6,:]  = f3
        data_tiles[7,:]  = f4[:,llc*0:llc*1]
        data_tiles[8,:]  = f4[:,llc*1:llc*2]
        data_tiles[9,:]  = f4[:,llc*2:]
        data_tiles[10,:] = f5[:,llc*0:llc*1]
        data_tiles[11,:] = f5[:,llc*1:llc*2]
        data_tiles[12,:] = f5[:,llc*2:]

    # -- 3D case
    if num_dims == 3:
        # loop over k
        for k in range(nk):
            data_tiles[k,0,:]  = f1[k,llc*0:llc*1,:]
            data_tiles[k,1,:]  = f1[k,llc*1:llc*2,:]
            data_tiles[k,2,:]  = f1[k,llc*2:,:]
            data_tiles[k,3,:]  = f2[k,llc*0:llc*1,:]
            data_tiles[k,4,:]  = f2[k,llc*1:llc*2,:]
            data_tiles[k,5,:]  = f2[k,llc*2:,:]
            data_tiles[k,6,:]  = f3[k,:]
            data_tiles[k,7,:]  = f4[k,:,llc*0:llc*1]
            data_tiles[k,8,:]  = f4[k,:,llc*1:llc*2]
            data_tiles[k,9,:]  = f4[k,:,llc*2:]
            data_tiles[k,10,:] = f5[k,:,llc*0:llc*1]
            data_tiles[k,11,:] = f5[k,:,llc*1:llc*2]
            data_tiles[k,12,:] = f5[k,:,llc*2:]

    # -- 4D case
    if num_dims == 4:
        #loop over l and k
        for l in range(nl):
            for k in range(nk):
                data_tiles[l,k,0,:]  = f1[l,k,llc*0:llc*1,:]
                data_tiles[l,k,1,:]  = f1[l,k,llc*1:llc*2,:]
                data_tiles[l,k,2,:]  = f1[l,k,llc*2:,:]
                data_tiles[l,k,3,:]  = f2[l,k,llc*0:llc*1,:]
                data_tiles[l,k,4,:]  = f2[l,k,llc*1:llc*2,:]
                data_tiles[l,k,5,:]  = f2[l,k,llc*2:,:]
                data_tiles[l,k,6,:]  = f3[l,k,:]
                data_tiles[l,k,7,:]  = f4[l,k,:,llc*0:llc*1]
                data_tiles[l,k,8,:]  = f4[l,k,:,llc*1:llc*2]
                data_tiles[l,k,9,:]  = f4[l,k,:,llc*2:]
                data_tiles[l,k,10,:] = f5[l,k,:,llc*0:llc*1]
                data_tiles[l,k,11,:] = f5[l,k,:,llc*1:llc*2]
                data_tiles[l,k,12,:] = f5[l,k,:,llc*2:]

    return data_tiles



def llc_tiles_to_faces(data_tiles, less_output=False):
    """

    Converts an array of 13 'tiles' from the lat-lon-cap grid
    and rearranges them to 5 faces.  Faces 1,2,4, and 5 are approximately
    lat-lon while face 3 is the Arctic 'cap'

    Parameters
    ----------
    data_tiles :
        An array of dimension 13 x nl x nk x llc x llc

    If dimensions nl or nk are singular, they are not included
        as dimensions of data_tiles

    less_output : boolean
        A debugging flag.  False = less debugging output
        Default: False

    Returns
    -------
    F : dict
        a dictionary containing the five lat-lon-cap faces

        F[n] is a numpy array of face n, n in [1..5]

        dimensions of each 2D slice of F

        - f1,f2: 3*llc x llc
        -    f3: llc x llc
        - f4,f5: llc x 3*llc

    """

    # ascertain how many dimensions are in the faces (minimum 3, maximum 5)
    dims = data_tiles.shape
    num_dims = len(dims)

    # the final dimension is always length llc
    llc = dims[-1]

    # tiles is always just before (y,x) dims
    num_tiles = dims[-3]

    # data type of original data_tiles
    arr_dtype = data_tiles.dtype

    if not less_output:
        print('llc_tiles_to_faces: num_tiles, ', num_tiles)

    if num_dims == 3: # we have a 13 2D slices (tile, y, x)
        f1 = np.zeros((3*llc, llc), dtype=arr_dtype)
        f2 = np.zeros((3*llc, llc), dtype=arr_dtype)
        f3 = np.zeros((llc, llc), dtype=arr_dtype)
        f4 = np.zeros((llc, 3*llc), dtype=arr_dtype)
        f5 = np.zeros((llc, 3*llc), dtype=arr_dtype)

    elif num_dims == 4: # 13 3D slices (time or depth, tile, y, x)

        nk = dims[0]

        f1 = np.zeros((nk, 3*llc, llc), dtype=arr_dtype)
        f2 = np.zeros((nk, 3*llc, llc), dtype=arr_dtype)
        f3 = np.zeros((nk, llc, llc), dtype=arr_dtype)
        f4 = np.zeros((nk, llc, 3*llc), dtype=arr_dtype)
        f5 = np.zeros((nk, llc, 3*llc), dtype=arr_dtype)

    elif num_dims == 5: # 4D slice (time or depth, time or depth, tile, y, x)
        nl = dims[0]
        nk = dims[1]

        f1 = np.zeros((nl,nk, 3*llc, llc), dtype=arr_dtype)
        f2 = np.zeros((nl,nk, 3*llc, llc), dtype=arr_dtype)
        f3 = np.zeros((nl,nk, llc, llc), dtype=arr_dtype)
        f4 = np.zeros((nl,nk, llc, 3*llc), dtype=arr_dtype)
        f5 = np.zeros((nl,nk, llc, 3*llc), dtype=arr_dtype)

    else:
        print ('llc_tiles_to_faces: can only handle tiles that have 2, 3, or 4 dimensions!')
        return []

    # Map data on the tiles to the faces

    # 2D slices on 13 tiles
    if num_dims == 3:

        f1[llc*0:llc*1,:] = data_tiles[0,:]

        f1[llc*1:llc*2,:] = data_tiles[1,:]
        f1[llc*2:,:]      = data_tiles[2,:]

        f2[llc*0:llc*1,:] = data_tiles[3,:]
        f2[llc*1:llc*2,:] = data_tiles[4,:]
        f2[llc*2:,:]      = data_tiles[5,:]

        f3                = data_tiles[6,:]

        f4[:,llc*0:llc*1] = data_tiles[7,:]
        f4[:,llc*1:llc*2] = data_tiles[8,:]
        f4[:,llc*2:]      = data_tiles[9,:]

        f5[:,llc*0:llc*1] = data_tiles[10,:]
        f5[:,llc*1:llc*2] = data_tiles[11,:]
        f5[:,llc*2:]      = data_tiles[12,:]

    # 3D slices on 13 tiles
    elif num_dims == 4:

        for k in range(nk):
            f1[k,llc*0:llc*1,:] = data_tiles[k,0,:]

            f1[k,llc*1:llc*2,:] = data_tiles[k,1,:]
            f1[k,llc*2:,:]      = data_tiles[k,2,:]

            f2[k,llc*0:llc*1,:] = data_tiles[k,3,:]
            f2[k,llc*1:llc*2,:] = data_tiles[k,4,:]
            f2[k,llc*2:,:]      = data_tiles[k,5,:]

            f3[k,:]             = data_tiles[k,6,:]

            f4[k,:,llc*0:llc*1] = data_tiles[k,7,:]
            f4[k,:,llc*1:llc*2] = data_tiles[k,8,:]
            f4[k,:,llc*2:]      = data_tiles[k,9,:]

            f5[k,:,llc*0:llc*1] = data_tiles[k,10,:]
            f5[k,:,llc*1:llc*2] = data_tiles[k,11,:]
            f5[k,:,llc*2:]      = data_tiles[k,12,:]

    # 4D slices on 13 tiles
    elif num_dims == 5:
        for l in range(nl):
            for k in range(nk):
                f1[l,k,llc*0:llc*1,:] = data_tiles[l,k,0,:]

                f1[l,k,llc*1:llc*2,:] = data_tiles[l,k,1,:]
                f1[l,k,llc*2:,:]      = data_tiles[l,k,2,:]

                f2[l,k,llc*0:llc*1,:] = data_tiles[l,k,3,:]
                f2[l,k,llc*1:llc*2,:] = data_tiles[l,k,4,:]
                f2[l,k,llc*2:,:]      = data_tiles[l,k,5,:]

                f3[l,k,:]             = data_tiles[l,k,6,:]

                f4[l,k,:,llc*0:llc*1] = data_tiles[l,k,7,:]
                f4[l,k,:,llc*1:llc*2] = data_tiles[l,k,8,:]
                f4[l,k,:,llc*2:]      = data_tiles[l,k,9,:]

                f5[l,k,:,llc*0:llc*1] = data_tiles[l,k,10,:]
                f5[l,k,:,llc*1:llc*2] = data_tiles[l,k,11,:]
                f5[l,k,:,llc*2:]      = data_tiles[l,k,12,:]

    # Build the F dictionary
    F = {}
    F[1] = f1
    F[2] = f2
    F[3] = f3
    F[4] = f4
    F[5] = f5

    return F


def llc_faces_to_compact(F, less_output=True):
    """

    Converts a dictionary containing five 'faces' of the lat-lon-cap grid
    and rearranges it to the 'compact' llc format.


    Parameters
    ----------
    F : dict
        a dictionary containing the five lat-lon-cap faces

        F[n] is a numpy array of face n, n in [1..5]

        dimensions of each 2D slice of F

        - f1,f2: 3*llc x llc
        -    f3: llc x llc
        - f4,f5: llc x 3*llc

    less_output : boolean, optional, default False
        A debugging flag.  False = less debugging output

    Returns
    -------
    data_compact : ndarray
        an array of dimension nl x nk x nj x ni

        F is in the llc compact format.

    Note
    ----
    If dimensions nl or nk are singular, they are not included
    as dimensions of data_compact

    """

    # pull the individual faces out of the F dictionary
    f1 = F[1]
    f2 = F[2]
    f3 = F[3]
    f4 = F[4]
    f5 = F[5]

    # ascertain how many dimensions are in the faces (minimum 2, maximum 4)
    dims = f3.shape
    num_dims = len(dims)

    # data type of original faces
    arr_dtype = f1.dtype

    # the final dimension is always the llc #
    llc = dims[-1]

    # initialize the 'data_compact' array
    if num_dims == 2: # we have a 2D slice (x,y)
        data_compact = np.zeros((13*llc, llc), dtype=arr_dtype)

    elif num_dims == 3: # 3D slice (x, y, time or depth)
        nk = dims[0]
        data_compact = np.zeros((nk, 13*llc, llc), dtype=arr_dtype)

    elif num_dims == 4: # 4D slice (x,y,time and depth)
        nl = dims[0]
        nk = dims[1]
        data_compact = np.zeros((nl, nk, 13*llc, llc), dtype=arr_dtype)
    else:
        print ('llc_faces_to_compact: can only handle faces that have 2, 3, or 4 dimensions!')
        return []

    if not less_output:
        print ('llc_faces_to_compact: face 3 shape', f3.shape)

    if num_dims == 2:

        data_compact[:3*llc,:]      = f1
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

    elif num_dims == 3:
        # loop through k indicies
        print ('llc_faces_to_compact: data_compact array shape', data_compact.shape)

        for k in range(nk):
            data_compact[k,:3*llc,:]      = f1[k,:]
            data_compact[k,3*llc:6*llc,:] = f2[k,:]
            data_compact[k,6*llc:7*llc,:] = f3[k,:]

            # if someone could explain why I have to transpose
            # f4 and f5 when num_dims =3 or 4 I would be so grateful.
            # Just could not figure this out.  Transposing works but why!?
            for f in range(8,11):
                i1 = np.arange(0, llc)+(f-8)*llc
                i2 = np.arange(0,3*llc,3) + 7*llc + f - 8

                data_compact[k,i2,:] = f4[k,0:llc,i1].T

            for f in range(11,14):
                i1 = np.arange(0, llc)+(f-11)*llc
                i2 = np.arange(0,3*llc,3) + 10*llc + f -11
                data_compact[k,i2,:] = f5[k,:,i1].T

    elif num_dims == 4:
        # loop through l and k indices
        for l in range(nl):
            for k in range(nk):
                data_compact[l,k,:3*llc,:]      = f1[l,k,:]
                data_compact[l,k,3*llc:6*llc,:] = f2[l,k,:]
                data_compact[l,k,6*llc:7*llc,:] = f3[l,k,:]

                for f in range(8,11):
                    i1 = np.arange(0, llc)+(f-8)*llc
                    i2 = np.arange(0,3*llc,3) + 7*llc + f -8
                    data_compact[l,k,i2,:]      = f4[l,k,:,i1].T

                for f in range(11,14):
                    i1 = np.arange(0, llc)+(f-11)*llc
                    i2 = np.arange(0,3*llc,3) + 10*llc + f -11
                    data_compact[l,k,i2,:]      = f5[l,k,:,i1].T


    if not less_output:
        print ('llc_faces_to_compact: data_compact array shape', data_compact.shape)
        print ('llc_faces_to_compact: data_compact array dtype', data_compact.dtype)

    return data_compact

#%%
def llc_tiles_to_xda(data_tiles, var_type=None, grid_da=None, less_output=True,
                     dim4=None,dim5=None):
    """
    Convert numpy or dask array in tiled format to xarray DataArray
    with minimal coordinates: (time,k,tile,j,i) ; (time,k,tile,j_g,i) etc...
    unless a DataArray or Dataset is provided as a template
    to provide more coordinate info

    4D field (5D array with tile dimension) Example:
    A 4D field (3D in space and 1D in time) living on tracer points with
    dimension order resulting from read_bin_llc.read_llc_to_tiles:

       >> array.shape
       [N_tiles, N_recs, N_z, N_y, N_x]

    We would read this in as follows:

        >> xda = llc_tiles_to_xda(data_tiles=array, var_type='c',
                                  dim4='depth', dim5='time')

    or equivalently

        >> xda = llc_tiles_to_xda(data_tiles=array, var_type='c',
                                  dim4='k', dim5='time')

    since 'depth' has been coded to revert to vertical coordinate 'k'...

    Note:
    1. for the 3D case, dim5 is not necessary
    2. for the 2D case, dim4 and dim5 are not necessary

    Special case!
    data_tiles can also be a 1D array ONLY if the user provides
    grid_da as a template for how to shape it back to a numpy array, then
    to DataArray.
    See calc_section_trsp._rotate_the_grid for an example usage.

    Parameters
    ----------
    data_tiles : numpy or dask+numpy array
        see above for specified dimension order

    var_type : string, optional
        Note: only optional if grid_da is provided!
        specification for where on the grid the variable lives
        'c' - grid cell center, i.e. tracer point, e.g. XC, THETA, ...
        'w' - west grid cell edge, e.g. dxG, zonal velocity, ...
        's' - south grid cell edge, e.g. dyG, meridional velocity, ...
        'z' - southwest grid cell edge, zeta/vorticity point, e.g. rAz

    grid_da : xarray DataArray, optional
        a DataArray or Dataset with the grid coordinates already loaded

    less_output : boolean, optional
        A debugging flag.  False = less debugging output

    dim4, dim5 : string, optional
        Specify name of fourth and fifth dimension, e.g. 'depth', 'k', or 'time'

    Returns
    -------
    da : xarray DataArray
    """

    if var_type is None and grid_da is None:
        raise TypeError('Must specify var_type="c","w","s", or "z" if grid_da is not provided')

    # Test for special case: 1D data
    if len(data_tiles.shape)==1:
        if grid_da is None:
            raise TypeError('If converting 1D array, must specify grid_da as template')

        if not less_output:
            print('Found 1D array, will use grid_da input to shape it')
    elif len(data_tiles.shape)>5:
        raise TypeError('Found unfamiliar array shape: ', data_tiles.shape)

    # If a DataArray or Dataset is given to model after, use this first!
    if grid_da is not None:

        # Add case for 1D array
        # This is like the gcmfaces routine convert2gcmfaces or convert2array
        # except it's practically two lines of code
        if len(data_tiles.shape)==1:
            data_tiles = np.reshape(data_tiles, np.shape(grid_da.values))

        # don't copy over attributes from grid_da.  Let user specify own attributes
        da = xr.DataArray(data=data_tiles,
                          coords=grid_da.coords.variables,
                          dims=grid_da.dims,
                          attrs=dict())
        return da

    # Provide dims and coords based on grid location
    if var_type == 'c':
        da = _make_data_array(data_tiles,'i','j','k',less_output,dim4,dim5)

    elif var_type == 'w':
        da = _make_data_array(data_tiles,'i_g','j','k',less_output,dim4,dim5)

    elif var_type == 's':
        da = _make_data_array(data_tiles,'i','j_g','k',less_output,dim4,dim5)

    elif var_type == 'z':
        da = _make_data_array(data_tiles,'i_g','j_g','k',less_output,dim4,dim5)

    else:
        raise NotImplementedError("Can only take 'c', 'w', 's', or 'z', other types not implemented.")

    return da

#%%
def _make_data_array(data_tiles, iVar, jVar, kVar, less_output=False,dim4=None,dim5=None):
    """Non user facing function to make a data array from tiled numpy/dask array
    and strings denoting grid location

    Note that here, I'm including the "tiles" dimension...
    so dim4 refers to index vector d_4, and dim5 refers to index d_5
    No user should have to deal with this though

    Parameters
    ----------
    data_tiles : numpy/dask array
        Probably loaded from binary via mds_io.read_bin_to_tiles and rearranged
        in llc_tiles_to_xda
    iVar : string
        denote x grid location, 'i' or 'i_g'
    jVar : string
        denote y grid location, 'j' or 'j_g'
    kVar : string
        denote x grid location, 'k' only implemented for now.
        possible to implement 'k_u' for e.g. vertical velocity ... at some point
    less_output : boolean, optional
        debugging flag, False => print more
    dim4, dim5 : string, optional
        Specify name of fourth and fifth dimension, e.g. 'depth', 'k', or 'time'

    Returns
    -------
    da : xarray DataArray
    """

    # Save shape and num dimensions for below
    data_shape = data_tiles.shape
    Ndims = len(data_shape)

    # Create minimal coordinate information
    i = np.arange(data_shape[-1])
    j = np.arange(data_shape[-2])
    tiles = np.arange(data_shape[-3])
    d_4 = []
    d_5 = []
    if len(data_shape)>3:
        if dim4 is None:
            raise TypeError("Please specify 4th dimension as dim4='depth' or dim4='time'")
        d_4 = np.arange(data_shape[-4])

    if len(data_shape)>4:
        if dim5 is None:
            raise TypeError("Please specify 5th dimension as dim5='depth' or dim5='time'")
        d_5 = np.arange(data_shape[-5])

    # Create dims tuple, which will at least have
    # e.g. ('tile','j','i') for a 'c' variable
    dims = ('tile',jVar,iVar)

    # Coordinates will be a dictionary of 1 dimensional xarray DataArrays
    # each with their own set of attributes
    coords = OrderedDict()
    if Ndims>3:
        if dim4=='depth':
            mydim = kVar
        else:
            mydim = dim4

        dims = (mydim,) + dims
        attrs = dimensions[mydim] if mydim in dimensions else {}
        xda4 = xr.DataArray(data=d_4, coords={mydim: d_4},
                            dims=(mydim,),attrs=attrs)
        coords[mydim] = xda4

    if Ndims>4:
        if dim5=='depth':
            mydim = kVar
        else:
            mydim = dim5

        dims = (mydim,) + dims
        attrs = dimensions[mydim] if mydim in dimensions else {}
        xda5 = xr.DataArray(data=d_5, coords={mydim: d_5},
                            dims=(mydim,),attrs=attrs)
        coords[mydim] = xda5

    # Now add the standard coordinates
    tile_da = xr.DataArray(data=tiles, coords={'tile': tiles},
                           dims=('tile',),
                           attrs=OrderedDict([('standard_name','tile_index')]))
    j_da = xr.DataArray(data=j,coords={jVar: j},dims=(jVar,),
                        attrs=dimensions[jVar]['attrs'])
    i_da = xr.DataArray(data=i, coords={iVar: i}, dims=(iVar,),
                        attrs=dimensions[iVar]['attrs'])

    coords['tile'] = tile_da
    coords[jVar] = j_da
    coords[iVar] = i_da

    return xr.DataArray(data=data_tiles, coords=coords, dims=dims)
