
from __future__ import division, print_function
import warnings
from pathlib import Path
import numpy as np
import xarray as xr
import pytest
import ecco_v4_py as ecco

from .test_common import llc_mds_datadirs,get_test_ds

# Define bin directory for test reading
_PKG_DIR = Path(__file__).resolve().parent.parent.parent
_DATA_DIR = _PKG_DIR.joinpath('binary_data')

_basin =    (_PKG_DIR.joinpath('binary_data'),'basins.data',1,1)
_hfac  =    ('xmitgcm','hFacC.data',50,1)
_state2d  = ('xmitgcm','state_2d_set1.0000000008.data',1,25)
_state3d  = ('xmitgcm','state_2d_set1.0000000008.data',5,5)

# Test convert from tiles #
###########################
@pytest.mark.parametrize("mydir, fname, nk, nl",[_basin,_hfac,_state2d,_state3d])
def test_convert_tiles_to_faces(llc_mds_datadirs,mydir,fname,nk,nl):
    """Read in tiles data, convert to faces.
    Verify this equals reading in face data
    """

    if mydir == 'xmitgcm':
        mydir,_ = llc_mds_datadirs

    data_tiles = ecco.read_llc_to_tiles(fdir=mydir,
                                        fname=fname,
                                        llc=90, nk=nk, nl=nl, filetype='>f4',
                                        less_output=False,use_xmitgcm=False)

    data_faces = ecco.read_llc_to_faces(fdir=mydir,
                                        fname=fname,
                                        llc=90, nk=nk, nl=nl, filetype='>f4',
                                        less_output=False)

    data_converted = ecco.llc_tiles_to_faces(data_tiles)
    for f in range(1,len(data_faces)+1):
        assert np.all(np.equal( data_converted[f], data_faces[f] ))

@pytest.mark.parametrize("mydir, fname, nk, nl",[_basin,_hfac,_state2d,_state3d])
def test_convert_tiles_to_compact(llc_mds_datadirs,mydir,fname,nk,nl):
    """Read in tiles data, convert to compact.
    Verify this equals reading in compact data
    """

    if mydir == 'xmitgcm':
        mydir,_ = llc_mds_datadirs

    data_tiles = ecco.read_llc_to_tiles(fdir=mydir,
                                        fname=fname,
                                        llc=90, nk=nk, nl=nl, filetype='>f4',
                                        less_output=False,
                                        use_xmitgcm=False)
    data_compact = ecco.read_llc_to_compact(fdir=mydir,
                                            fname=fname,
                                            llc=90, nk=nk, nl=nl, filetype='>f4',
                                            less_output=False)

    data_converted = ecco.llc_tiles_to_compact(data_tiles)
    assert np.all(np.equal( data_converted, data_compact ))

@pytest.mark.parametrize("mydir, fname, nk, nl",[_basin,_hfac,_state2d])
@pytest.mark.parametrize("grid_da",[None,True])
def test_convert_tiles_to_xda(llc_mds_datadirs,get_test_ds,mydir,fname,nk,nl,grid_da):

    if mydir == 'xmitgcm':
        mydir,_ = llc_mds_datadirs

    ds = get_test_ds
    data_tiles = ecco.read_llc_to_tiles(fdir=mydir,
                                        fname=fname,
                                        llc=90, nk=nk, nl=nl, filetype='>f4',
                                        less_output=False,
                                        use_xmitgcm=False)

    if grid_da:
        grid_da = ds['Depth'] if nk==1 else ds['hFacC']
        if nl>1:
            recdim = xr.DataArray(np.arange(nl),{'time':np.arange(nl)},('time',))
            grid_da = grid_da.broadcast_like(recdim)
    if nk>1 and nl>1:
        d4='depth'
        d5='time'
    elif nk>1 and nl==1:
        d4='depth'
        d5=None
    elif nl>1:
        d4='time'
        d5=None
    else:
        d4=None
        d5=None

    xda = ecco.llc_tiles_to_xda(np.squeeze(data_tiles),var_type='c',grid_da=grid_da,
                           less_output=False,dim4=d4,dim5=d5)

    assert np.all(xda.values == np.squeeze(data_tiles))

# Test convert from compact #
#############################
@pytest.mark.parametrize("mydir, fname, nk, nl",[_basin,_hfac,_state2d,_state3d])
def test_convert_compact_to_faces(llc_mds_datadirs,mydir,fname,nk,nl):
    """Read in compact data, convert to faces.
    Verify this equals reading in face data
    """
    if mydir == 'xmitgcm':
        mydir,_ = llc_mds_datadirs

    data_compact = ecco.read_llc_to_compact(fdir=mydir,
                                            fname=fname,
                                            llc=90, nk=nk, nl=nl, filetype='>f4',
                                            less_output=False)
    data_faces = ecco.read_llc_to_faces(fdir=mydir,
                                        fname=fname,
                                        llc=90, nk=nk, nl=nl, filetype='>f4',
                                        less_output=False)

    data_converted = ecco.llc_compact_to_faces(data_compact)
    for f in range(1,len(data_faces)+1):
        assert np.all(np.equal( data_converted[f], data_faces[f] ))

@pytest.mark.parametrize("mydir, fname, nk, nl",[_basin,_hfac,_state2d,_state3d])
def test_convert_compact_to_tiles(llc_mds_datadirs,mydir,fname,nk,nl):
    """Read in compact data, convert to tiles.
    Verify this equals reading in face data
    """

    if mydir == 'xmitgcm':
        mydir,_ = llc_mds_datadirs

    data_compact = ecco.read_llc_to_compact(fdir=mydir,
                                            fname=fname,
                                            llc=90, nk=nk, nl=nl, filetype='>f4',
                                            less_output=False)
    data_tiles = ecco.read_llc_to_tiles(fdir=mydir,
                                        fname=fname,
                                        llc=90, nk=nk, nl=nl, filetype='>f4',
                                        less_output=False)

    data_converted = ecco.llc_compact_to_tiles(data_compact)
    assert np.all(np.equal( data_converted, data_tiles ))

# Test convert from faces #
###########################
@pytest.mark.parametrize("mydir, fname, nk, nl",[_basin,_hfac,_state2d,_state3d])
def test_convert_faces_to_tiles(llc_mds_datadirs,mydir,fname,nk,nl):
    """Read in faces data, convert to tiles.
    Verify this equals reading in face data
    """

    if mydir == 'xmitgcm':
        mydir,_ = llc_mds_datadirs

    data_faces = ecco.read_llc_to_faces(fdir=mydir,
                                        fname=fname,
                                        llc=90, nk=nk, nl=nl, filetype='>f4',
                                        less_output=False)
    data_tiles = ecco.read_llc_to_tiles(fdir=mydir,
                                        fname=fname,
                                        llc=90, nk=nk, nl=nl, filetype='>f4',
                                        less_output=False)

    data_converted = ecco.llc_faces_to_tiles(data_faces)
    assert np.all(np.equal( data_converted, data_tiles ))

@pytest.mark.parametrize("mydir, fname, nk, nl",[_basin,_hfac,_state2d,_state3d])
def test_convert_faces_to_compact(llc_mds_datadirs,mydir,fname,nk,nl):
    """Read in faces data, convert to compact.
    Verify this equals reading in compact data
    """

    if mydir == 'xmitgcm':
        mydir,_ = llc_mds_datadirs

    data_faces = ecco.read_llc_to_faces(fdir=mydir,
                                        fname=fname,
                                        llc=90, nk=nk, nl=nl, filetype='>f4',
                                        less_output=False)
    data_compact = ecco.read_llc_to_compact(fdir=mydir,
                                            fname=fname,
                                            llc=90, nk=nk, nl=nl, filetype='>f4',
                                            less_output=False)

    data_converted = ecco.llc_faces_to_compact(data_faces)
    assert np.all(np.equal( data_converted, data_compact ))


# Tests handling recognized unacceptable array sizes
@pytest.mark.parametrize("myfunc",
        [   ecco.llc_compact_to_faces,
            ecco.llc_faces_to_tiles,
            ecco.llc_tiles_to_faces,
            ecco.llc_faces_to_compact
        ])
def test_5d(myfunc):
    """this should create a pseudo error... print something and return
    an empty list"""

    arr = np.zeros((6,1,1,1,1,1))
    test = myfunc(arr)
    assert test==[]

# These all raise errors, for different reasons
@pytest.mark.parametrize("test, var_type",[
        (np.zeros(2), 'c'),
        (np.zeros((1,1,1,1,1)),'c'),
        (np.zeros((1,1,1)),None),
        (np.zeros((1,1,1)),'f')]
def test_xda5d(test,var_type):
    myerr = TypeError if var_type!='f' and test.shape!=(1,1,1) else NotImplementedError
    with pytest.raises(myerr):
        ecco.llc_tiles_to_xda(test,var_type=var_type)
