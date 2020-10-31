
from __future__ import division, print_function
import warnings
from pathlib import Path
import numpy as np
import xarray as xr
import pytest
import ecco_v4_py as ecco

from .test_common import llc_mds_datadirs,get_global_ds

# Define bin directory for test reading
_PKG_DIR = Path(__file__).resolve().parent.parent.parent
_DATA_DIR = _PKG_DIR.joinpath('binary_data')

_basin =    (_PKG_DIR.joinpath('binary_data'),'basins.data',1,1,0)
_hfac  =    ('xmitgcm','hFacC.data',50,1,0)
_state2d  = ('xmitgcm','state_2d_set1.0000000008.data',1,25,0)
_state3d  = ('xmitgcm','state_2d_set1.0000000008.data',5,5,0)
_state2dsingle  = ('xmitgcm','state_2d_set1.0000000008.data',1,1,24)
_skip2d  = ('xmitgcm','state_2d_set1.0000000008.data',1,20,5)
_skip3d  = ('xmitgcm','state_2d_set1.0000000008.data',5,4,2)
_m1noskip  = ('xmitgcm','state_2d_set1.0000000008.data',-1,1,0)
_m1skip  = ('xmitgcm','state_2d_set1.0000000008.data',-1,1,5)
_m1single = ('xmitgcm','state_2d_set1.0000000008.data',-1,1,24)
_m1_kerr = ('xmitgcm','state_2d_set1.0000000008.data',-10,1,0)
_m1_lerr = ('xmitgcm','state_2d_set1.0000000008.data',1,-10,0)
_nofile = (_PKG_DIR.joinpath('binary_data'),'myfile',1,1,0)

# Test convert from tiles #
###########################
@pytest.mark.parametrize("mydir, fname, nk, nl, skip",
                         [_basin,_hfac,_state2d,_state3d,_skip2d,_skip3d])
@pytest.mark.parametrize("use_xmitgcm",[True,False])
def test_convert_tiles_to_faces(llc_mds_datadirs,mydir,fname,nk,nl,skip,
                                use_xmitgcm):
    """Read in tiles data, convert to faces.
    Verify this equals reading in face data
    """

    if mydir == 'xmitgcm':
        mydir,_ = llc_mds_datadirs

    data_tiles = ecco.read_llc_to_tiles(fdir=mydir,
                                        fname=fname,
                                        llc=90, nk=nk, nl=nl,
                                        skip=skip,
                                        filetype='>f4',
                                        less_output=False,use_xmitgcm=False)

    data_faces = ecco.read_llc_to_faces(fdir=mydir,
                                        fname=fname,
                                        llc=90, nk=nk, nl=nl,
                                        skip=skip,
                                        filetype='>f4',
                                        less_output=False)

    data_converted = ecco.llc_tiles_to_faces(data_tiles)
    for f in range(1,len(data_faces)+1):
        assert np.all(np.equal( data_converted[f], data_faces[f] ))

@pytest.mark.parametrize("mydir, fname, nk, nl, skip",
                         [_basin,_hfac,_state2d,_state3d,_skip2d,_skip3d])
@pytest.mark.parametrize("use_xmitgcm",[True,False])
def test_convert_tiles_to_compact(llc_mds_datadirs,mydir,fname,nk,nl,skip,
                                  use_xmitgcm):
    """Read in tiles data, convert to compact.
    Verify this equals reading in compact data
    """

    if mydir == 'xmitgcm':
        mydir,_ = llc_mds_datadirs

    data_tiles = ecco.read_llc_to_tiles(fdir=mydir,
                                        fname=fname,
                                        llc=90, nk=nk, nl=nl, filetype='>f4',
                                        skip=skip,
                                        less_output=False,
                                        use_xmitgcm=False)
    data_compact = ecco.read_llc_to_compact(fdir=mydir,
                                            fname=fname,
                                            llc=90, nk=nk, nl=nl, filetype='>f4',
                                            skip=skip,
                                            less_output=False)

    data_converted = ecco.llc_tiles_to_compact(data_tiles)
    assert np.all(np.equal( data_converted, data_compact ))

@pytest.mark.parametrize("mydir, fname, nk, nl, skip",
                         [_basin,_hfac,_state2d,_state3d,_skip2d,_skip3d])
@pytest.mark.parametrize("grid_da",[None,True])
@pytest.mark.parametrize("var_type",['c','w','s','z'])
@pytest.mark.parametrize("use_xmitgcm",[True,False])
def test_convert_tiles_to_xda(llc_mds_datadirs,get_global_ds,mydir,fname,nk,nl, skip,
                              grid_da, var_type, use_xmitgcm):

    if mydir == 'xmitgcm':
        mydir,_ = llc_mds_datadirs

    ds = get_global_ds

    data_tiles = ecco.read_llc_to_tiles(fdir=mydir,
                                fname=fname,
                                llc=90, nk=nk, nl=nl, filetype='>f4',
                                skip=skip,
                                less_output=False,
                                use_xmitgcm=use_xmitgcm)
    if grid_da:
        grid_da = ds[f'hFac{var_type.upper()}'].isel(k=0) if var_type != 'z' else ds['XG']
        if nk>1:
            recdim = xr.DataArray(np.arange(nk),{'k':np.arange(nk)},('k',))
            grid_da = grid_da.broadcast_like(recdim)
        if nl>1:
            recdim = xr.DataArray(np.arange(nl),
                                  {'time':np.arange(nl)},('time',))
            grid_da = grid_da.broadcast_like(recdim)
    if nk>1 and nl>1:
        d4='k'
        d5='time'
    elif nk>1 and nl==1:
        d4='k'
        d5=None
    elif nl>1:
        d4='time'
        d5=None
    else:
        d4=None
        d5=None

    xda = ecco.llc_tiles_to_xda(np.squeeze(data_tiles),
                                var_type='c',grid_da=grid_da,
                                less_output=False,dim4=d4,dim5=d5)

    assert np.all(xda.values == np.squeeze(data_tiles))

# Test convert from compact #
#############################
@pytest.mark.parametrize("mydir, fname, nk, nl, skip",
                         [_basin,_hfac,_state2d,_state3d,_skip2d,_skip3d])
def test_convert_compact_to_faces(llc_mds_datadirs,mydir,fname,nk,nl,skip):
    """Read in compact data, convert to faces.
    Verify this equals reading in face data
    """
    if mydir == 'xmitgcm':
        mydir,_ = llc_mds_datadirs

    data_compact = ecco.read_llc_to_compact(fdir=mydir,
                                            fname=fname,
                                            llc=90, nk=nk, nl=nl, filetype='>f4',
                                            skip=skip,
                                            less_output=False)
    data_faces = ecco.read_llc_to_faces(fdir=mydir,
                                        fname=fname,
                                        llc=90, nk=nk, nl=nl, filetype='>f4',
                                        skip=skip,
                                        less_output=False)

    data_converted = ecco.llc_compact_to_faces(data_compact)
    for f in range(1,len(data_faces)+1):
        assert np.all(np.equal( data_converted[f], data_faces[f] ))

@pytest.mark.parametrize("mydir, fname, nk, nl, skip",
                         [_basin,_hfac,_state2d,_state3d,_skip2d,_skip3d])
def test_convert_compact_to_tiles(llc_mds_datadirs,mydir,fname,nk,nl,skip):
    """Read in compact data, convert to tiles.
    Verify this equals reading in face data
    """

    if mydir == 'xmitgcm':
        mydir,_ = llc_mds_datadirs

    data_compact = ecco.read_llc_to_compact(fdir=mydir,
                                            fname=fname,
                                            llc=90, nk=nk, nl=nl, filetype='>f4',
                                            skip=skip,
                                            less_output=False)
    data_tiles = ecco.read_llc_to_tiles(fdir=mydir,
                                        fname=fname,
                                        llc=90, nk=nk, nl=nl, filetype='>f4',
                                        skip=skip,
                                        less_output=False)

    data_converted = ecco.llc_compact_to_tiles(data_compact)
    assert np.all(np.equal( data_converted, data_tiles ))

# Test convert from faces #
###########################
@pytest.mark.parametrize("mydir, fname, nk, nl, skip",
                         [_basin,_hfac,_state2d,_state3d,_skip2d,_skip3d])
def test_convert_faces_to_tiles(llc_mds_datadirs,mydir,fname,nk,nl,skip):
    """Read in faces data, convert to tiles.
    Verify this equals reading in face data
    """

    if mydir == 'xmitgcm':
        mydir,_ = llc_mds_datadirs

    data_faces = ecco.read_llc_to_faces(fdir=mydir,
                                        fname=fname,
                                        llc=90, nk=nk, nl=nl, filetype='>f4',
                                        skip=skip,
                                        less_output=False)
    data_tiles = ecco.read_llc_to_tiles(fdir=mydir,
                                        fname=fname,
                                        llc=90, nk=nk, nl=nl, filetype='>f4',
                                        skip=skip,
                                        less_output=False)

    data_converted = ecco.llc_faces_to_tiles(data_faces)
    assert np.all(np.equal( data_converted, data_tiles ))

@pytest.mark.parametrize("mydir, fname, nk, nl, skip",
                         [_basin,_hfac,_state2d,_state3d,_skip2d,_skip3d])
def test_convert_faces_to_compact(llc_mds_datadirs,mydir,fname,nk,nl,skip):
    """Read in faces data, convert to compact.
    Verify this equals reading in compact data
    """

    if mydir == 'xmitgcm':
        mydir,_ = llc_mds_datadirs

    data_faces = ecco.read_llc_to_faces(fdir=mydir,
                                        fname=fname,
                                        llc=90, nk=nk, nl=nl, filetype='>f4',
                                        skip=skip,
                                        less_output=False)
    data_compact = ecco.read_llc_to_compact(fdir=mydir,
                                            fname=fname,
                                            skip=skip,
                                            llc=90, nk=nk, nl=nl, filetype='>f4',
                                            less_output=False)

    data_converted = ecco.llc_faces_to_compact(data_faces)
    assert np.all(np.equal( data_converted, data_compact ))

### test read -1 nl
@pytest.mark.parametrize("test, expected",
                        [(_m1noskip,_state2d),
                         (_m1skip,_skip2d),
                         (_m1single,_state2dsingle),
                         (_m1_kerr,TypeError),
                         (_m1_lerr,TypeError),
                         (_nofile,IOError)])
@pytest.mark.parametrize("myfunc",
        [   ecco.read_llc_to_compact,
            ecco.read_llc_to_faces,
            ecco.read_llc_to_tiles])
@pytest.mark.parametrize("nx",[90,-1])
def test_read_m1_recs(llc_mds_datadirs,test,expected,myfunc,nx):
    mydir,_ = llc_mds_datadirs
    if isinstance(expected,tuple) and nx==90:
        data_expected = myfunc(fdir=mydir,
                               fname=expected[1],
                               llc=nx,
                               nk=expected[2],
                               nl=expected[3],
                               filetype='>f4',
                               skip=expected[4])

        data_test = myfunc(fdir=mydir,
                               fname=test[1],
                               llc=nx,
                               nk=test[2],
                               nl=test[3],
                               filetype='>f4',
                               skip=test[4])

        if len(data_expected)==5:
            for i in range(len(data_expected)):
                assert np.all(np.equal(data_test[i+1],np.squeeze(data_expected[i+1])))
        else:
            assert np.all(np.equal(data_test,np.squeeze(data_expected)))
    else:
        expected = expected if isinstance(expected,type) else TypeError
        with pytest.raises(expected):
            data_test = myfunc(fdir=mydir,
                               fname=test[1],
                               llc=nx,
                               nk=test[2],
                               nl=test[3],
                               filetype='>f4',
                               skip=test[4])

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
        (np.zeros((1,1,1,1,1,1)),'c'),
        (np.zeros((1,1,1)),None),
        (np.zeros((1,1,1)),'f'),
        (np.zeros((1,1,1,1)),'c'),
        (np.zeros((1,1,1,1)),'w'),
        (np.zeros((1,1,1,1)),'s'),
        (np.zeros((1,1,1,1)),'z'),
        (np.zeros((1,1,1,1,1)),'cc'),
        ])
def test_xda5d(test,var_type):
    myerr = NotImplementedError if var_type=='f' and test.shape==(1,1,1) else TypeError
    if var_type=='cc':
        var_type='c'
        dim4='depth'
    else:
        dim4=None
    with pytest.raises(myerr):
        ecco.llc_tiles_to_xda(test,var_type=var_type,dim4=dim4)
