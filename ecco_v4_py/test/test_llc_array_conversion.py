
from __future__ import division, print_function
import warnings
from pathlib import Path
import numpy as np
import pytest
import ecco_v4_py as ecco

from .test_common import llc_mds_datadirs

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
                                        less_output=False)
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
                                        less_output=False)
    data_compact = ecco.read_llc_to_compact(fdir=mydir,
                                            fname=fname,
                                            llc=90, nk=nk, nl=nl, filetype='>f4',
                                            less_output=False)

    data_converted = ecco.llc_tiles_to_compact(data_tiles)
    assert np.all(np.equal( data_converted, data_compact ))

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
