
from __future__ import division, print_function
import warnings
import os
import numpy as np
import pytest
import ecco_v4_py as ecco

# Define bin directory for test reading
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_PKG_DIR,'../../binary_data')

_TEST_FILES = ['basins.data', 'hFacC.data', 'state_3d_set1.0000000732.data']
_TEST_NK = [1, 50, 50]
_TEST_RECS = [1, 1, 3]


# Look for files #
##################
if not os.path.isfile(os.path.join(_DATA_DIR,_TEST_FILES[0])):

    warnings.warn('\n\nCannot find necessary binaries in ' + _DATA_DIR + '\n' +\
    'You can download all *.meta/data files needed for testing here:\n' +\
    '   https://github.com/ECCO-GROUP/ECCOv4-py/tree/master/binary_data \n' +\
    ' or \n' +\
    '   https://figshare.com/articles/Binary_files_for_the_ecco-v4-py_Python_package_/9932162 \n' +\
    'Download these files to ecco_v4_py/../binary_data/')
    pytest.skip("Test files not available.",allow_module_level=True)

# Test convert from tiles #
###########################
def test_convert_tiles_to_faces():
    """Read in tiles data, convert to faces. 
    Verify this equals reading in face data
    """

    # Loop through 2D, 3D, 4D cases
    for fname, nk, nl in zip(_TEST_FILES, _TEST_NK, _TEST_RECS):
        data_tiles = ecco.read_llc_to_tiles(fdir=_DATA_DIR, 
                                            fname=fname,
                                            llc=90, nk=nk, nl=nl, filetype='>f4',
                                            less_output=False)
        data_faces = ecco.read_llc_to_faces(fdir=_DATA_DIR, 
                                            fname=fname,
                                            llc=90, nk=nk, nl=nl, filetype='>f4',
                                            less_output=False)

        data_converted = ecco.llc_tiles_to_faces(data_tiles)
        for f in range(1,len(data_faces)+1):
            assert np.all(np.equal( data_converted[f], data_faces[f] ))

def test_convert_tiles_to_compact():
    """Read in tiles data, convert to compact. 
    Verify this equals reading in compact data
    """

    # Loop through 2D, 3D, 4D cases
    for fname, nk, nl in zip(_TEST_FILES, _TEST_NK, _TEST_RECS):
        data_tiles = ecco.read_llc_to_tiles(fdir=_DATA_DIR, 
                                            fname=fname,
                                            llc=90, nk=nk, nl=nl, filetype='>f4',
                                            less_output=False)
        data_compact = ecco.read_llc_to_compact(fdir=_DATA_DIR, 
                                                fname=fname,
                                                llc=90, nk=nk, nl=nl, filetype='>f4',
                                                less_output=False)

        data_converted = ecco.llc_tiles_to_compact(data_tiles)
        assert np.all(np.equal( data_converted, data_compact ))

# Test convert from compact #
#############################
def test_convert_compact_to_faces():
    """Read in compact data, convert to faces. 
    Verify this equals reading in face data
    """

    # Loop through 2D, 3D, 4D cases
    for fname, nk, nl in zip(_TEST_FILES, _TEST_NK, _TEST_RECS):
        data_compact = ecco.read_llc_to_compact(fdir=_DATA_DIR, 
                                                fname=fname,
                                                llc=90, nk=nk, nl=nl, filetype='>f4',
                                                less_output=False)
        data_faces = ecco.read_llc_to_faces(fdir=_DATA_DIR, 
                                            fname=fname,
                                            llc=90, nk=nk, nl=nl, filetype='>f4',
                                            less_output=False)

        data_converted = ecco.llc_compact_to_faces(data_compact)
        for f in range(1,len(data_faces)+1):
            assert np.all(np.equal( data_converted[f], data_faces[f] ))

def test_convert_compact_to_tiles():
    """Read in compact data, convert to tiles. 
    Verify this equals reading in face data
    """

    # Loop through 2D, 3D, 4D cases
    for fname, nk, nl in zip(_TEST_FILES, _TEST_NK, _TEST_RECS):
        data_compact = ecco.read_llc_to_compact(fdir=_DATA_DIR, 
                                                fname=fname,
                                                llc=90, nk=nk, nl=nl, filetype='>f4',
                                                less_output=False)
        data_tiles = ecco.read_llc_to_tiles(fdir=_DATA_DIR, 
                                            fname=fname,
                                            llc=90, nk=nk, nl=nl, filetype='>f4',
                                            less_output=False)

        data_converted = ecco.llc_compact_to_tiles(data_compact)
        assert np.all(np.equal( data_converted, data_tiles ))

# Test convert from faces #
###########################
def test_convert_faces_to_tiles():
    """Read in faces data, convert to tiles. 
    Verify this equals reading in face data
    """

    # Loop through 2D, 3D, 4D cases
    for fname, nk, nl in zip(_TEST_FILES, _TEST_NK, _TEST_RECS):
        data_faces = ecco.read_llc_to_faces(fdir=_DATA_DIR, 
                                            fname=fname,
                                            llc=90, nk=nk, nl=nl, filetype='>f4',
                                            less_output=False)
        data_tiles = ecco.read_llc_to_tiles(fdir=_DATA_DIR, 
                                            fname=fname,
                                            llc=90, nk=nk, nl=nl, filetype='>f4',
                                            less_output=False)

        data_converted = ecco.llc_faces_to_tiles(data_faces)
        assert np.all(np.equal( data_converted, data_tiles ))

def test_convert_faces_to_compact():
    """Read in faces data, convert to compact. 
    Verify this equals reading in compact data
    """

    # Loop through 2D, 3D, 4D cases
    for fname, nk, nl in zip(_TEST_FILES, _TEST_NK, _TEST_RECS):
        data_faces = ecco.read_llc_to_faces(fdir=_DATA_DIR, 
                                            fname=fname,
                                            llc=90, nk=nk, nl=nl, filetype='>f4',
                                            less_output=False)
        data_compact = ecco.read_llc_to_compact(fdir=_DATA_DIR, 
                                                fname=fname,
                                                llc=90, nk=nk, nl=nl, filetype='>f4',
                                                less_output=False)

        data_converted = ecco.llc_faces_to_compact(data_faces)
        assert np.all(np.equal( data_converted, data_compact ))
