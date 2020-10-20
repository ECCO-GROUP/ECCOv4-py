"""
Test routines for the vector calculations module
"""
from __future__ import division, print_function
import warnings
from pathlib import Path
import numpy as np
import xarray as xr
import pytest
from ecco_v4_py import vector_calc

from .test_common import llc_mds_datadirs, get_test_vectors

def test_no_angles(get_test_vectors):
    """quick error handling test"""

    ds = get_test_vectors
    ds = ds.drop(['CS','SN'])
    with pytest.raises(KeyError):
        vector_calc.UEVNfromUXVY(ds['UVELMASS'],ds['VVELMASS'],ds)

