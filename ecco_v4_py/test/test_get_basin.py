import os
import numpy as np
import ecco_v4_py

from .test_common import llc_mds_datadirs, get_test_ds

_test_dir = os.path.dirname(os.path.abspath(__file__))

def test_all_basin_masks(get_test_ds):
    """make sure we can make the basin masks
    """

    ds = get_test_ds
    all_basins = ecco_v4_py.read_llc_to_tiles(os.path.join(_test_dir,'../../binary_data'),'basins.data',less_output=True)
    for i,basin in enumerate(ecco_v4_py.get_available_basin_names(),start=1):
        mask = ecco_v4_py.get_basin_mask(basin,ds.maskC.isel(k=0))
        assert np.all(mask.values == (all_basins==i))
