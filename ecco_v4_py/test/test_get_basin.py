import os
import numpy as np
import ecco_v4_py

from .test_common import llc_mds_datadirs, get_test_ds

_test_dir = os.path.dirname(os.path.abspath(__file__))

def test_each_basin_masks(get_test_ds):
    """make sure we can make the basin masks
    """

    ds = get_test_ds
    all_basins = ecco_v4_py.read_llc_to_tiles(os.path.join(_test_dir,'../../binary_data'),'basins.data',less_output=True)
    ext_names = ['atlExt','pacExt','indExt']
    for i,basin in enumerate(ecco_v4_py.get_available_basin_names(),start=1):
        mask = ecco_v4_py.get_basin_mask(basin,ds.maskC.isel(k=0))
        assert np.all(mask.values == (all_basins==i))

def test_ext_basin_masks(get_test_ds):
    """make sure we can make the extended masks
    """

    ds = get_test_ds

    ext_names = ['atlExt','pacExt','indExt']
    individual_names = [['atl','mexico','hudson','med','north','baffin','gin'],
                        ['pac','bering','okhotsk','japan','eastChina'],
                        ['ind','southChina','java','timor','red','gulf']]
    for ext,ind in zip(ext_names,individual_names):
        maskE = ecco_v4_py.get_basin_mask(ext,ds.maskC.isel(k=0))
        maskI = ecco_v4_py.get_basin_mask(ind,ds.maskC.isel(k=0))
        assert np.all(maskE==maskI)
