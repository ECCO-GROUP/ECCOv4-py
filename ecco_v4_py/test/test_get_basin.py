import ecco_v4_py

from .test_common import llc_mds_datadirs, get_test_ds

def test_all_basin_masks(get_test_ds):
    """make sure we can make the basin masks
    """

    ds = get_test_ds
    for basin in ecco_v4_py.get_available_basin_names():
        assert ecco_v4_py.get_basin_mask(basin,ds.maskC.isel(k=0)) is not None
