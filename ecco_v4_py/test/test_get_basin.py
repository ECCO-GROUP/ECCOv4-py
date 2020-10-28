import os
import shutil
import numpy as np
import ecco_v4_py
import pytest

from .test_common import (
        llc_mds_datadirs, get_global_ds,
        all_mds_datadirs, get_test_ds)

_test_dir = os.path.dirname(os.path.abspath(__file__))

# test error out for different domains
def test_notimplemented(get_test_ds):
    ds = get_test_ds
    with pytest.raises(NotImplementedError):
        if len(ds.tile)<13:
            ecco_v4_py.get_basin_mask('atl',ds.maskC)
        else:
            ecco_v4_py.get_basin_mask('atl',ds.sel(tile=0).maskC)
            ecco_v4_py.get_basin_mask('atl',(1.*ds.maskC).diff(dim='i'))
            ecco_v4_py.get_basin_mask('atl',(1.*ds.maskC).diff(dim='j'))

def test_each_basin_masks(get_global_ds):
    """make sure we can make the basin masks
    """

    ds = get_global_ds
    all_basins = ecco_v4_py.read_llc_to_tiles(os.path.join(_test_dir,'..','..','binary_data'),'basins.data',less_output=True)
    ext_names = ['atlExt','pacExt','indExt']
    for i,basin in enumerate(ecco_v4_py.get_available_basin_names(),start=1):
        mask = ecco_v4_py.get_basin_mask(basin,ds.maskC.isel(k=0))
        assert np.all(mask.values == (all_basins==i))

def test_ext_basin_masks(get_global_ds):
    """make sure we can make the extended masks
    """

    ds = get_global_ds

    ext_names = ['atlExt','pacExt','indExt']
    individual_names = [['atl','mexico','hudson','med','north','baffin','gin'],
                        ['pac','bering','okhotsk','japan','eastChina'],
                        ['ind','southChina','java','timor','red','gulf']]
    for ext,ind in zip(ext_names,individual_names):
        maskE = ecco_v4_py.get_basin_mask(ext,ds.maskC.isel(k=0))
        maskI = ecco_v4_py.get_basin_mask(ind,ds.maskC.isel(k=0))
        assert np.all(maskE==maskI)

def test_3d(get_global_ds):
    """check that vertical coordinate"""

    ds = get_global_ds
    grid = ecco_v4_py.get_llc_grid(ds)
    maskK = ds['maskC']
    maskL = grid.interp(maskK,'Z',to='left',boundary='fill')
    maskU = grid.interp(maskK,'Z',to='right',boundary='fill')
    maskKp1 = ds.maskC.isel(k=0).broadcast_like(ds.k_p1)
    for mask in [maskK,maskL,maskU,maskKp1]:
        ecco_v4_py.get_basin_mask('atl',mask)

def test_bin_dir_is_here(get_global_ds,hide_bin_dir):

    hide_bin_dir
    ds = get_global_ds
    with pytest.raises(OSError):
        ecco_v4_py.get_basin_mask('atl',ds.maskC.isel(k=0))



@pytest.fixture(scope='function')
def hide_bin_dir():

    bin_path = os.path.join(_test_dir,'..','..','binary_data')
    tmp_path = os.path.join(_test_dir,'..','..','tmp_bin')
    shutil.move(bin_path,tmp_path)

    yield

    shutil.move(tmp_path,bin_path)
