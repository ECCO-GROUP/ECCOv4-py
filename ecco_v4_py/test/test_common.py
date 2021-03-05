"""
Helper functions for all tests
"""
import pytest
from xmitgcm.utils import get_extra_metadata
from xmitgcm.test.test_xmitgcm_common import (
        _experiments, llc_mds_datadirs, setup_mds_dir, dlroot )
import ecco_v4_py as ecco

# Following xmitgcm's lead to add an ASTE domain for testing
_experiments['aste270']= {'geometry':'llc',
                          'dlink': dlroot + '25286756',
                          'md5': 'f616fe46330f1125472f274af2c96e44',
                          'shape': (50,6,270,270),
                          'ref_date':'2002-01-01 00:00:00',
                          'diagnostics':('state_2d_set1',
                            ['ETAN    ', 'SIarea  ', 'SIheff  ', 'SIhsnow ',
                            'DETADT2 ', 'PHIBOT  ', 'sIceLoad', 'MXLDEPTH',
                            'oceSPDep', 'SIatmQnt', 'SIatmFW ', 'oceQnet ',
                            'oceFWflx', 'oceTAUX ', 'oceTAUY ', 'ADVxHEFF',
                            'ADVyHEFF', 'DFxEHEFF', 'DFyEHEFF', 'ADVxSNOW',
                            'ADVySNOW', 'DFxESNOW', 'DFyESNOW', 'SIuice  ',
                            'SIvice  ' 'ETANSQ  ']),
                          'test_iternum':8}

mds_dir_info = None
llc_dir_info= None

# Modify xmitgcm's function for both global ECCO and ASTE
@pytest.fixture(scope='module', params=['global_oce_llc90','aste270'])
def all_mds_datadirs(tmpdir_factory, request):

    global mds_dir_info
    if mds_dir_info == None:
       mds_dir_info = setup_mds_dir(tmpdir_factory,request, _experiments)
   
    print('-------- made mds_dirr_info ')
    print(mds_dir_info) 
    return mds_dir_info
    

@pytest.fixture(scope='module')
def get_test_ds(all_mds_datadirs):
    """make 2 tests when called, one with global, one with ASTE,
    using fixture above"""

    dirname, expected = all_mds_datadirs

    kwargs = {}
    if 'aste' in dirname:
        kwargs['extra_metadata']=get_extra_metadata('aste',270)
        kwargs['tiles_to_load']=[0,1,2,3,4,5]
        kwargs['nx']=270
        domain = 'aste'
    else:
        domain = 'global'

    # read in array
    ds = ecco.load_ecco_vars_from_mds(dirname,
            model_time_steps_to_load=expected['test_iternum'],
            mds_files=['state_2d_set1','U','V','W','T','S'],
            **kwargs)
    ds.load()
    ds.attrs['domain'] = domain
    return ds

@pytest.fixture(scope='module')
def get_global_ds(llc_mds_datadirs):
    """just get the global dataset"""

    global llc_dir_info
    if llc_dir_info == None:
        dirname, expected = llc_mds_datadirs
        llc_dir_info = [dirname, expected]
    else:
        dirname = llc_dir_info[0]
        expected = llc_dir_info[1] 

    # read in array
    ds = ecco.load_ecco_vars_from_mds(dirname,
            model_time_steps_to_load=expected['test_iternum'],
            mds_files=['state_2d_set1','U','V','W','T','S'])
    ds.load()
    return ds

@pytest.fixture(scope='module')
def get_test_array_2d(llc_mds_datadirs):
    """download, unzip and return 2D field"""
    global llc_dir_info
    if llc_dir_info == None:
        dirname, expected = llc_mds_datadirs
        llc_dir_info = [dirname, expected]
    else:
        dirname = llc_dir_info[0]
        expected = llc_dir_info[1] 
    
    # read in array
    ds = ecco.load_ecco_vars_from_mds(dirname,
            model_time_steps_to_load=expected['test_iternum'],
            mds_files='state_2d_set1')

    ds.load()
    xda = ds['ETAN']

    if 'time' in xda.dims:
        xda = xda.isel(time=0)
    return xda
