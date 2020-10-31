import warnings
from datetime import datetime
import numpy as np
import xarray as xr
import pytest
from ecco_v4_py import load_ecco_vars_from_mds

from .test_common import llc_mds_datadirs

@pytest.mark.parametrize("iters",['all',8,[0,8]])
@pytest.mark.parametrize("freq",['','AVG_MON','AVG_DAY','AVG_YEAR','SNAPSHOT'])
@pytest.mark.parametrize("vars_to_load",['all','U',['U','V','T']])
def test_read_mdsdataset(llc_mds_datadirs,iters,freq,vars_to_load):
    """honestly not sure what all of these options are for
    but just make sure this works"""

    data_dir,_ = llc_mds_datadirs
    ds = load_ecco_vars_from_mds(data_dir,
                                 mds_files=['U','V','W','T','S'],
                                 vars_to_load=vars_to_load,
                                 model_time_steps_to_load=iters,
                                 output_freq_code=freq)
