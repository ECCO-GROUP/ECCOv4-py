#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 17:27:09 2019

@author: ifenty
"""

import sys
import json
import glob
import numpy as np

sys.path.append('/home/ifenty/ECCOv4-py')
import ecco_v4_py as ecco

#%%
######################################################################

## TIME STEPS TO LOAD
time_steps_to_load = [ 732 , 210360]
#time_steps_to_load = 'all'


## GRID DIR -- MAKE SURE USE GRID FIELDS WITHOUT BLANK TILES!!
#mds_grid_dir = '/nobackupp2/ifenty/grids/grid_llc90/no_blank_all'
mds_grid_dir = '/home/ifenty/data/grids/grid_llc90/no_blank_all'

## JSON DIR 
meta_json_dir = '/home/ifenty/ECCOv4-py/meta_json'

## MDS FILE TOP LEVEL DIRECTORY
#mds_diags_root_dir = '/nobackupp7/owang/runs/R3.summerschool/diags'
mds_diags_root_dir = '/home/ifenty/data/model_output/ECCOv4/R3_data_meta_v2/diags_monthly/'

## AVERAGING TIME PERIOD
output_freq_code = 'AVG_MON' ## 'AVG_DAY' or 'SNAPSHOT'

if 'AVG_MON' in output_freq_code:
    files = np.sort(glob.glob(mds_diags_root_dir + '*mon_mean*'))

elif 'AVG_DAY' in output_freq_code:
    files = np.sort(glob.glob(mds_diags_root_dir + '*day_mean*'))

elif 'SNAPSHOT' in output_freq_code:
    files = np.sort(glob.glob(mds_diags_root_dir + '*day_inst*'))
 
## WHICH VARIABLE TO LOAD
vars_to_load = 'all'

## OUTPUT DIRECTORY
output_dir = '/home/ifenty/data/model_output/ECCOv4/netcdf_test2/'

#%%

## START PROGRAM

# load METADATA
# --- common meta data
with open(meta_json_dir + '/ecco_meta_common.json', 'r') as fp:
    meta_common = json.load(fp)

# -- variable specific meta data
with open(meta_json_dir + '/ecco_meta_variable.json', 'r') as fp:    
    meta_variable_specific = json.load(fp)
              
#print (meta_common)
#print (meta_variable_specific)
    
    
## load variable file and directory names
print (len(files))
mds_diagnostic_filesets_to_load = []

for f in files:
    print (f)
    f_basename = str.split(f,'/')[-1]
    
    mds_diagnostic_filesets_to_load.append(f_basename)


print (mds_diagnostic_filesets_to_load)

#%%
## process each file, one at a time.
for mds_diagnostic_fileset_to_load in mds_diagnostic_filesets_to_load:
    
    mds_var_dir = mds_diags_root_dir + '/' + mds_diagnostic_fileset_to_load
    print ('processing ' , mds_diagnostic_fileset_to_load)


    ts = ecco.get_time_steps_from_mds_files(mds_var_dir, mds_diagnostic_fileset_to_load)
    print (mds_var_dir, mds_diagnostic_fileset_to_load)
            
    if 'all' in time_steps_to_load:
        print ('loading all time steps ')
        time_steps_to_load = ts
    
    else:
        print ('\nloading time steps : ', time_steps_to_load)
        
        
    edx, edxp = \
        ecco.create_nc_variable_files_on_native_grid_from_mds(mds_var_dir, 
                         mds_diagnostic_fileset_to_load,
                         mds_grid_dir,
                         output_dir,
                         output_freq_code=output_freq_code,
                         vars_to_load = vars_to_load,
                         tiles_to_load = [0,1,2,3,4,5,6,7,8,9,10,11,12],
                         time_steps_to_load = time_steps_to_load,
                         meta_variable_specific = dict(),
                         meta_common = dict(),
                         mds_datatype = '>f4',
                         method = 'time_interval_and_combined_tiles')

    print ('\nfinished processing ' + mds_diagnostic_fileset_to_load  + '\n\n')
   
#
