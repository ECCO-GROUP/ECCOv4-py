#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 15:13:33 2019

@author: ifenty
"""
from __future__ import division

import numpy as np
import matplotlib.pylab as plt

from llc_array_conversion  import llc_compact_to_tiles
from llc_array_conversion  import llc_compact_to_faces
from llc_array_conversion  import llc_faces_to_tiles
from llc_array_conversion  import llc_faces_to_compact
from llc_array_conversion  import llc_tiles_to_faces
from llc_array_conversion  import llc_tiles_to_compact


from mds_io import load_llc_compact
from mds_io import load_llc_compact_to_faces
from mds_io import load_llc_compact_to_tiles
from tile_plot import plot_tiles


# Tests the mds_io and llc_array_conversion routines
# %%
### Load model grid coordinates (longitude, latitude)


def run_mds_io_and_llc_conversion_test(llc_grid_dir, llc_lons_fname='XC.data', 
                                   llc_hfacc_fname='hFacC.data', llc=90, 
                                   llc_grid_filetype = '>f', 
                                   make_plots=False):

    """

    Runs test on the mds_io and llc_conversion routines


    Parameters
    ----------
    llc_grid_dir : string
        A string with the directory of the binary file to open

    llc_lons_fname : string
        A string with the name of the XC grid file  [XC.data]

    llc_hfacc_fname : string
        A string with the name of the hfacC grid file [hFacC.data]

    llc : int
        the size of the llc grid.  For ECCO v4, we use the llc90 domain 
        so `llc` would be `90`.  
        Default: 90

    llc_grid_filetype: string
        the file type, default is big endian (>) 32 bit float (f)
        alternatively, ('<d') would be little endian (<) 64 bit float (d)
        Deafult: '>f'
        
    make_plots : boolean
        A boolean specifiying whether or not to make plots
        Deafult: False

    Returns
    -------
    1 : all tests passed
    0 : at least one test failed
    
    """
    

    # SET TEST RESULT = 1 TO START
    TEST_RESULT = 1

    # %% ----------- TEST 1: 2D field XC FOM GRID FILE
    
    #%% 1a LOAD COMPACT
    tmpXC_c = load_llc_compact(llc_grid_dir, llc_lons_fname, llc=llc, nk=-1)
    tmpXC_f = load_llc_compact_to_faces(llc_grid_dir, llc_lons_fname, llc=llc, nk=-1)
    tmpXC_t = load_llc_compact_to_tiles(llc_grid_dir, llc_lons_fname, llc=llc, nk=-1)
    
    if make_plots:
        #plt.close('all')

        for f in range(1,6):
            plt.figure()
            plt.imshow(tmpXC_f[f]);plt.colorbar()    
    
        plot_tiles(tmpXC_t)
        plt.draw()
        raw_input("Press Enter to continue...")

    
    #%% 1b CONVERT COMPACT TO FACES, TILES
    tmpXC_cf = llc_compact_to_faces(tmpXC_c)
    tmpXC_ct = llc_compact_to_tiles(tmpXC_c)
    

    for f in range(1,6):
        tmp = np.unique(tmpXC_f[f] - tmpXC_cf[f])
        print ('unique diffs CF ', f, tmp)

        if len(tmp) != 1 or tmp[0] != 0:
            TEST_RESULT = 0
            print ('failed on 1b-1')
            return TEST_RESULT

    tmp = np.unique(tmpXC_ct - tmpXC_t)
    print ('unique diffs for CT ', tmp)

    if len(tmp) != 1 or tmp[0] != 0:
        TEST_RESULT = 0
        print ('failed on 1b-2')
        return TEST_RESULT
            
    
    
    #%% 1c CONVERT FACES TO TILES, COMPACT
    tmpXC_ft = llc_faces_to_tiles(tmpXC_f)
    tmpXC_fc = llc_faces_to_compact(tmpXC_f)
    
    # unique diff tests    
    tmp = np.unique(tmpXC_t - tmpXC_ft)
    print ('unique diffs for FT ', tmp)

    if len(tmp) != 1 or tmp[0] != 0:
        TEST_RESULT = 0
        print ('failed on 1c-1')
        return TEST_RESULT
    
    
    tmp = np.unique(tmpXC_fc - tmpXC_c)
    print ('unique diffs FC', tmp )

    if len(tmp) != 1 or tmp[0] != 0:
        TEST_RESULT = 0
        print ('failed on 1c-2')
        return TEST_RESULT
    
    
    #%% 1d CONVERT TILES to FACES, COMPACT 
    tmpXC_tf = llc_tiles_to_faces(tmpXC_t)
    tmpXC_tc = llc_tiles_to_compact(tmpXC_t)
    
    # unique diff tests    
    for f in range(1,6):
        tmp = np.unique(tmpXC_f[f] - tmpXC_tf[f])
        print ('unique diffs for TF ', f, tmp)
        
        if len(tmp) != 1 or tmp[0] != 0:
            TEST_RESULT = 0
            print ('failed on 1d-1')
            return TEST_RESULT
        
    
    tmp = np.unique(tmpXC_tc - tmpXC_c)
    print ('unique diffs TC', tmp)
    
    if len(tmp) != 1 or tmp[0] != 0:
        TEST_RESULT = 0
        print ('failed on 1d-2')
        return TEST_RESULT
    
    
    #%% 1e CONVERT COMPACT TO FACES TO TILES TO FACES TO COMPACT
    
    tmpXC_cftfc = llc_faces_to_compact(llc_tiles_to_faces(llc_faces_to_tiles(llc_compact_to_faces(tmpXC_c))))
    tmp = np.unique(tmpXC_cftfc - tmpXC_c)
    
    print ('unique diffs CFTFC', tmp)
    if len(tmp) != 1 or tmp[0] != 0:
        TEST_RESULT = 0
        print ('failed on 1e')
        return TEST_RESULT
    
    
    # %% ----------- TEST 2: 3D fields HFACC FOM GRID FILE
    
    #%% 2a LOAD COMPACT
    tmpHF_c = load_llc_compact(llc_grid_dir, llc_hfacc_fname, llc=llc,nk=-1)
    tmpHF_f = load_llc_compact_to_faces(llc_grid_dir, llc_hfacc_fname, llc=llc, nk=-1)
    tmpHF_t = load_llc_compact_to_tiles(llc_grid_dir, llc_hfacc_fname, llc=llc, nk=-1)
    
    tmpHF_c.shape
    
    if make_plots:
        #plt.close('all')
        plt.imshow(tmpHF_c[0,:]);plt.colorbar()   
        plot_tiles(tmpHF_t[:,0,:])
        plot_tiles(tmpHF_t[:,20,:])
        plt.draw()
        raw_input("Press Enter to continue...")
    
    #%% 2b CONVERT COMPACT TO FACES, TILES
    tmpHF_cf = llc_compact_to_faces(tmpHF_c)
    tmpHF_ct = llc_compact_to_tiles(tmpHF_c)
    
    # unique diff tests    
    for f in range(1,6):
        tmp = np.unique(tmpHF_f[f] - tmpHF_cf[f])
        print ('unique diffs CF ', f, tmp)
        if len(tmp) != 1 or tmp[0] != 0:
            TEST_RESULT = 0
            print ('failed on 2b-1')
            return TEST_RESULT
        

    tmp =  np.unique(tmpHF_ct - tmpHF_t)
    print ('unique diffs CT ', tmp)
    if len(tmp) != 1 or tmp[0] != 0:
        TEST_RESULT = 0
        print ('failed on 2b-2')
        return TEST_RESULT

    if make_plots:    
        for k in [0, 20]:
            for f in range(1,6):
                plt.figure()
                plt.imshow(tmpHF_cf[f][k,:], origin='lower');plt.colorbar()    
        plt.draw()
        raw_input("Press Enter to continue...")

    #%% 2c CONVERT FACES TO TILES, COMPACT
    tmpHF_ft = llc_faces_to_tiles(tmpHF_f)
    tmpHF_fc = llc_faces_to_compact(tmpHF_f)
    
    if make_plots:
        #plt.close('all')
        plot_tiles(tmpHF_ft[:,0,:])
        plot_tiles(tmpHF_ft[:,20,:])
        plt.draw()
        raw_input("Press Enter to continue...")

    # unique diff tests    
    tmp = np.unique(tmpHF_t - tmpHF_ft)
    print ('unique diffs FT ', tmp)
    if len(tmp) != 1 or tmp[0] != 0:
        TEST_RESULT = 0
        print ('failed on 2c-1')
        return TEST_RESULT

    tmp = np.unique(tmpHF_fc - tmpHF_c)
    print ('unique diffs FC', tmp)
    if len(tmp) != 1 or tmp[0] != 0:
        TEST_RESULT = 0
        print ('failed on 2c-2')
        return TEST_RESULT

    
    #%% 2d CONVERT TILES to FACES, COMPACT 
    tmpHF_tf = llc_tiles_to_faces(tmpHF_t)
    tmpHF_tc = llc_tiles_to_compact(tmpHF_t)
    
    if make_plots:
        #plt.close('all')
        for k in [0, 20]:
            for f in range(1,6):
                plt.figure()
                plt.imshow(tmpHF_tf[f][k,:], origin='lower');plt.colorbar()    
        plt.draw()
        raw_input("Press Enter to continue...")


    # unique diff tests    
    for f in range(1,6):
        tmp = np.unique(tmpHF_f[f] - tmpHF_tf[f])
        print ('unique diffs TF ', f, tmp)
        if len(tmp) != 1 or tmp[0] != 0:
            TEST_RESULT = 0
            print ('failed on 2d-1')
            return TEST_RESULT
            
    tmp = np.unique(tmpHF_tc - tmpHF_c)
    print ('unique diffs TC ', tmp)
    if len(tmp) != 1 or tmp[0] != 0:
            TEST_RESULT = 0
            print ('failed on 2d-1')
            return TEST_RESULT
    
    
    #%% 2e CONVERT COMPACT TO FACES TO TILES TO FACES TO COMPACT
    
    tmpHF_cftfc = llc_faces_to_compact(llc_tiles_to_faces(
            llc_faces_to_tiles(llc_compact_to_faces(tmpHF_c))))

    tmp = np.unique(tmpHF_cftfc - tmpHF_c)

    print ('unique diffs CFTFC ', tmp)
    if len(tmp) != 1 or tmp[0] != 0:
            TEST_RESULT = 0
            print ('failed on 2e')
            return TEST_RESULT
    
    print ('YOU MADE IT THIS FAR, TESTS PASSED!')
        
    return TEST_RESULT




##################################################
if __name__== "__main__":

    import sys
    import matplotlib
    sys.path.append('/Users/ifenty/ECCOv4-py/')
    import ecco_v4_py as ecco
    import matplotlib.pylab as plt

    model_grid_dir = '/Volumes/ECCO_BASE/ECCO_v4r3/grid_llc90/'

    TEST_RESULT = ecco.run_mds_io_and_llc_conversion_test(model_grid_dir, make_plots=True)

    print(TEST_RESULT)



