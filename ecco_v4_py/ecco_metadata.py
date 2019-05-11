#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""ECCO v4 Python: Dataset Utililites

This module includes utility routines that operate on the Dataset or DataArray Objects 

.. _ecco_v4_py Documentation :
   https://github.com/ECCO-GROUP/ECCOv4-py
"""

from __future__ import division, print_function
import numpy as np
import xarray as xr
import datetime
import shapefile
import time
import xmitgcm as xm
import dateutil
import glob
import os



#%%

def make_time_bounds_and_center_times_from_ecco_dataset(ecco_dataset, \
                                                        output_freq_code):
    """

    Given an ecco_dataset object (ecco_dataset) with time variables that 
    correspond with the 'end' of averaging time periods
    and an output frequency code (AVG_MON, AVG_DAY, AVG_WEEK, or AVG_YEAR), 
    create a time_bounds array of dimension 2xn, with 'n' being the number
    of time averaged records in ecco_dataset, each with two datetime64 
    variables, one for the averaging period start, one
    for the averaging period end.  
    
    The routine also creates an array of times corresponding to the 'middle'
    of each averaging period.
    
    Input
    ----------    
    ecco_dataset : an xarray dataset with 'time' variables representing 
    the times at the 'end' of an averaging periods


    Output: 
    ----------    
    time_bnds  :  a 2xn numpy array of type datetime64 that has the averaging
    period start and end times.
    
    center_times : a datetime64 object representing the middle of the 
    averaging period
    

    """ 
    if ecco_dataset.time.size == 1:
        if isinstance(ecco_dataset.time.values, np.ndarray):
            time_tmp = ecco_dataset.time.values[0]
        else:
            time_tmp = ecco_dataset.time.values
        time_bnds, center_times = \
            make_time_bounds_from_ds64(time_tmp,\
                                       output_freq_code)
        time_bnds=np.expand_dims(time_bnds,1)
        time_bnds = time_bnds.T
    else:
        time_start = []
        time_end  = []
        center_time = []
        for time_i in range(len(ecco_dataset.iter)):
             tb, ct = \
                 make_time_bounds_from_ds64(ecco_dataset.time.values[time_i], 
                                  output_freq_code)
             
             time_start.append(tb[0])
             time_end.append(tb[1])
             
             center_time.append(ct)
             
        # convert list to array
        center_times = np.array(center_time,dtype=np.datetime64)
        time_bnds    = np.array([time_start, time_end],dtype='datetime64')  
        time_bnds    = time_bnds.T
       
    # make time bounds dataset
    if 'time' not in ecco_dataset.dims.keys():
         ecco_dataset = ecco_dataset.expand_dims(dim='time')
         
    time_bnds_ds = xr.Dataset({'time_bnds': (['time','nv'], time_bnds)},
                             coords={'time':ecco_dataset.time})#,
                                     #'nv':range(2)})
    
    return time_bnds_ds, center_times

#%%
def make_time_bounds_from_ds64(rec_avg_end, output_freq_code):
    """

    Given a datetime64 object (rec_avg_end) representing the 'end' of an 
    averaging time period (usually derived from the mitgcm file's timestep)
    and an output frequency code
    (AVG_MON, AVG_DAY, AVG_WEEK, or AVG_YEAR), create a time_bounds array
    with two datetime64 variables, one for the averaging period start, one
    for the averaging period end.  Also find the middle time between the
    two..
    
    Input
    ----------    
    rec_avg_end : a datetime64 object representing the time at the 'end'
    of an averaging period


    Output: 
    ----------    
    time_bnds  :  a 2x1 numpy array of type datetime64 that has the averaging
    period start and end time.
    
    rec_avg_middle : a datetime64 object representing the middle of the 
    averaging period
    

    """ 
    
    if  output_freq_code in ('AVG_MON','AVG_DAY','AVG_WEEK','AVG_YEAR'):
        rec_year, rec_mon, rec_day, \
        rec_hour, rec_min, rec_sec = \
            extract_yyyy_mm_dd_hh_mm_ss_from_datetime64(rec_avg_end)
        
        
        rec_avg_end_as_dt = datetime.datetime(rec_year, rec_mon, 
                                                  rec_day, rec_hour,
                                                  rec_min, rec_sec)
        
        if output_freq_code     == 'AVG_MON':
            rec_avg_start =  rec_avg_end_as_dt - \
                dateutil.relativedelta.relativedelta(months=1)    
        elif output_freq_code   == 'AVG_DAY':
            rec_avg_start =  rec_avg_end_as_dt - \
                dateutil.relativedelta.relativedelta(days=1)  
        elif output_freq_code   == 'AVG_WEEK':
            rec_avg_start =  rec_avg_end_as_dt - \
                dateutil.relativedelta.relativedelta(weeks=1)  
        elif output_freq_code   == 'AVG_YEAR':
            rec_avg_start =  rec_avg_end_as_dt - \
                dateutil.relativedelta.relativedelta(years=1)    

        rec_avg_start =  np.datetime64(rec_avg_start)
        
        rec_avg_delta = rec_avg_end - rec_avg_start
        rec_avg_middle = rec_avg_start + rec_avg_delta/2
        #print rec_avg_end, rec_avg_start, rec_avg_middle
        
        rec_time_bnds = np.array([rec_avg_start, rec_avg_end])
        
        return rec_time_bnds, rec_avg_middle
    
    else:
        print ('output_freq_code must be: AVG_MON, AVG_DAY, AVG_WEEK, OR AVG_YEAR')
        print ('you provided ' + str(output_freq_code))
        return [],[]   
 
#%%
def extract_yyyy_mm_dd_hh_mm_ss_from_datetime64(dt64):
    """

    Extract separate fields for year, monday, day, hour, min, sec from
    a datetime64 object
    
    Input
    ----------    
    dt64      : a datetime64 object 


    Output: 
    ----------    
    year, mon, day, hh, mm, ss : self-explanatory
    """
    
    s = str(dt64)
    year = int(s[0:4])
    mon = int(s[5:7])
    day = int(s[8:10])
    hh = int(s[11:13])
    mm = int(s[14:16])
    ss = int(s[17:18])
    
    #print year, mon, day, hh, mm, ss
    return year,mon,day,hh,mm,ss 

#%%
def createShapefileFromXY(outDir, outName, X,Y,subset):
    """

    This routine takes an X,Y array of grid points (e.g., XC, YC or XG, YG)) 
    and creates one of two types of shapefiles.
    1: polylines shapefile that trace the cell boundaries  (subset = 'boundary_points')
    2: point shapefile with a point in the cell centers    (subset = 'center points') 

    # This routine was originally written by Michael Wood.
    # This version was modified by Fenty, 2/28/2019 to handle a newer version of pyshp (2.1.0)

    Parameters
    ----------
    outDir    : directory into which shapefile and accessory files will be written
    outName   : base of the filename (4 files will be created, 
    X,Y       : array of points in X and Y (must be lat/lon)
    subset    : either 'boundary_points' to create polyline shapefile
                 or 'points' to create point shapefile

    """

    if subset=='center_points':
        
        fname = outDir +'/' + outName + '_Grid_Center_Points/' + outName + '_Grid_Center_Points'
        
        w=shapefile.Writer(fname + '.shp')
        w.shapeType = shapefile.POINT
        w.field('id')

        counter=0
        for i in range(np.shape(X)[0]):
            for j in range(np.shape(X)[1]):
                w.point(X[i,j],Y[i,j])
                w.record(counter)
                counter+=1
        w.close()

        f=open(fname + '.prj','w')
        f.write('GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]]')
        f.close()

    elif subset=='boundary_points':
        
        fname = outDir + '/' + outName + '_Grid_Boundary_Points/' + outName + '_Grid_Boundary_Points'
        print(fname)
        w=shapefile.Writer(fname + '.shp')
        w.shapeType = shapefile.POLYLINE
        w.field('id')

        counter=0
        #create the vertical lines
        for i in range(np.shape(X)[0]):
            lines=[]
            for j in range(np.shape(X)[1]):
                lines.append( [ X[i,j], Y[i,j] ])

            w.line([lines])
            w.record(counter)
            counter+=1

        # create the horizontal lines
        XT = X.T
        YT = Y.T
        for i in range(np.shape(XT)[0]):
            lines=[]
            for j in range(np.shape(XT)[1]):
                lines.append( [ XT[i,j], YT[i,j] ])

            w.line([lines])
            w.record(counter)
            counter+=1

        w.close()

        f = open(fname + '.prj', 'w')
        f.write(
            'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]]')
        f.close()
    else:
        print("subset must be either center_points or boundary_points")


#%%
def minimal_metadata(ds):
    """

    This routine removes some of the redundant metadata that is included with the ECCO v4 netcdf tiles from the Dataset object `ds`.  Specifically, metadata with the tags `A` through `Z` (those metadata records) that describe the origin of the ECCO v4 output.    

    Parameters
    ----------
    ds : Dataset
        An `xarray` Dataset object that was created by loading an 
        ECCO v4 tile netcdf file

    
        
    """

    print('Removing Dataset Attributes A-Z\n')
    # generate a list of upper case letters in teh alphabet
    myDict= map(chr, range(65, 91))

    for key, value in ds.attrs.items():
        if key in myDict: 
            del ds.attrs[key]
         


#%%
def months2days(nmon=288, baseyear=1992, basemon=1):
    
    """ 

    This rouine converts the mid-month time to days from January 1st of a particular year.
    
    Input:  nmon:           number of months (dtype=integer) 
            baseyear:       year of time of origin (dtype=integer)
            basemon:        month of time of origin (dtype=integer)
    Output: time_days:      the middle time of each month in days
                                from 01/01/baseyear (numpy array [nmon], dtype=double)
            time_days_bnds: time bounds (numpy array [nmon, 2], dtype=double)
            ansi_date:      ANSI date 

    """
    
    time_days_bnds = np.zeros([nmon,2])
    time_1stdayofmon = np.zeros([nmon+1])
    
    basetime = datetime.datetime(baseyear, basemon, 1, 0, 0, 0)

    for mon in range(nmon+1):
        #monfrombasemon is how many months fron basemon
        monfrombasemon=basemon+mon-1
        yrtmp = monfrombasemon//12+baseyear
        montmp = monfrombasemon%12+1
        tmpdate = datetime.datetime(yrtmp,montmp,1,0,0,0)-basetime
        time_1stdayofmon[mon] = tmpdate.days
    #time bounds are the 1st day of each month.
    time_days_bnds[:,0]= time_1stdayofmon[0:nmon]
    time_days_bnds[:,1]= time_1stdayofmon[1:nmon+1]
    #center time of each month is the mean of the time bounds.    
    time_days = np.mean(time_days_bnds,axis=1)

    ansi_datetmp = np.array([basetime + datetime.timedelta(days=time_days[i]) for i in xrange(nmon)])
    ansi_date = [str.replace(ansi_datetmp[i].isoformat(),'T',' ') for i in range(nmon)]

    return time_days, time_days_bnds, ansi_date

#%%


def load_ecco_vars_from_mds(mds_var_dir, 
                            mds_files, 
                            mds_grid_dir,
                            vars_to_load=[], 
                            tiles_to_load = range(13),
                            iters_to_load = [],
                            output_freq_code='', 
                            meta_variable_specific=dict(),
                            meta_common=dict(),
                            mds_datatype = '>f4'):
                                 
    
    #%%
    #ECCO v4 r3 starts 1992/1/1 12:00:00
    ecco_v4_start_year = 1992
    ecco_v4_start_mon  = 1
    ecco_v4_start_day  = 1
    ecco_v4_start_hour = 12
    ecco_v4_start_min  = 0
    ecco_v4_start_sec  = 0
    
    # ECCO v4 r3 has 1 hour (3600 s) time steps    
    delta_t = 3600
    
    # define reference date for xmitgcm
    ref_date = str(ecco_v4_start_year) + '-' + str(ecco_v4_start_mon)  + '-' + \
        str(ecco_v4_start_day)  + ' ' +  str(ecco_v4_start_hour) + ':' +  \
        str(ecco_v4_start_min)  + ':' + str(ecco_v4_start_sec)

    # if iters_to_load is empty list, load all files 
    if len(iters_to_load) > 0:
        ecco_dataset = xm.open_mdsdataset(data_dir = mds_var_dir, 
                                          grid_dir = mds_grid_dir,
                                          read_grid = True,
                                          prefix = mds_files,
                                          geometry = 'llc', 
                                          iters = iters_to_load,
                                          ref_date = ref_date, 
                                          delta_t  = delta_t,
                                          default_dtype = np.dtype(mds_datatype),
                                          grid_vars_to_coords=True)
    
    else:
        ecco_dataset = xm.open_mdsdataset(data_dir = mds_var_dir, 
                                          grid_dir = mds_grid_dir,
                                          read_grid = True,
                                          prefix = mds_files, 
                                          geometry = 'llc', 
                                          ref_date = ref_date, 
                                          delta_t = delta_t,
                                          default_dtype = np.dtype(mds_datatype),
                                          grid_vars_to_coords=True)
   
    
    # replace the xmitgcm coordinate name of 'FACE' with 'TILE'
    if 'face' in ecco_dataset.coords.keys():
        ecco_dataset = ecco_dataset.rename({'face': 'tile'})
        ecco_dataset.tile.attrs['standard_name'] = 'tile_index'
       
    # if vars_to_load is an empty list, keep all variables.  otherwise,
    # only keep those variables in the vars_to_load list.
    
    if len(vars_to_load) > 0:
        print ('loading subset of variables: ', vars_to_load)
        # remove variables that are not on the vars_to_load_list
        for ecco_var in ecco_dataset.keys():
            print (ecco_var)
            
            if ecco_var not in vars_to_load:
                print ( ecco_var + ' not in vars to load')
                ecco_dataset = ecco_dataset.drop(ecco_var)
            else:
                print ( ecco_var + ' in vars to load')
            
    # only keep those tiles in the 'tiles_to_load' list.
    if isinstance(tiles_to_load, (tuple,list)) and len(tiles_to_load) > 0:
        tiles_to_load = list(tiles_to_load)
        ecco_dataset = ecco_dataset.sel(tile = tiles_to_load)
    
    #ecco_dataset = ecco_dataset.isel(time=0)
    if 'AVG' in output_freq_code and \
        'time_bnds' not in ecco_dataset.keys():

        time_bnds_ds, center_times = \
            make_time_bounds_and_center_times_from_ecco_dataset(ecco_dataset,\
                                                                output_freq_code)
        
        if isinstance(ecco_dataset.time.values, np.datetime64):
            ecco_dataset.time.values = center_times
        elif isinstance(center_times, np.datetime64):
            center_times = np.array(center_times)
            ecco_dataset.time.values[:] = center_times
        
        if 'ecco-v4-time-average-center-no-units' in meta_common:
            ecco_dataset.time.attrs = \
                meta_common['ecco-v4-time-average-center-no-units']
          
        ecco_dataset = xr.merge((ecco_dataset, time_bnds_ds))
        
        if 'time_bnds-no-units' in meta_common:
            ecco_dataset.time_bnds.attrs=meta_common['time_bnds-no-units']
      
        ecco_dataset = ecco_dataset.set_coords('time_bnds')
        
    elif  'SNAPSHOT' in output_freq_code:
         if 'ecco-v4-time-snapshot-no-units' in meta_common:
            ecco_dataset.time.attrs = \
                meta_common['ecco-v4-time-snapshot-no-units']
       

    #%% DROP SOME EXTRA FIELDS THAT DO NOT NEED TO BE IN THE DATASET
    if 'maskCtrlS' in ecco_dataset.coords.keys():
        ecco_dataset=ecco_dataset.drop('maskCtrlS')
    if 'maskCtrlW' in ecco_dataset.coords.keys():
        ecco_dataset=ecco_dataset.drop('maskCtrlW')
    if 'maskCtrlC' in ecco_dataset.coords.keys():
        ecco_dataset=ecco_dataset.drop('maskCtrlC')
        
    # UPDATE THE VARIABLE SPECIFIC METADATA USING THE 'META_VARSPECIFIC' DICT.
    # if it exists
    for ecco_var in ecco_dataset.variables.keys():
        if ecco_var in meta_variable_specific.keys():
            ecco_dataset[ecco_var].attrs = meta_variable_specific[ecco_var]


    #%% UPDATE THE GLOBAL METADATA USING THE 'META_COMMON' DICT, if it exists
    ecco_dataset.attrs = dict()

    if 'ecco-v4-global' in meta_common:
        ecco_dataset.attrs.update(meta_common['ecco-v4-global'])

    if 'k' in ecco_dataset.dims.keys() and \
        'ecco-v4-global-3D' in meta_common:
        ecco_dataset.attrs.update(meta_common['ecco-v4-global-3D'])
    
   
    ecco_dataset.attrs['date_created'] = time.ctime()
    
    # give it a hug?
    ecco_dataset = ecco_dataset.squeeze()

    return ecco_dataset


#%%
