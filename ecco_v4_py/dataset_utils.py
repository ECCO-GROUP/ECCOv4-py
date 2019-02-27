#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""ECCO v4 Python: Dataset Utililites

This module includes utility routines that operate on the Dataset or DataArray Objects 

.. _ecco_v4_py Documentation :
   https://github.com/ECCO-GROUP/ECCOv4-py
"""

from __future__ import division
import numpy as np
import xarray as xr
import datetime

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def minimal_metadata(ds):
    """

    This routine removes some of the redundant metadata that is included with the ECCO v4 netcdf tiles from the Dataset object `ds`.  Specifically, metadata with the tags `A` through `Z` (those metadata records) that describe the origin of the ECCO v4 output.    

    Parameters
    ----------
    ds : Dataset
        An `xarray` Dataset object that was created by loading an 
        ECCO v4 tile netcdf file

    
        
    """

    print 'Removing Dataset Attributes A-Z\n'
    # generate a list of upper case letters in teh alphabet
    myDict= map(chr, range(65, 91))

    for key, value in ds.attrs.iteritems():
        if key in myDict: 
            del ds.attrs[key]
         
def months2days(nmon=288, baseyear=1992):
    
    """
    This rouine converts time in days from January 1st of a particular year.
    
    Input:  nmon:           number of months (dtype=integer) 
            baseyear:       year of time of origin (dtype=integer)
    Output: time_days:      the middle time of each month in days
                                from 01/01/baseyear (numpy array [nmon], dtype=double)
            time_days_bnds: time bounds (numpy array [nmon, 2], dtype=double)
            ansi_date:      ANSI date 

    """
    
    time_days_bnds = np.zeros([nmon,2])
    time_1stdayofmon = np.zeros([nmon+1])
    
    basetime = datetime.datetime(baseyear, 1, 1, 0, 0, 0)

    for mon in range(nmon+1):
        yrtmp = mon//12+baseyear
        montmp = mon%12+1
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
