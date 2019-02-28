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
import shapefile

######################################################################
#set up functions to read/write out the grids as shapefiles

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
        fname = outDir+'/'+outName+'_Grid_Center_Points/'+outName+'_Grid_Center_Points'
        w=shapefile.Writer(fname + '.shp')
        w.shapeType = shapefile.POINT
        w.field('id')
        counter=0
        for i in range(np.shape(X)[0]):
            for j in range(np.shape(X)[1]):
                w.point(X[i,j],Y[i,j])
                w.record(counter)
                counter+=1
        w.save(fname + '.shp')
        f=open(fname + '.prj','w')
        f.write('GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]]')
        f.close()

    elif subset=='boundary_points':
        filename = outDir + '/' + outName + '_Grid_Boundary_Points/' + outName + '_Grid_Boundary_Points'
        w=shapefile.Writer(fname + '.shp')
        w.shapeType = shapefile.POLYLINE
        counter=0
        w.field('id')
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
        print "subset must be either center_points or boundary_points"


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
         



def months2days(nmon=288, baseyear=1992, basemon=1):
    
    """ 
    This rouine converts time in days from January 1st of a particular year.
    
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
