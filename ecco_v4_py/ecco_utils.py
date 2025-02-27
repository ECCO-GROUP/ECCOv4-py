#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECCO v4 Python: Utililites

This module includes utility routines that operate on the Dataset or DataArray Objects

.. _ecco_v4_py Documentation :
   https://github.com/ECCO-GROUP/ECCOv4-py
"""

from __future__ import division, print_function
import numpy as np
import pandas as pd
import xarray as xr
import datetime
import dateutil
import xgcm
from pprint import pprint
from collections import OrderedDict

def find_metadata_in_json_dictionary(var, key, metadata, print_output=False):
    for m in metadata:
        if key in m.keys():
            if m[key] == var:
                if print_output:
                    print(m)
                return m

    return []



def add_global_metadata(metadata, G, dataset_dim, less_output=True):

    if not less_output:
        print('adding global metadata')
        pprint(metadata)

    # loop through pairs
    for mc in metadata:
        # get name and type
        mname = mc['name']
        mtype = mc['type']

        # by default add the key/pair
        add_field = True

        # unless it has as specific 'grid_dimension' associated
        # with it. If so, then only add it if this dataset indicates
        # it is necessary.  for example, don't provide geospatial
        # depth information for 2D datasets
        if 'grid_dimension' in mc.keys():
            gd = mc['grid_dimension']

            if dataset_dim not in gd:
                add_field = False

        # if we do add the field, we have to convert to the
        # appropriate data type
        if add_field == True:
            if isinstance(mc['value'], str) and 'TBD_' in mc['value']:
                G.attrs[mname] = mc['value']
            else:
                if mtype == 's':
                    G.attrs[mname] = mc['value']
                elif mtype == 'f':
                    G.attrs[mname] = float(mc['value'])
                elif mtype == 'i':
                    G.attrs[mname] = np.int32(mc['value'])
                else:
                    print('INVALID MTYPE ! ', mtype)
        else:
            if not less_output:
                print('\t> not adding ', mc)

    return G


def add_coordinate_metadata(metadata_dict, G, less_output=True):
    # G : dataset
    # metadata_dict: dictionary of metadata records with name as a key
    keys_to_exclude = ['grid_dimension','name']

    for coord in G.coords:

        if not less_output:
            print('\n### ', coord)
        # look for coordinate in metadat dictionary
        mv = find_metadata_in_json_dictionary(coord, 'name', metadata_dict)

        if len(mv) > 0:
            # if metadata for this coordinate is present
            # loop through all of the keys and if it is not
            # on the excluded list, add it
            for m_key in sorted(mv.keys()):
                if m_key not in keys_to_exclude:
                    G[coord].attrs[m_key] = mv[m_key]

                    if not less_output:
                        print('\t',m_key, ':', mv[m_key])
        else:
            if not less_output:
                print('...... no metadata found in dictionary')

    return G


def add_variable_metadata(variable_metadata_dict, G, \
                          grouping_gcmd_keywords=[], less_output=True):

    # ADD VARIABLE METADATA  & SAVE GCMD KEYWORDS
    keys_to_exclude = ['grid_dimension','name', 'GCMD_keywords', "variable_rename",\
                       'comments_1', 'comments_2', 'internal_note','internal note',\
                       'grid_location']

    for var in G.data_vars:
        if not less_output:
            print('\n### ', var)

        mv = find_metadata_in_json_dictionary(var, 'name', variable_metadata_dict)

        if len(mv) == 0:
            if not less_output:
                print('...... no metadata found in dictionary')

        else:
            # loop through each key, add if not on exclude list
            for m_key in sorted(mv.keys()):
                if m_key not in keys_to_exclude:
                    G[var].attrs[m_key] = mv[m_key]
                    if not less_output:
                        print('\t',m_key, ':', mv[m_key])

            # merge the two comment fields (both *MUST* be present in the json file
            # but they can be empty strings "")

            # if either one is not-empty, add a comment field
            if len(mv['comments_1']) > 0 or len(mv['comments_2']) > 0:

                # if both are not-empty, merge them, make sure '.' between
                if len(mv['comments_1']) > 0 and len(mv['comments_2']) > 0:
                    if mv['comments_1'][-1] == '.':
                        G[var].attrs['comment'] =  mv['comments_1'] + ' ' + mv['comments_2']
                    else:
                        G[var].attrs['comment'] =  mv['comments_1'] + '. ' + mv['comments_2']

                # if only comments_1 is not empty
                elif len(mv['comments_1']) > 0:
                    G[var].attrs['comment'] = mv['comments_1']

                # if only comments_2 is not empty
                elif len(mv['comments_2']) > 0:
                   G[var].attrs['comment'] = mv['comments_2']

                if not less_output:
                    print('\t', 'comment', ':', G[var].attrs['comment'])
            else:
                if not less_output:
                    print('\t', 'comment fields are empty')

            # append GCMD keywords, if present
            if 'GCMD_keywords' in mv.keys():
               # Get the GCMD keywords, these will be added into the global
               # attributes
               gcmd_keywords = mv['GCMD_keywords'].split(',')

               if not less_output:
                   print('\t','GCMD keywords : ', gcmd_keywords)

               for gcmd_keyword in gcmd_keywords:
                   grouping_gcmd_keywords.append(gcmd_keyword.strip())

    return G, grouping_gcmd_keywords


def sort_attrs(attrs):
    """

    Alphabetically sort all keys in a dictionary

    Parameters
    ----------
    attrs : dict
        a dictionary of key/value pairs

    Returns
    -------
    attrs : dict
        a dictionary of key/value pairs sorted alphabetically by key

    """

    od = OrderedDict()

    keys = sorted(list(attrs.keys()),key=str.casefold)

    for k in keys:
        od[k] = attrs[k]

    return od


def sort_all_attrs(ecco_dataset, print_output=False):
    """

    Alphabetically sort all attributes in a ecco_dataset object (ecco_dataset)
    including the attributes of coordinates, data variables,
    and the global attributes.

    Parameters
    ----------
    ecco_dataset : xarray Dataset
        an xarray dataset

    Returns
    -------
    ecco_dataset : xarray Dataset
        an xarray dataset with all attributes sorted alphabetically

    """

    for coord in list(ecco_dataset.coords):
        if print_output:
            print(coord)
        new_attrs = sort_attrs(ecco_dataset[coord].attrs)
        ecco_dataset[coord].attrs = new_attrs

    for dv in list(ecco_dataset.data_vars):
        if print_output:
            print(dv)
        new_attrs = sort_attrs(ecco_dataset[dv].attrs)
        ecco_dataset[dv].attrs = new_attrs

    new_attrs = sort_attrs(ecco_dataset.attrs)
    ecco_dataset.attrs = new_attrs

    return ecco_dataset


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

    Parameters
    ----------
    ecco_dataset : xarray Dataset
        an xarray dataset with 'time' variables representing the times at
        the 'end' of an averaging periods


    Returns
    -------
    time_bnds : np.array(dtype=np.datetime64)
        a datetime64 with the start and end time(s) of the averaging periods

    center_times :np.array(dtype=np.datetime64)
        a numpy array containing the 'center' time of the averaging period(s)


    """

    if ecco_dataset.time.size == 1:

        if isinstance(ecco_dataset.time.values, np.ndarray):
            time_tmp = ecco_dataset.time.values[0]
        else:
            time_tmp = ecco_dataset.time.values

        time_bnds, center_times = \
            make_time_bounds_from_ds64(time_tmp,\
                                       output_freq_code)

        time_bnds=np.expand_dims(time_bnds,0)

    else:
        time_start = []
        time_end  = []
        center_time = []
        for time_i in range(len(ecco_dataset.timestep)):
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
    if 'time' not in ecco_dataset.dims:
         ecco_dataset = ecco_dataset.expand_dims(dim='time')

    #print ('-- tb shape ', time_bnds.shape)
    #print ('-- tb type  ', type(time_bnds))

    time_bnds_ds = xr.Dataset({'time_bnds': (['time','nv'], time_bnds)},
                               coords={'time':ecco_dataset.time}) #,

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

    Parameters
    ----------
    rec_avg_end : numpy.datetime64
        the time at the end of an averaging period

    output_freq_code : str
        code indicating the time period of the averaging period
        - AVG_DAY, AVG_MON, AVG_WEEK, or AVG_YEAR


    Returns
    -------
    time_bnds : numpy.array(dtype=numpy.datetime64)
        a datetime64 array with the start and end time of the averaging periods

    center_times : numpy.datetime64
        the 'center' of the averaging period

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

    Extract separate fields for year, month, day, hour, min, sec from
    a datetime64 object, or an array-like object of datetime64 objects

    Parameters
    ----------
    dt64 : xarray DataArray, np.ndarray, list of, or single numpy.datetime64
        datetime64 object

    Returns
    -------
    year, mon, day, hh, mm, ss : int

    """
    # use xarray to do this robustly
    if isinstance(dt64,xr.core.dataarray.DataArray):
        year = dt64.dt.year.astype(int)
        mon = dt64.dt.month.astype(int)
        day = dt64.dt.day.astype(int)
        hh = dt64.dt.hour.astype(int)
        mm = dt64.dt.minute.astype(int)
        ss = dt64.dt.second.astype(int)
        return year, mon, day, hh, mm, ss


    # otherwise transform the problem to use xarray
    elif isinstance(dt64,list):
        xdates = extract_yyyy_mm_dd_hh_mm_ss_from_datetime64(xr.DataArray(np.array(dt64)))
        return tuple([list(x.values) for x in xdates])
    elif isinstance(dt64,np.ndarray):
        xdates = extract_yyyy_mm_dd_hh_mm_ss_from_datetime64(xr.DataArray(dt64))
        return tuple([x.values for x in xdates])
    elif isinstance(dt64,np.datetime64):
        xdates = extract_yyyy_mm_dd_hh_mm_ss_from_datetime64(xr.DataArray(pd.to_datetime(dt64)))
        return tuple([int(x.values) for x in xdates])

#%%
def minimal_metadata(ds):
    """

    This routine removes some of the redundant metadata that is included with the ECCO v4 netcdf tiles from the Dataset object `ds`.  Specifically, metadata with the tags `A` through `Z` (those metadata records) that describe the origin of the ECCO v4 output.

    Parameters
    ----------
    ds : xarray Dataset
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

    This routine converts the mid-month time to days from January 1st of a particular year.

    Parameters
    ----------
    nmon : dtype=integer
        number of months

    baseyear : dtype=integer
        year of time of origin

    basemon : dtype=integer
        month of time of origin

    Returns
    -------
        time_days : ndarray
            the middle time of each month in days from Jan 1 baseyear (numpy array [nmon], dtype=double)

        time_days_bnds : ndarray
            time bounds (numpy array [nmon, 2], dtype=double)

        ansi_date : ndarray
            array of ANSI date strings

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

    ansi_datetmp = np.array([basetime + datetime.timedelta(days=time_days[i]) for i in range(nmon)])
    ansi_date = [str.replace(ansi_datetmp[i].isoformat(),'T',' ') for i in range(nmon)]

    return time_days, time_days_bnds, ansi_date

#%%

def get_llc_grid(ds,domain='global'):
    """
    Define xgcm Grid object for the LLC grid
    See example usage in the xgcm documentation:
    https://xgcm.readthedocs.io/en/latest/example_eccov4.html#Spatially-Integrated-Heat-Content-Anomaly

    Parameters
    ----------
    ds : xarray Dataset
        formed from LLC90 grid, must have the basic coordinates:
        i,j,i_g,j_g,k,k_l,k_u,k_p1

    Returns
    -------
    grid : xgcm Grid object
        defines horizontal connections between LLC tiles

    """

    if 'domain' in ds.attrs:
        domain = ds.attrs['domain']

    if domain == 'global':
        # Establish grid topology
        tile_connections = {'tile':  {
                0: {'X': ((12, 'Y', False), (3, 'X', False)),
                    'Y': (None, (1, 'Y', False))},
                1: {'X': ((11, 'Y', False), (4, 'X', False)),
                    'Y': ((0, 'Y', False), (2, 'Y', False))},
                2: {'X': ((10, 'Y', False), (5, 'X', False)),
                    'Y': ((1, 'Y', False), (6, 'X', False))},
                3: {'X': ((0, 'X', False), (9, 'Y', False)),
                    'Y': (None, (4, 'Y', False))},
                4: {'X': ((1, 'X', False), (8, 'Y', False)),
                    'Y': ((3, 'Y', False), (5, 'Y', False))},
                5: {'X': ((2, 'X', False), (7, 'Y', False)),
                    'Y': ((4, 'Y', False), (6, 'Y', False))},
                6: {'X': ((2, 'Y', False), (7, 'X', False)),
                    'Y': ((5, 'Y', False), (10, 'X', False))},
                7: {'X': ((6, 'X', False), (8, 'X', False)),
                    'Y': ((5, 'X', False), (10, 'Y', False))},
                8: {'X': ((7, 'X', False), (9, 'X', False)),
                    'Y': ((4, 'X', False), (11, 'Y', False))},
                9: {'X': ((8, 'X', False), None),
                    'Y': ((3, 'X', False), (12, 'Y', False))},
                10: {'X': ((6, 'Y', False), (11, 'X', False)),
                     'Y': ((7, 'Y', False), (2, 'X', False))},
                11: {'X': ((10, 'X', False), (12, 'X', False)),
                     'Y': ((8, 'Y', False), (1, 'X', False))},
                12: {'X': ((11, 'X', False), None),
                     'Y': ((9, 'Y', False), (0, 'X', False))}
        }}

        grid = xgcm.Grid(ds,
                periodic=False,
                face_connections=tile_connections
        )
    elif domain == 'aste':
        tile_connections = {'tile':{
                    0:{'X':((5,'Y',False),None),
                       'Y':(None,(1,'Y',False))},
                    1:{'X':((4,'Y',False),None),
                       'Y':((0,'Y',False),(2,'X',False))},
                    2:{'X':((1,'Y',False),(3,'X',False)),
                       'Y':(None,(4,'X',False))},
                    3:{'X':((2,'X',False),None),
                       'Y':(None,None)},
                    4:{'X':((2,'Y',False),(5,'X',False)),
                       'Y':(None,(1,'X',False))},
                    5:{'X':((4,'X',False),None),
                       'Y':(None,(0,'X',False))}
                   }}
        grid = xgcm.Grid(ds,periodic=False,face_connections=tile_connections)
    else:
        raise TypeError(f'Domain {domain} not recognized')


    return grid
