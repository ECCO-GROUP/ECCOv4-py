#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 16:11:15 2017

@author: ifenty
"""
from __future__ import division
import numpy as np
import xarray as xr

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def minimal_metadata(ds):
    #%%
    print 'Removing Dataset Attributes A-Z\n'
    # generate a list of upper case letters in teh alphabet
    myDict= map(chr, range(65, 91))

    for key, value in ds.attrs.iteritems():
        if key in myDict: 
            del ds.attrs[key]
            

