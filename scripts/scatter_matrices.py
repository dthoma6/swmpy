#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 10:13:57 2026

@author: Dean Thomas
"""
#
# This routine creates a scatter matrices of OMNI solar wind data
# We'll use it to throw out parameters that are highly correlated
#

import swmpy as swm     

if __name__ == "__main__":
    
    # Directory for OMNI data
    OMNIDIRECTORY = './OMNI/'
    
    # Specify years that the input files cover
    # The years are used to generate the pickle file names
    yrs = [2025, ]
    # yrs = [2023, 2024, 2025]

    # number of minutes over which statistics gathered (e.g., statistics 
    # for number = 30 are calculated over consecutive 30 minute windows)    
    nums = [30, ]
    # nums = [30, 60]
    
    # distance from earth (toward sun) at which solar wind data is valid.
    # Solar wind data will be ballistically propagated from bow shock nose
    # to this point on the GSE x axis.
    dists = [2, ]
    # dists = [10, 8, 6, 4, 2, 0]
    
    # level defines which variables are removed.  Level 0: include all 
    # variables.  Level 1: Remove derived parameters as defined in OMNI
    # documentatio, see below.  Level 2: Remove remaining parameters that 
    # have a correlation coefficient over 0.9
    level=3
    
    for yr in yrs:
        for num in nums:
            for dist in dists:
                # Generate scatter plots
                swm.scatter_matrix(OMNIDIRECTORY, 
                                  yr, num, dist,level=level)