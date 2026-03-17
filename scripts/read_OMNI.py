#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 9 Dec 2025

@author: Dean Thomas
"""
import swmpy as swm
from file_info_all_SuperMAG import file_info

if __name__ == "__main__":

    # whether to parse OMNI data
    PARSE = False

    # whether to generate OMNI stats
    STATS = True
    
    # whether to generate raw OMNI files
    RAW = True
    
    # whether to generate OMNI plots
    PLOTS = True

    ############## read OMNI data

    # Process the OMNI solar wind files. OMNI data downloaded from:
    # https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/modified/

    # Specify years that the input OMNI files cover
    # The years are used to generate the pickle file names
    yrs = file_info["years"]
                
    # number of minutes over which statistics gathered (e.g., statistics 
    # for number = 30 are calculated over consecutive 30 minute windows)    
    nums = file_info['numbers']
     
    # distance from earth (toward sun) at which solar wind data is valid.
    # Solar wind data will be ballistically propagated from bow shock nose
    # to this point on the GSE x axis.
    dists = file_info['distances']

    ############## parse OMNI files

    if PARSE: 
        for yr in yrs: 
            swm.omni_read( file_info, yr )

    ############## generate OMNI stats

    if STATS:
        for yr in yrs:
            for num in nums:
                for dist in dists:
                    print( yr, num, dist )
                    swm.omni_stats(file_info, yr, num, dist)
                    
    if RAW:
        for yr in yrs:
            swm.omni_raw( file_info, yr )
    
   ############## generate OMNI Plots
   
    if PLOTS:
        for yr in yrs:
            for num in nums:
                for dist in dists:
                    swm.omni_plots(file_info, yr, num, dist)
       
