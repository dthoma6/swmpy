#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 16:05:13 2026

@author: Dean Thomas
"""
import swmpy as swm

from file_info import file_info

if __name__ == "__main__":
        
    # whether to parse Kp data
    PARSE = False

    # whether to generate Kp stats
    STATS = False
    
    # whether to generate raw Kp files
    RAW = True
    
    # Process the Kp solar wind files. Kp data downloaded from:
    # https://kp.gfz.de/en/data
    #
    # Data was downloaded, one file for each year.

    # Specify years that the input Kp files cover
    # The years are used to generate the pickle file names
    yrs = file_info["years"]

    # number of minutes over which statistics gathered (e.g., statistics 
    # for number = 30 are calculated over consecutive 30 minute windows)    
    nums = file_info['numbers']

    ############## parse Kp files

    if PARSE: 
        for yr in yrs:
            swm.kp_read( file_info, yr )

    ############## generate Kp stats

    if STATS:
        for yr in yrs:
            for num in nums:
                swm.kp_stats(file_info, yr, num)

    if RAW:
        for yr in yrs:
            swm.kp_raw(file_info, yr)
