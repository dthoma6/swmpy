#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 13:55:28 2026

@author: Dean Thomas
"""

import swmpy as swm

if __name__ == "__main__":
        
    # Directory for SuperMAG data
    supermagdirectory = './SuperMAG/'
    
    # Directory for OMNI data
    omnidirectory = './OMNI/'
    
    # Specify years that the input files cover
    # The years are used to generate the pickle file names
    years = [2023, ]
    # years = [2023, 2024, 2025]

    # number of minutes over which statistics gathered (e.g., statistics 
    # for number = 30 are calculated over consecutive 30 minute windows)    
    numbers = [30, ]
    # numbers = [30, 60]
        
    # distance from earth (toward sun) at which solar wind data is valid.
    # Solar wind data will be ballistically propagated from bow shock nose
    # to this point on the GSE x axis.
    # distances = [2, ]
    distances = [10, 8, 6, 4, 2, 0]
    
    for year in years:
        for number in numbers:
            # Generate distribution plots
            swm.supermag_distribution(supermagdirectory, year, number)
    
    for year in years:
        for number in numbers:
            for distance in distances:
                # Generate distribution plots
                swm.omni_distribution(omnidirectory, year, number, distance )