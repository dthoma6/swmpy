#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 09:37:44 2026

@author: Dean Thomas
"""

from os.path import join

############################################################
# Contain infomation, such as paths to directories, for run
############################################################

data_dir = r'/Volumes/PhysicsHD/swmpy/'

# "SuperMAG UserID" ==> UserID for SuperMAG website
# "years" ==> downloaded data will cover these years
# "numbers" ==> number of minutes over which statistics are calcualted (e.g., 
#     number = 30 means 30 minutes windows are used to determine mean and standard 
#     deviation of solar wind and magnetometer data)
# "distances" ==> OMNI solar wind data is ballistically propagated to these 
#     distances from Earth (measured in Re)
# "SuperMAG Directory" ==> Directory for SuperMAG magnetometer data
# "SPUD Directory" ==> Directory for SPUD data
# "OMNI Directory" ==> Directory for OMNI solar wind data
# "Match Directory" ==> Directory for matching SPUD and SuperMAG stations

file_info = {
        "info": "file",
        "SuperMAG UserID": "USERNAME",
        "years": [2023,2024,2025,],
        "distances": [None,], #[None, 10, 8, 6, 4, 2, 0],
        "numbers": [5,10,], #[15, 20, 25,], #[15, 20, 25, 30,],
        "SuperMAG Directory": join(data_dir, "SuperMag"),
        "SPUD Directory": join(data_dir, "SPUD_EDI_bundle_2025-12-29T15"),
        "OMNI Directory": join(data_dir, "OMNI"),
        "Kp Directory": join(data_dir, "Kp"),
        "Match Directory": join(data_dir, "SPUD_SuperMAG_matching"),
        "Fit Directory": join(data_dir, "Fit"),
}
    
