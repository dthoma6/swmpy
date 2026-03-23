#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 10:39:19 2026

@author: Dean Thomas
"""
from os.path import join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import swmpy as swm
from file_info import file_info

if __name__ == "__main__":
    
    # Whether to download SuperMAG data 
    DOWNLOAD = False
    
    # Whether to calculate SuperMAG statistics
    STATS = False
    
    # Whether to generate raw SuperMAG files
    RAW = False
    
    # Whether to plot SuperMAG statistics
    PLOTSTATS = False
    
    # Whether to plot Bn, Be, Bd
    PLOTB = False    
    
    # Whether to plot world maps of SuperMAG sites
    PLOTWORLD = True

    ############## download SuperMAG
    
    # We will download SuperMAG data for these years
    yrs=file_info["years"]
    
    # Divide SuperMAG data into chunks number minutes long.
    # For number=30, we will have the first chunk at 
    # 0, 1, 2, ... 28, 29 minutes. The second chunk at 
    # 30, 31, 32, ... 58, 59 minutes, etc. 
    nums = file_info["numbers"]

    if DOWNLOAD: 
        for yr in yrs:
            swm.supermag_download_all(file_info, yr)

    ################# SuperMAG stats
    
    if STATS:
        for yr in yrs:
            for num in nums: 
                if num is None: break; # Handled by RAW below
                swm.supermag_stats(file_info, yr, num)
                
    if RAW:
        for yr in yrs:
            swm.supermag_raw(file_info, yr)

    ################# SuperMAG stats
    
    if PLOTSTATS:
        for yr in yrs:
            for num in nums: 
                swm.supermag_plots(file_info, yr, num)

    ################# SuperMAG magnetic field plots
    
    # Takes pickle files created by download_SuperMAG.py and plots
    # magnetic field data in NEZ
    
    if PLOTB:
        # Set some plot configs
        swm.set_plot_rcParams()
    
        # Read pickle files for each SuperMAG station and plot magnetic field data
        for yr in yrs:
            # get list of stations
            stations = swm.stations_list(yr, file_info["SuperMAG Directory"])
    
            for station in stations:
                file = station + '-' + str(yr) + '.pkl' 
                df = pd.read_pickle( join(file_info["SuperMAG Directory"], file ) )
                
                N_nez = [temp['nez'] for temp in df.N]
                E_nez = [temp['nez'] for temp in df.E]
                Z_nez = [temp['nez'] for temp in df.Z]
                
                dval = df.Datetime.to_list()
                
                plt.plot( dval, N_nez, label='N' )
                plt.plot( dval, E_nez, label='E' )
                plt.plot( dval, Z_nez, label='Z' )
                plt.legend()
                plt.title(station)
                plt.xlabel('Date')
                plt.ylabel(r'$\delta B$')
                plt.grid(True)
                plt.show()

    ################## SuperMAG world maps
    
    if PLOTWORLD:
        # SuperMAG station file, downloaded from:
        # https://supermag.jhuapl.edu/mag/?fidelity=high&tab=stationinfo&start=2023-01-01T00%3A00%3A00.000Z&interval=00%3A23%3A59
        # Slight editing of file required too give every column a name
        csv_file = join( file_info["SuperMAG Directory"], 
                        '20251209-18-31-supermag-stations.csv' )
        
        # Read SuperMAG csv file that contains the lat/longs of the magnetometer stations
        df = pd.read_csv(csv_file)

        swm.set_plot_rcParams(fontsize=6, figsize=[6,3])
    
        # Draw world map with all SuperMAG sites with matching SPUD sites
        import cartopy.crs as ccrs
        # Map -180->180 in longitude
        proj = ccrs.PlateCarree()
        
        for yr in yrs:
            # get list of stations
            stations = swm.stations_list(yr, file_info["SuperMAG Directory"])
        
            # Read stations names and lat/lons from dataframe
            lat = np.empty(len(stations), dtype=float)
            lon = np.empty(len(stations), dtype=float)
        
            for i in range(len(stations)):
                
                # Extract matching station data from dataframe
                # then get lat/lon
                tmp = df[df['IAGA'] == stations[i]]
                tmp = tmp.reset_index()
                lat[i] = tmp['GEOLAT'][0]
                lon[i] = tmp['GEOLON'][0]
                
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.coastlines()
            plt.scatter( lon, lat )
            plt.xlim( -180, 180 )
            plt.ylim( -90, 90 )
            plt.title(str(yr))
        
            i=0
            for xy in zip(lon, lat):
                plt.annotate(stations[i], xy)
                i +=1
        
            plt.grid()
            plt.savefig( join(file_info["SuperMAG Directory"], 
                                      'world-map-' + str(yr) +'.png') )
            plt.show()

