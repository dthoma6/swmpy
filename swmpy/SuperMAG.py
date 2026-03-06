#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 13:51:03 2025

@author: Dean Thomas
"""
# Downloads the SuperMAG data for each SuperMAG station that matches a SPUD 
# station (both stations in a 60nm radius circle). spud_supermag_matching.py
# generated the list of matching stations.

from datetime import date, timedelta
from os.path import join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone

from swmpy.utils import stations_list, set_plot_rcParams, calc_dXdt
import swmpy.supermag_api as sm

# SuperMAG data description from: 
# https://supermag.jhuapl.edu/mag/?fidelity=high&start=2023-01-01T00%3A00%3A00.000Z&interval=00%3A23%3A59&tab=description
#
# SuperMAG 1-sec data have been rotated into a local magnetic coordinate system 
# (see below) and the main field has been removed using the baseline technique 
# described in the first reference below (e.g. Gjerloev, J. W., 2012).
#
# Data Cadence
# The SuperMAG 1-sec data have been derived from measurements with a 0.5 Hz or 
# higher sampling rate (see ULF Parameters for details)
#
# Data Gaps
# Data gaps are indicated by IEEE 754-1985 NaN values in downloadable netCDF 
# files and by the value 999999 in data tables and popups on the website. The 
# value 999999 is also used to indicated data gaps in downloadable ASCII text files.
#
# Coordinate System
# Global studies require all data to be rotated into a common known coordinate 
# system. Data provided to SuperMAG from the collaborators are typically in either:
#
# -- Geographic coordinates (north (X), east (Y), vertical down (Z))
# -- Geographic coordinates (horizontal intensity (H), declination (D) and vertical down (Z))
# -- Geomagnetic coordinates> (magnetic north (H), magnetic east (D) and vertical down (Z))
#
# with or without baselines subtracted. During intitial setup the sensor axes 
# are oriented in either the geographic or local magnetic coordinate system. 
# The Earth main field, however, is constantly changing so the geomagnetic 
# coordinate system is time dependent. The various uncertainties in mind SuperMAG 
# decided to make no assumptions as to the initial setup of the magnetometer other 
# than the Z-axis being vertical. Using the two horizontal components SuperMAG 
# determines a slowly varying time dependent declination angle and subsequently 
# rotates the horizontal components into a local magnetic coordinate system for 
# which the magnetic east component (E) is minimized and the magnetic north 
# component (N) is maximized. Note that geomagnetic coordinates are routinely 
# labeled HDZ although the units of the D-component can be nT or an angle. 
# Likewise, the D-component is often found to have a significant offset. As a 
# consequence SuperMAG decided to denote the components:
# B=(BN,BE,BZ)
#
# where
# -- N-direction is local magnetic north
# -- E-direction is local magnetic east
# -- Z-direction is vertically down
#
# By definition the typical value (offset) of the E-component is zero. This 
# reference system is independent of the actual orientation of the two horizontal 
# magnetometer axes and the data can be rotated to any desired coordinate system 
# using the appropriate IGRF model.
#
# Magnetic Local Time (MLT) is calculated using the solar local time (Jean Meeus, 
# Astronomical Algorithms, 2nd edition, ISBN-13: 978-0943396613) and the AACGM 
# system. For more info and a cautionary note see: https://omniweb.gsfc.nasa.gov/vitmo/cgmm_des.html
#
# Main Field or Baseline
# For SuperMAG 1-sec data the main field, or baseline, has been removed using 
# the technique described in Gjerloev, J. W., 2012 to subtract the daily variations 
# and yearly trend (see description of 1 minute resolution Mmagnetometer data 
# for details).
#
# ULF Parameters
# All ULF data products are derived from the 1-sec SuperMAG data (see ULF 
# Parameters for a description of the details).

#############################################################################

def supermag_download(info, year):
    """Downloads the SuperMAG data for each SuperMAG station that matches
    a SPUD station (both stations in a circle with 60nm radius). 
    spud_supermag_matching.py generated the list of matching stations and stored 
    them in SPUD_SuperMAG_Matches.pkl. SuperMAG files for specified year are 
    retrieved and saved as pickle files.

    Inputs:
        info = infomation, such as paths to directories, for run
        
        year = all files for specified year are downloaded
                       
     Outputs:
        pickle file for each station
    """

    smdirectory = info["SuperMAG Directory"]
    userid = info["SuperMAG UserID"]
    spuddirectory = info["Match Directory"]

    start = [year,1,1] 
    end   = [year,12,31]
    
    # Read pickle file from spud_supermag_matching.py
    # Get the SuperMAG stations (iaga) that have matching SPUD stations
    matchdf = pd.read_pickle( join( spuddirectory, 'SPUD_SuperMAG_Matches.pkl' ) )
    matchdf = matchdf[matchdf['SPUD station'] != '']  # skip if no matching SPUD station
    iaga    = matchdf['IAGA'].values
    
    # Create dict with empty dataframes in which we'll store the SuperMAG data.
    # We concat each day's data onto each station's previous dataframe
    dfs ={}
    for station in iaga:
        dfs[station] = pd.DataFrame()
        
    # Create lists for failed SuperMAGGetData calls
    fail_day = []
    fail_msg = []
    
    # Determine data range that we're looking at
    start_date = date(start[0], start[1], start[2]) 
    end_date   = date(end[0], end[1], end[2])   
    delta      = end_date - start_date  
    
    # Loop through all days between start and end
    for i in range(delta.days + 1):
        day = start_date + timedelta(days=i)
    
        # Find which stations we have data on "day" (each day is 86400 sec long)
        (status,stations) = sm.SuperMAGGetInventory(userid,day,86400)
        
        if status == 1: # success
            print(day.year, day.month, day.day)
            
            # Loop through each SuperMAG station and see if data is available
            for station in iaga:
                
                # if iaga station is in stations, request SuperMAG data
                if station in stations:
                    (status,data) = sm.SuperMAGGetData(userid, day, 86400,
                                            'all,delta=start,baseline=yearly', 
                                            station)
                    
                    if status == 1 and not data.empty: 
                        print(station)
                        
                        # Convert tval timestamp (seconds) to datetime
                        data['Datetime'] = pd.to_datetime(data.tval, unit='s', 
                                                          utc=True)
                
                        # Concat the data onto the station dataframe
                        dfs[station] = pd.concat([dfs[station], data])
        
        else: # failure
            fail_day.append( day )
            fail_msg.append( stations )
    
    # print/save info on failed calls to SuperMAGGetData
    if len(fail_day) > 0:
        for i in range( len(fail_day) ):
            print( '############# Failed calls to SuperMAGGetInventory' )
            print( fail_day[i], fail_msg[i] )

    if len(fail_day) > 0:
        fails = pd.DataFrame()
        fails['Day'] = fail_day
        fails['Error'] = fail_msg
        file = 'Failed-SuperMAGGetInventory-' + str(year) + '.pkl'
        fails.to_pickle(join(smdirectory, file))
    
    # Save SuperMAG dataframes
    stations_w_data = []
    for station in iaga:
        if not dfs[station].empty:
            stations_w_data.append(station)
            file = station + '-' + str(year) + '.pkl'
            dfs[station].to_pickle(join(smdirectory, file))

    # Save which SuperMAG stations have data
    file = 'stations-' + str(year) + '.pkl'
    dfyr = pd.DataFrame({'Stations': stations_w_data})
    dfyr.to_pickle(join(smdirectory, file))
    return

def supermag_download_all(info, year):
    """Downloads the SuperMAG data for each SuperMAG station for a given year.
    This differs from supermag_download, which is restricted to matching SPUD
    and SuperMAG sites.  There is no matching restriction in this procedure.
    All SuperMAG files available for specified year are retrieved and saved as 
    pickle files.

    Inputs:
        info = infomation, such as paths to directories, for run
        
        year = all files for specified year are downloaded
                       
     Outputs:
        pickle file for each station
        
        pickle for for station list
    """

    smdirectory = info["SuperMAG Directory"]
    userid = info["SuperMAG UserID"]

    start = [year,1,1] 
    end   = [year,12,31]
    
    # Create dict in which we'll store the SuperMAG dataframes.
    # We concat each day's data onto each station's previous dataframe
    dfs ={}
        
    # Create lists for failed SuperMAGGetData calls
    fail_day = []
    fail_msg = []
    
    # Determine data range that we're looking at
    start_date = date(start[0], start[1], start[2]) 
    end_date   = date(end[0], end[1], end[2])   
    delta      = end_date - start_date  
    
    # Loop through all days between start and end
    for i in range(delta.days + 1):
        day = start_date + timedelta(days=i)
    
        # Find which stations we have data on "day" (each day is 86400 sec long)
        (status,stations) = sm.SuperMAGGetInventory(userid,day,86400)
        
        if status == 1: # success
            print(day.year, day.month, day.day)
            
            # Loop through each SuperMAG station and see if data is available
            for station in stations:
                (status,data) = sm.SuperMAGGetData(userid, day, 86400,
                                        'all,delta=start,baseline=yearly', 
                                        station)
                
                if status == 1 and not data.empty: 
                    print(station)
                    
                    # Convert tval timestamp (seconds) to datetime
                    data['Datetime'] = pd.to_datetime(data.tval, unit='s', 
                                                      utc=True)
            
                    # Concat the data onto the station dataframe
                    if station in dfs:
                        dfs[station] = pd.concat([dfs[station], data])
                    else:
                        dfs[station] = data
        
        else: # failure
            fail_day.append( day )
            fail_msg.append( stations )
    
    # print/save info on failed calls to SuperMAGGetData
    if len(fail_day) > 0:
        for i in range( len(fail_day) ):
            print( '############# Failed calls to SuperMAGGetInventory' )
            print( fail_day[i], fail_msg[i] )

    if len(fail_day) > 0:
        fails = pd.DataFrame()
        fails['Day'] = fail_day
        fails['Error'] = fail_msg
        file = 'Failed-SuperMAGGetInventory-' + str(year) + '.pkl'
        fails.to_pickle(join(smdirectory, file))
    
    # Save SuperMAG dataframes
    stations_w_data = []
    for station in dfs.keys():
        if not dfs[station].empty:
            stations_w_data.append(station)
            file = station + '-' + str(year) + '.pkl'
            dfs[station].to_pickle(join(smdirectory, file))

    # Save which SuperMAG stations have data
    file = 'stations-' + str(year) + '.pkl'
    dfyr = pd.DataFrame({'Stations': stations_w_data})
    dfyr.to_pickle(join(smdirectory, file))
    return

def supermag_stats(info, year, number):
    """Calculates statistics (mean) for parameters in the SuperMAG data.  

    Inputs:
        info = infomation, such as paths to directories, for run

        year = year for associated SuperMAG station data filenames
        
        number = number of minutes over which statistics gathered (e.g., number = 30
             examine 30 minutes windows)
        
     Outputs:
        pickle files with statistics
    """
        
    # get list of stations
    smdirectory = info["SuperMAG Directory"]
    stations = stations_list(year, smdirectory)
    
    # Read pickle filenames for each SuperMAG station and determine stats for magnetic field
    for station in stations:
        print( year, station )
        
        filename = station + '-' + str(year) + '.pkl' 
        df = pd.read_pickle( join(smdirectory, filename) )
        
        # Get magnetic field data for station
        N_nez = np.array( [temp['nez'] for temp in df.N] )
        E_nez = np.array( [temp['nez'] for temp in df.E] )
        # Z_nez = np.array( [temp['nez'] for temp in df.Z] )
        B_H   = np.sqrt( N_nez**2 + E_nez**2 )
        
        tval = df.tval.to_numpy()
        dval = df.Datetime.to_numpy(dtype=datetime)
        
        glon   = df.glon.to_numpy()
        glat   = df.glat.to_numpy()
        mlt    = df.mlt.to_numpy()
        mcolat = df.mcolat.to_numpy()
        
        # Create lists to store stats in
        # B_mag_mean = []
        B_H_mean   = []
        tval_mean  = []
        dval_mean  = []
        cnt_mean   = []
        glon_df    = []
        glat_df    = []
        mlt_df     = []
        mcolat_df  = []
        dBHdt_mean = []
        
        maxidx = len(B_H)
        
        # Below we determine startdate and enddate for each chuck of data that is 
        # number minutes long.
        
        # Get time since UNIX epoch for first day of year, the SuperMAG data uses these dates
        # Note, we start at 0 minutes into the year.  See comment for number variable
        startdate = datetime(year, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp()
        startidx = 0
    
        # Loop through SuperMAG data, find data between startdate and enddate.
        for i in range(len(dval)): # This for-loop is too long, but we break out of it below
        
            # To determine enddate, number-1 is needed.  See comment for number variable.
            # If number=30, we want times at 0, 1, 2, 3, ... 28, 29 minutes
            # So if startdate = 0, we add number-1 to get enddate
            #
            # (number-1)*60 because startdate and enddate are in seconds
            enddate = startdate + (number-1)*60  
            
            # print(datetime.fromtimestamp(startdate), datetime.fromtimestamp(enddate))
            
            # Counter to keep track of where we are in reading chunks of data.
            # Each chunk is nominally number data points long.  So in the loops below, we
            # skip number forward, unless some values are missing, then we skip j forward
            cnt = number
            
            # Arrays to temporarily store values for calculating stats
            # B_mag_tmp = np.full( number, np.nan, dtype=float )
            B_H_tmp   = np.full( number, np.nan, dtype=float )
            tval_tmp  = np.full( number, np.nan, dtype=float )
            cnt_tmp   = 0 # counter to see if we have enough samples
    
            # Loop through array to get number samples for stats        
            for j in range(number):
    
                # Don't go past end of array
                if j+startidx < maxidx:
                    
                    # Make sure data is past the startdate, if not we have a problem
                    if tval[j+startidx] >= startdate:
                        
                        # Make sure we don't run past the enddate.
                        # If this happens, some time steps were skipped in the SuperMAG data
                        if tval[j+startidx] <= enddate:
                            B_H_tmp[j]   = B_H[j+startidx]
                            tval_tmp[j]  = tval[j+startidx]
                            cnt_tmp      += 1
                        else:
                            cnt = j
                            break
                        
                    else:
                        print( 'Time problem ', j, tval[j+startidx], startdate )
                    
            # If we have enough points to determine mean and store stats
            if cnt_tmp > 1:
                B_H_mean.append( np.nanmean(B_H_tmp) )
                tval_mean.append( startdate )
                dval_mean.append( datetime.fromtimestamp(startdate, tz=timezone.utc) )
                cnt_mean.append( cnt_tmp )
                glon_df.append( glon[startidx] )   
                glat_df.append( glat[startidx] )   
                mlt_df.append( mlt[startidx] )    
                mcolat_df.append( mcolat[startidx] ) 
                
                # Calculate time derivatives of BH
                dBHdt = calc_dXdt(B_H_tmp, tval_tmp)
                dBHdt_mean.append( np.nanmean(dBHdt) )

            
            # Update startdate for next loop
            # It's one minute (60 seconds) past last enddate
            startdate = enddate + 60
            startidx += cnt
            cnt       = number
            if startidx >= maxidx: break
        
        # We're done with the loop, store the stats in a pickle filename
        statsdf = pd.DataFrame( ) 
        statsdf['tval']        = tval_mean
        statsdf['Datetime']    = dval_mean
        statsdf['B_H Mean']    = B_H_mean
        statsdf['dB_H/dt Mean']= dBHdt_mean
        statsdf['Sample Size'] = cnt_mean
        statsdf['glon']        = glon_df
        statsdf['glat']        = glat_df
        statsdf['mlt']         = mlt_df
        statsdf['mcolat']      = mcolat_df
    
        file = station + '-stats-' + str(number) + 'min-' + str(year) + '.pkl'
        statsdf.to_pickle( join(smdirectory, file) ) 
        
    return

def supermag_raw(info, year):
    """Converts raw SuperMAG to same format as statistics data (see supermag_stats)
    so autogluon fit can be made on the raw data using the same routines as used
    for the other data.

    Inputs:
        info = infomation, such as paths to directories, for run

        year = year for associated SuperMAG station data filenames
        
     Outputs:
        pickle files with raw data
    """
        
    # get list of stations
    smdirectory = info["SuperMAG Directory"]
    stations = stations_list(year, smdirectory)
    
    # Read pickle filenames for each SuperMAG station and determine stats for magnetic field
    for station in stations:
        print( year, station )
        
        filename = station + '-' + str(year) + '.pkl' 
        df = pd.read_pickle( join(smdirectory, filename) )
        
        # Get magnetic field data for station
        N_nez = np.array( [temp['nez'] for temp in df.N] )
        E_nez = np.array( [temp['nez'] for temp in df.E] )
        # Z_nez = np.array( [temp['nez'] for temp in df.Z] )
        B_H   = np.sqrt( N_nez**2 + E_nez**2 )
        
        tval = df.tval.to_numpy()
        dval = df.Datetime.to_numpy(dtype=datetime)
        
        glon   = df.glon.to_numpy()
        glat   = df.glat.to_numpy()
        mlt    = df.mlt.to_numpy()
        mcolat = df.mcolat.to_numpy()
        
        dBHdt = calc_dXdt(B_H, tval)
        
        statsdf = pd.DataFrame( ) 
        statsdf['tval']        = tval
        statsdf['Datetime']    = dval
        statsdf['B_H Mean']    = B_H
        statsdf['dB_H/dt Mean']= dBHdt
        statsdf['Sample Size'] = np.ones( len(tval) )
        statsdf['glon']        = glon
        statsdf['glat']        = glat
        statsdf['mlt']         = mlt
        statsdf['mcolat']      = mcolat
   
        file = station + '-stats-None-' + str(year) + '.pkl'
        statsdf.to_pickle( join(smdirectory, file) ) 
        
    return

def supermag_plots(info, year, number):
    """Generates plots of the statistics (mean) for a subset of the parameters 
    in the SuperMAG magnetometer files.  Used as a quick check of results. 

    Inputs:
        info = infomation, such as paths to directories, for run

        year = year for associated SuperMAG station data filenames
        
        number = number of minutes over which statistics gathered (e.g., number = 30
             examine 30 minutes windows)
        
     Outputs:
        png files of plotted data
    """
        
    # get list of stations
    smdirectory = info["SuperMAG Directory"]
    stations = stations_list(year, smdirectory)
    
    # Read pickle filenames for each SuperMAG station and determine stats for magnetic field
    for station in stations:
        filename = station + '-' + str(year) + '.pkl' 
        df = pd.read_pickle( join(smdirectory, filename) )
        
        # Get magnetic field data for station
        # We use it to see when B field is valid for dataavail below
        N_nez = np.array( [temp['nez'] for temp in df.N] )
        E_nez = np.array( [temp['nez'] for temp in df.E] )
        Z_nez = np.array( [temp['nez'] for temp in df.Z] )
        B_mag = np.sqrt( N_nez**2 + E_nez**2 + Z_nez**2 )
        
        dval = df.Datetime.to_numpy(dtype=datetime)
            
        file = station + '-stats-' + str(number) + 'min-' + str(year) + '.pkl'
        statsdf = pd.read_pickle( join(smdirectory, file) ) 
        
        # Create array showing when data is available.  
        # A check to verify that we're doing everything correctly.
        # 
        # dataavail is number+2 when a point is non-nan, and nan otherwise
        #
        # Plot along side cnt_std below to see if they match up.
        B_H = statsdf['B_H Mean']
        dataavail = np.full(len(B_mag), np.nan, dtype=float)    
        for i in range(len(B_mag)): 
            if not np.isnan( B_mag[i] ): 
                dataavail[i] = number+2
                
        # Plot the results for each station
        set_plot_rcParams( fontsize=5 )
        fig, ax = plt.subplots(3, sharex=True)
        ax[0].scatter( statsdf['Datetime'], statsdf['B_H Mean'], 
                      label=r'$B_H$ Mean', s=3 )
        ax[1].scatter( statsdf['Datetime'], statsdf['dB_H/dt Mean'], 
                      label=r'$dB_{H}/dt$ Mean', s=3 )
        ax[2].scatter( statsdf['Datetime'], statsdf['Sample Size'], 
                      label=r'Number of Samples', s=3 )
        ax[2].scatter( dval    , dataavail, label=r'Data Available', s=3 )
        ax[0].set_ylabel(r'$B_H$ Mean')
        ax[1].set_ylabel(r'$dB_{H}/dt$ Mean')
        ax[2].set_ylabel(r'Sample Size')
        ax[2].legend()
        ax[2].set_xlabel("Date")
        fig.suptitle(station + ' ' + str(number) + 'min '+ str(year))
        plt.show()
        file = station + '-stats-' + str(number) + 'min-' + str(year) + '.png'
        fig.savefig( join(smdirectory, file) )
        
    return


