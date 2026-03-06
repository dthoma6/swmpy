#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 15:29:14 2026

@author: Dean Thomas
"""

import pandas as pd
from datetime import datetime, timedelta, timezone
from os.path import join, basename, dirname
import fortranformat as ff

def kp_read(info, year):
    """Parses Kp data text files, and stores the data in a pickle file for
    future analyses.  

    Inputs:
        info = infomation, such as paths to directories, for run
        
        year = Kp text file for specified year is read
                
    Outputs:
        pickle file with parsed Kp solar wind data
    """
    # path to Kp file
    filepath = join(info["Kp Directory"], "Kp_" + str(year) + ".txt" )

    # Format derived from description in https://kp.gfz.de/app/format/Kp_ap_Ap_SN_F107_format.txt
    reader = ff.FortranRecordReader('(I5, 2I3, I6, F8.1, I5, I3, 8F7.3, 8I5, I5, I4, 2F9.1, I2)')
        
    # Names of Kp variables plus datetime generated from Kp variables
    names = ["Year", "Month", "Day", "Days", "Days (mid)", "BSR", "Day in BSR",  
             "Kp1", "Kp2", "Kp3", "Kp4", "Kp5", "Kp6", "Kp7", "Kp8", 
             "ap1", "ap2", "ap3", "ap4", "ap5", "ap6", "ap7", "ap8", 
             "Ap", "SN", "F10.7obs", "F10.7adj", "D", "Datetime"]
    
    # Read Kp date into a list
    with open(filepath, 'r') as file:
        
        # Empty list for data
        kp = [] 
        cnt = 0
    
        # Read Kp data from file
        for line in file:
            tmp = reader.read(line)
    
            # Generate datetime from Kp data
            yy = tmp[0] # year
            mm = tmp[1] # month
            dd = tmp[2] # day
    
            date = datetime(yy, mm, dd, 0, 0, tzinfo=timezone.utc) 
            tmp.append(date)
    
            # Append Kp date to list
            kp.append(tmp)
            cnt += 1
            if cnt % 100 == 0: print( 'Processed line: ', cnt )
    
    # Create dataframe with Kp data    
    kpdf = pd.DataFrame( kp, columns = names )
    
    # Save data to pickle file
    base = basename( filepath )
    dire = dirname( filepath )
    newfile = join( dire, base + '.pkl' )
    kpdf.to_pickle( newfile )
    return

def kp_stats(info, year, number):
    """Provides Kp values to match time periods for SuperMAG stats data.  

    Inputs:
        info = information, such as paths to directories, for run

        year = year for associated SuperMAG station data filenames
        
        number = number of minutes over which SuperMAG statistics gathered 
            (e.g., number = 30 examine 30 minutes windows,  To match, we supply
             Kp value for each 30 minute window.)
        
     Outputs:
        pickle files with Kp values
    """
    # Check that we stay within 180 minute windows.  Kp values are for 3 hour
    # windows.  If number > 180, we would need to do some sort of interpolation.
    if number > 180: 
        import sys
        sys.exit('Error: number must <= 180 minutes')

    # Read Kp pickle file
    kpdirectory = info["Kp Directory"]
    filepath = join(kpdirectory, "Kp_" + str(year) + ".txt.pkl" )
    df = pd.read_pickle( filepath )
    
    # We have 8 Kp entries per day. So each Kp entry covers 3 hours
    # (See the columns labeled Kp1, Kp2, ... Kp8 in dataframe)
    # We have to repeat each Kp value num times...
    num = int(3*60/number)
    
    # Storage for dates and Kp values
    dval  = []
    kpval = []
    
    # Loop through entries in Kp dataframe
    for i in range(len(df)):
    
        # Each line in Kp dataframe is for one day
        tmpdate = df['Datetime'][i] 
        
        # 8 Kp values/day
        for j in range(8):
            # num values per 3 hr window
            for k in range(num):
                dval.append( tmpdate + timedelta( hours=3*j, minutes=k*number ) )
                kpval.append( df['Kp'+str(j+1)][i] )
        
    # We're done with the loop, store the stats in a pickle file
    kpdf = pd.DataFrame( ) 
    kpdf['Datetime'] = dval
    kpdf['Kp']       = kpval

    file = 'Kp-stats-' + str(number) + 'min-' + str(year) + '.pkl'
    kpdf.to_pickle( join(kpdirectory, file) ) 
    
    return

def kp_raw(info, year):
    """Provides Kp values to one minute time periods for raw SuperMAG data.  

    Inputs:
        info = information, such as paths to directories, for run

        year = year for associated SuperMAG station data filenames
        
     Outputs:
        pickle files with Kp values
    """

    # Read Kp pickle file
    kpdirectory = info["Kp Directory"]
    filepath = join(kpdirectory, "Kp_" + str(year) + ".txt.pkl" )
    df = pd.read_pickle( filepath )
    
    # We have 8 Kp entries per day. So each Kp entry covers 3 hours
    # (See the columns labeled Kp1, Kp2, ... Kp8 in dataframe)
    # We have to repeat each Kp value num times...
    num = 180
    
    # Storage for dates and Kp values
    dval  = []
    kpval = []
    
    # Loop through entries in Kp dataframe
    for i in range(len(df)):
    
        # Each line in Kp dataframe is for one day
        tmpdate = df['Datetime'][i] 
        
        # 8 Kp values/day
        for j in range(8):
            # num values per 3 hr window
            for k in range(num):
                dval.append( tmpdate + timedelta( hours=3*j, minutes=k ) )
                kpval.append( df['Kp'+str(j+1)][i] )
        
    # We're done with the loop, store the stats in a pickle file
    kpdf = pd.DataFrame( ) 
    kpdf['Datetime'] = dval
    kpdf['Kp']       = kpval

    file = 'Kp-stats-' + str(year) + '.pkl'
    kpdf.to_pickle( join(kpdirectory, file) ) 
    
    return

