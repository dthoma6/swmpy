#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 17:05:32 2026

@author: Dean Thomas
"""

from os.path import join
import pandas as pd
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def _remove_correlated_omni( df ):
    """Drop dataframe columns that contain highly-correlated 
    OMNI solar wind variables identifed in scattermatrix.py 
     
    Inputs:
        df = dataframe that includes OMNI and SuperMAG data
        
    Outputs:
        df = dataframe without correlated vOMNI ariables
    """

    # Same as By, Bz in GSE, just different coordinate system.  We dropped 
    # these before calculating correlation coefficents
    df = df.drop(['By, nT (GSM) Mean'], axis=1)
    df = df.drop(['Bz, nT (GSM) Mean'], axis=1)
    
    # Drop derived variables, which are based on measured data
    # These have high correlation to measured data
    #
    # Derived variables described in OMNI documentation at 
    # https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/modified/hro_modified_format.txt
    
    # Flow pressure 
    df = df.drop(['Flow pressure, nPa Mean'], axis=1)

    # Electric field 
    df = df.drop(['Electric field, mV/m Mean'], axis=1)

    # Plasma beta
    df = df.drop(['Plasma beta Mean'], axis=1)

    # Alfven mach 
    df = df.drop(['Alfven mach number Mean'], axis=1)

    # Magnetosonic mach
    df = df.drop(['Magnetosonic mach number Mean'], axis=1)

    # Drop other variables that are highly correlated, see scatter_matrix.py

    # Bx and By correlated
    df = df.drop(['By, nT (GSE) Mean'], axis=1)

    # Temperature correlated with |V| and Vx
    df = df.drop(['Temperature, K Mean'], axis=1)
    
    return df

def _remove_unused( df ):
    """Drops variables from dataframe that we don't use in fits.  
    
    Inputs:
        df = dataframe that includes OMNI and SuperMAG data
        
    Outputs:
        dataframe without the unused columns
    """
    
    # Drop these variables, we won't use them in fit     
    df = df.drop(['tval_x'], axis=1)
    df = df.drop(['Datetime_x'], axis=1)
    df = df.drop(['Sample Size_x'], axis=1)
    df = df.drop(['tval_y'], axis=1)
    df = df.drop(['Datetime_y'], axis=1)
    df = df.drop(['Sample Size_y'], axis=1)
    df = df.drop(['glon'], axis=1)
    df = df.drop(['glat'], axis=1)
    df = df.drop(['mlt'], axis=1)
    df = df.drop(['mcolat'], axis=1)
    df = df.drop(['Datetime'], axis=1)
    df = df.drop(['Kp'], axis=1)

    return df
 
def _remove_kp_upper( df, kp ):
    """Drops rows that have a Kp > kp  
    
    Inputs:
        df = dataframe that includes OMNI and SuperMAG data
        
        kp = Kp threshold
                
    Outputs:
        dataframe without rows values above Kp threshold
    """
    
    if kp is not None:
        df = df.drop(df[df['Kp'] > kp].index)

    return df
 
def _remove_kp_lower( df, kp ):
    """Drops rows that have a Kp < kp  
    
    Inputs:
        df = dataframe that includes OMNI and SuperMAG data
        
        kp = Kp threshold
                
    Outputs:
        dataframe without rows values below Kp threshold
    """
    
    if kp is not None:
        df = df.drop(df[df['Kp'] < kp].index)

    return df
 
def _merge_files( file_info, run_info, drop_large=True ):
    """Combines the SuperMAG and OMNI data for the given year, number, distance, 
    station, and Kp into a single dataframe
    
    Inputs:
        file_info = information, such as paths to directories

        run_info = information on flag settings, etc. for this run.  Includes
            year, number, distance, station, uselogy, etc.
            
        drop_large = drop excessively large B_H values (i.e., > 100,000 nT)
        
    Outputs:
        dataframe of merged data
    """
    
    # Get parameters needed below from file_info and run_info
    supermagdirectory = file_info["SuperMAG Directory"] 
    omnidirectory     = file_info["OMNI Directory"]
    kpdirectory       = file_info["Kp Directory"]
    
    year     = run_info['year']
    number   = run_info['number']
    distance = run_info['distance']
    station  = run_info['station']
    kpupper  = run_info['Kp Upper']
    kplower  = run_info['Kp Lower']
    uselogy  = run_info['uselogy']
                 
    # Get OMNI, SuperMAG, and Kp dataframes and merge them.  Filenames depend 
    # upon run_info parameters
    if number is None: # Implies that distance is None
        filename = 'OMNI-stats-None-' + str(year) + '.pkl' 
    else:
        if distance is None:
            filename = 'OMNI-stats-' + str(number) + 'min-' + str(year) + '.pkl'
        else:
            filename = 'OMNI-stats-' + str(distance) + 'Re-' + str(number) + 'min-' + str(year) + '.pkl'
    omnidf = pd.read_pickle( join(omnidirectory, filename) )  
    omnidf.set_index('Datetime')

    if number is None:
        filename = station + '-stats-None-' + str(year) + '.pkl'
    else:
        filename = station + '-stats-' + str(number) + 'min-' + str(year) + '.pkl'
    smdf = pd.read_pickle( join(supermagdirectory, filename) )
    smdf.set_index('Datetime')
    
    if number is None:
        filename = 'Kp-stats-' + str(year) + '.pkl'
    else:
        filename = 'Kp-stats-' + str(number) + 'min-' + str(year) + '.pkl'
    kpdf = pd.read_pickle( join(kpdirectory, filename) )
    kpdf.set_index('Datetime')

    df = pd.merge(omnidf, smdf, left_index=True, right_index=True)
    df = pd.merge(df,     kpdf, left_index=True, right_index=True)
    
    # Based on the analysis in B_ Distribution.ipynb, we drop entries
    # with excessively large B_H values.  We observed a small number of values
    # of the order of 1,000,000 nT.  A cut-off of 10,000 nT removed these points.
    # In our sample of 6,852,208 points, 19 points were above 10,000 nT.
    if drop_large:
        df = df.drop(df[df['B_H Mean'] > 10000.].index)

    # Drop highly-correlated OMNI variables
    df = _remove_correlated_omni(df)
        
    # If we want log of response variable do it here
    if uselogy:
        df['B_H Mean']     = np.log10( df['B_H Mean'] )

    # if kpupper != None, drop rows with Kp values above threshold
    # if kplower != None, drop rows with Kp values below threshold
    if kpupper is not None: 
        df = _remove_kp_upper( df, kpupper )
    if kplower is not None: 
        df = _remove_kp_lower( df, kplower )
    
    # Add circular response for mlt. Multiply mlt by 15 -> 24*15=360 
    df['cosmlt'] = np.cos( 15.*df.mlt*np.pi/180. )
    df['sinmlt'] = np.sin( 15.*df.mlt*np.pi/180. )
    
    # Add circular response for mcolat if we use get_data_all. 
    # aka run_info['info'] == 'all'. 
    if run_info['info'] == 'all':
        df['cosmcolat'] = np.cos( df.mcolat*np.pi/180. )
        df['sinmcolat'] = np.sin( df.mcolat*np.pi/180. )
    
    # Drop variables that we won't use them in fit     
    df = _remove_unused( df )
    
    return df

def get_data_one( file_info, run_info, random_state=42, test_size=0.2, drop_large=True ):
    """Merges the SuperMAG and OMNI data, processes them based on the options, 
    and returns training and test dataframes with B_H Mean as the response variable.
    In this version, the dataframe is for a specific station during a specific
    year.
    
    Inputs:
        file_info = information, such as paths to directories

        run_info = information on flag settings, etc. for this run.  Includes
            year, number, distance, station, uselogy, etc.
            
        random_state = train_test_split parameter, controls the shuffling applied 
            to the data before applying the split. Pass an int for reproducible 
            output across multiple function calls
        
        test_size = train_test_split parameter, if float, should be between 0.0 
            and 1.0 and represent the proportion of the dataset to include in 
            the test split. If int, represents the absolute number of test samples.
            
        drop_large = drop excessively large B_H values (i.e., > 10,000 nT)
            
     Outputs:
        train_set and test_set = training and test dataframes
        
        scaler = Standard scalar used to scale data, can be used to reverse 
            scaling
    """
    
    # Verify we are wanting all data
    if run_info['info'] != 'single': 
       import sys
       sys.exit('Error: run_info is not for one station, one year.')

    # Merge the SuperMAG and OMNI files into a single dataframe
    df = _merge_files( file_info, run_info, drop_large=drop_large ) 
        
    # Skip files with less than 50 data points
    if len(df) < 50:
        return None, None
                   
    # Split data into training and testing sets.  With random_state set to an
    # integer, its repeatable.
    train_set, test_set = train_test_split(df, random_state=random_state, 
                                           test_size=test_size)
    
    # If standardize is True, standardize the data ==> data = (data-mean)/(std dev)
    #
    # The recommended way (see 'Elements of Statistical Learning', chapter 'The 
    # Wrong and Right Way to Do Cross-validation') is to calculate the mean and 
    # the standard deviation of the values in the training set and then apply them 
    # for standardizing both the training and testing sets.
    #
    # The idea behind this is to prevent data leakage from the testing to the 
    # training set because the aim of model validation is to subject the testing 
    # data to the same conditions as the data used for the model training.
    #
    # https://datascience.stackexchange.com/questions/63717/how-to-use-standardization-standardscaler-for-train-and-test
    if run_info['standardize']:
        scaler = StandardScaler()
        scaler.set_output(transform="pandas")
        train_set = scaler.fit_transform(train_set)
        test_set = scaler.transform (test_set)
    else:
        scaler = None
        
    return train_set, test_set, scaler

def get_data_all( file_info, run_info, random_state=42, test_size=0.2, drop_large=True ):
    """Merges the SuperMAG and OMNI data, processes them based on the options, 
    and returns training and test dataframes with B_H Mean as the response variable.
    In this version, we collect all years and all stations together into one
    dataframe
    
    Inputs:
        file_info = information, such as paths to directories

        run_info = information on flag settings, etc. for this run.  Includes
            number, distance, uselogy, etc.  
            
        random_state = train_test_split parameter, controls the shuffling applied 
            to the data before applying the split. Pass an int for reproducible 
            output across multiple function calls
        
        test_size = train_test_split parameter, if float, should be between 0.0 
            and 1.0 and represent the proportion of the dataset to include in 
            the test split. If int, represents the absolute number of test samples.
            
        drop_large = drop excessively large B_H values (i.e., > 10,000 nT)

        Outputs:
        train_set and test_set = training and test dataframes
        
        scaler = Standard scalar used to scale data, can be used to reverse 
            scaling
    """
    
    # Verify we are wanting all data
    if run_info['info'] != 'all': 
       import sys
       sys.exit('Error: run_info is not for all stations, all years.')
    
    # Loop through all the years and the stations to combine all the data into
    # a single dataframe
    
    # Initialize list of dataframes that we will combine
    df_list = []
    
    # We will need to add year and station to flag_info to create 
    # run_info like dict
    run_info_tmp = run_info
    
    for yr in file_info['years']:
            
        run_info_tmp['year'] = yr
        stations = stations_list( yr, file_info['SuperMAG Directory'] )
        
        for stat in stations:
            # print(yr, stat)
            run_info_tmp['station'] = stat
            
            # Merge the SuperMAG and OMNI files into a single dataframe
            tmp = _merge_files( file_info, run_info_tmp, drop_large=drop_large ) 
            
            # Skip files with less than 50 data points
            if len(tmp) >= 50:
                # Add a column with the station in it
                # We can use it in the fit
                tmp['station'] = np.full(shape=(len(tmp),), fill_value=stat)
                # Add to dataframe list
                df_list.append(tmp)

    # Combine all the dataframes
    if len(df_list) > 0:
        df = pd.concat(df_list)
        df = df.reset_index(drop=True)
    else:
        import sys
        sys.exit('Error: No data found in get_data_all.')

    # Split data into training and testing sets.  With random_state set to an
    # integer, its repeatable.
    train_set, test_set = train_test_split(df, random_state=random_state, 
                                           test_size=test_size)
    
    # If standardize is True, standardize the data ==> data = (data-mean)/(std dev)
    #
    # The recommended way (see 'Elements of Statistical Learning', chapter 'The 
    # Wrong and Right Way to Do Cross-validation') is to calculate the mean and 
    # the standard deviation of the values in the training set and then apply them 
    # for standardizing both the training and testing sets.
    #
    # The idea behind this is to prevent data leakage from the testing to the 
    # training set because the aim of model validation is to subject the testing 
    # data to the same conditions as the data used for the model training.
    #
    # https://datascience.stackexchange.com/questions/63717/how-to-use-standardization-standardscaler-for-train-and-test
    if run_info['standardize']:
        scaler = StandardScaler()
        scaler.set_output(transform="pandas")
        
        # Skip categorical variable station if its in dataframe
        # station is the last column
        names = list(train_set.columns)
        if 'station' in names: 
            names.remove('station')
        
        train_set[names] = scaler.fit_transform(train_set[names])
        test_set[names] = scaler.transform(test_set[names])
    else:
        scaler = None
        
    return train_set, test_set, scaler
 
def set_plot_rcParams( fontsize=12, figsize=None ):
    """Set matplotlib rcParams
    
    Inputs:
        fontsize = default font size for text, specified in points
        
        figsize = figure size in inches
        
     Outputs:
        None beyond seting rcParams
    """
    if figsize is None: 
        figsize = [6,6]
    
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["figure.dpi"] = 600
    plt.rcParams['axes.grid'] = True
    plt.rcParams['font.size'] = fontsize
    plt.rcParams.update({
        # "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
    })
    return
 
def get_prefix(run_info):
    """Determine prefix based on fit options.  Used to determine filenames,
    plot titles, etc.
    
    Inputs:
        run_info = information on flag settings, etc. for this run.  Includes
            year, number, distance, station, uselogy, etc.
        
     Outputs:
        prefix = a string
    """
    
    # Change titles and filenames based on options
    # Prefix added to beginning of titles and filenames
    prefix = ''
    
    if run_info['standardize']: prefix = prefix + 'Standardized '
    
    if run_info['uselogy']:    
        prefix = prefix + 'LogBH ' 
    else:
        prefix = prefix + 'BH ' 
             
    if run_info['Kp Lower'] is not None:            
        prefix = prefix + r'Kp$\geq$' + str(run_info['Kp Lower']) + ' '   
    if run_info['Kp Upper'] is not None:            
        prefix = prefix + r'Kp$\leq$' + str(run_info['Kp Upper']) + ' '   
    
    return prefix
 
def get_suffix( run_info, base='Autogluon' ):
    """Determine suffix based on fit options.  Used to determine directory
    names, etc.
    
    Inputs:
        run_info = information on flag settings, etc. for this run.  Includes
            year, number, distance, station, uselogy, etc.
        
        base = base of directory name
        
     Outputs:
        suffix = a string
    """

    # Used to determine directory names
    suffix = base
    
    if run_info['standardize']: suffix = suffix + '_Standardize'
    
    if run_info['uselogy']:    
        suffix = suffix + '_LogBH'
    else:
        suffix = suffix + '_BH'

    if run_info['Kp Lower'] is not None:            
        suffix = suffix + '_KpLower' + str(run_info['Kp Lower'])  
    if run_info['Kp Upper'] is not None:            
        suffix = suffix + '_KpUpper' + str(run_info['Kp Upper'])  

    return suffix

def stations_list( year, smdirectory ):
    """Provides an array of SuperMAG stations that match a SPUD station.  In 
    addition to matching a SPUD station, we also require we have SuperMAG data
    in year. download_SuperMAG.py generated the list of stations.  The list 
    varies by year.   
    
    Inputs:
        year = all stations for specified year are provided
        
        smdirectory = directory where SuperMAG pickle files are stored
        
    Outputs:
        stations = array of stations for year
    """
    
    # List of SuperMAG stations that correspond to a SPUD station
    # for which we also have SuperMAG data in year
    
    file = 'stations-' + str(year) + '.pkl'
    stats = pd.read_pickle( join(smdirectory, file) )
    stations = stats['Stations'].values
    
    return stations

def nse(test, predict):
    """Calculates Nash Sutcliffe efficency (aka prediction efficiency)
    
    Inputs:
        test = numpy array of test observations
        
        predict = numpy array of model predictions
               
     Outputs:
        nse = Nash Sutcliffe efficiency, aka prediction efficiency
    """
    
    eff = 1 - (np.sum((test - predict) ** 2) / np.sum((test - np.mean(test)) ** 2))

    return eff


if __name__ == "__main__":
    
    stats_list = stations_list( 2025, '/SuperMAG/' )
    
    print(stats_list)
    print(type(stats_list))
