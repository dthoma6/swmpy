#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 17:05:32 2026

@author: Dean Thomas
"""

from os.path import join
import pandas as pd
import numpy as np
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
    df = df.drop(['By, nT (GSM) STD'], axis=1)
    df = df.drop(['Bz, nT (GSM) Mean'], axis=1)
    df = df.drop(['Bz, nT (GSM) STD'], axis=1)
    
    # Drop derived variables, which are based on measured data
    # These have high correlation to measured data
    #
    # Derived variables described in OMNI documentation at 
    # https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/modified/hro_modified_format.txt
    
    # Flow pressure 
    df = df.drop(['Flow pressure, nPa Mean'], axis=1)
    df = df.drop(['Flow pressure, nPa STD'], axis=1)

    # Electric field 
    df = df.drop(['Electric field, mV/m Mean'], axis=1)
    df = df.drop(['Electric field, mV/m STD'], axis=1)

    # Plasma beta
    df = df.drop(['Plasma beta Mean'], axis=1)
    df = df.drop(['Plasma beta STD'], axis=1)

    # Alfven mach 
    df = df.drop(['Alfven mach number Mean'], axis=1)
    df = df.drop(['Alfven mach number STD'], axis=1)

    # Magnetosonic mach
    df = df.drop(['Magnetosonic mach number Mean'], axis=1)
    df = df.drop(['Magnetosonic mach number STD'], axis=1)

    # Drop other variables that are highly correlated, see scatter_matrix.py

    # Vx correlated with |V|
    df = df.drop(['Vx Velocity, km/s, GSE Mean'], axis=1)
    df = df.drop(['Vx Velocity, km/s, GSE STD'], axis=1)
   
    # Bx and By correlated
    df = df.drop(['By, nT (GSE) Mean'], axis=1)
    df = df.drop(['By, nT (GSE) STD'], axis=1)

    # Temperature correlated with |V| and Vx
    df = df.drop(['Temperature, K Mean'], axis=1)
    df = df.drop(['Temperature, K STD'], axis=1)
    
    return df

def _remove_correlated_supermag( df ):
    """Drop dataframe columns that contain highly-correlated 
    SuperMAG variables. |B| Mean and STD are correlated with B_H

    Inputs:
        df = dataframe that includes OMNI and SuperMAG 
        
    Outputs:
        df = dataframe without correlated SuperMAG variables
    """

    # |B| Mean and STD are correlated with B_H
    df = df.drop(['B_mag Mean'], axis=1)
    df = df.drop(['B_mag STD'], axis=1)
    
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
 
def _remove_pos_neg( df ):
    """Drops variables from dataframe that are both positive and negative,
    we can't use them when we fit with the log(variables).  
    
    Inputs:
        df = dataframe that includes OMNI and SuperMAG data
        
    Outputs:
        dataframe without the pos/neg columns
    """
    
    # Drop these variables  
    df = df.drop(['Bx, nT (GSE, GSM) STD'], axis=1)
    df = df.drop(['Bz, nT (GSE) STD'], axis=1)
    df = df.drop(['Bx, nT (GSE, GSM) Mean'], axis=1)
    df = df.drop(['Bz, nT (GSE) Mean'], axis=1)
    df = df.drop(['Vy Velocity, km/s, GSE STD'], axis=1)
    df = df.drop(['Vz Velocity, km/s, GSE STD'], axis=1)
    df = df.drop(['Vy Velocity, km/s, GSE Mean'], axis=1)
    df = df.drop(['Vz Velocity, km/s, GSE Mean'], axis=1)

    return df
 
def _remove_dXdt( df ):
    """Drops dX/dt variables from dataframe when we won't use them in fits.  
    
    Inputs:
        df = dataframe that includes OMNI and SuperMAG data
        
    Outputs:
        dataframe without the dXdt columns
    """
    
    # Drop these variables  
    df = df.drop(['d|B|/dt Mean'], axis=1)
    df = df.drop(['d|B|/dt STD'], axis=1)
    df = df.drop(['d|V|/dt Mean'], axis=1)
    df = df.drop(['d|V|/dt STD'], axis=1)
    df = df.drop(['dn/dt Mean'], axis=1)
    df = df.drop(['dn/dt STD'], axis=1)

    return df
 
def _remove_zeros_add_logs( df, uselog, includedXdt ):
    """Drops variables that equal 0, and if uselog is true, take the log of 
    the variables.  
    
    Inputs:
        df = dataframe that includes OMNI and SuperMAG data
        
        uselog = whether to use log(variables) in fit
        
        includedXdt = whether to include time derivative of variables
        
    Outputs:
        dataframe without the 0 values
    """
    
    if uselog:
        names = ['|B| Mean', '|B| STD', '|V| Mean', '|V| STD',
                'Proton Density, n/cc Mean', 'Proton Density, n/cc STD',
                'B_H Mean', 'B_H STD']
        
        # Drop rows with zero values so we can take log of them
        for name in names:
            df = df.drop(df[df[name] == 0].index)
            df[name] = np.log(df[name])
    else:
        if includedXdt:
            names = ['|B| Mean', '|B| STD', 'Bx, nT (GSE, GSM) Mean', 'Bz, nT (GSE) Mean',
                    'Bx, nT (GSE, GSM) STD', 'Bz, nT (GSE) STD', '|V| Mean', '|V| STD',
                    'Vy Velocity, km/s, GSE Mean', 'Vz Velocity, km/s, GSE Mean',
                    'Vy Velocity, km/s, GSE STD', 'Vz Velocity, km/s, GSE STD',
                    'Proton Density, n/cc Mean', 'Proton Density, n/cc STD',
                    'd|B|/dt Mean', 'd|B|/dt STD', 'd|V|/dt Mean', 'd|V|/dt STD', 
                    'dn/dt Mean', 'dn/dt STD', 'B_H Mean', 'B_H STD']
        else:
            names = ['|B| Mean', '|B| STD', 'Bx, nT (GSE, GSM) Mean', 'Bz, nT (GSE) Mean',
                    'Bx, nT (GSE, GSM) STD', 'Bz, nT (GSE) STD', '|V| Mean', '|V| STD',
                    'Vy Velocity, km/s, GSE Mean', 'Vz Velocity, km/s, GSE Mean',
                    'Vy Velocity, km/s, GSE STD', 'Vz Velocity, km/s, GSE STD',
                    'Proton Density, n/cc Mean', 'Proton Density, n/cc STD',
                    'B_H Mean', 'B_H STD']

        # Drop rows with zero values (I believe they're bad data, and 
        # we have a small number of them)
        for name in names:
            df = df.drop(df[df[name] == 0].index)

    return df

def _remove_kp( df, kp ):
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
 
def _merge_files( file_info, run_info ):
    """Combines the SuperMAG and OMNI data for the given year, number, distance, 
    station, and Kp into a single dataframe
    
    Inputs:
        file_info = information, such as paths to directories

        run_info = information on flag settings, etc. for this run.  Includes
            year, number, distance, station, uselog, etc.
        
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
    kp       = run_info['Kp']
    
    uselog      = run_info['uselog']
    includedXdt = run_info['includedXdt']
                 
    # Get dataframes and merge them
    filename = 'OMNI-stats-' + str(distance) + 'Re-' + str(number) + 'min-' + str(year) + '.pkl'
    omnidf = pd.read_pickle( join(omnidirectory, filename) )  
    omnidf.set_index('Datetime')

    filename = station + '-stats-' + str(number) + 'min-' + str(year) + '.pkl'
    smdf = pd.read_pickle( join(supermagdirectory, filename) )
    smdf.set_index('Datetime')
    
    filename = 'Kp-stats-' + str(number) + 'min-' + str(year) + '.pkl'
    kpdf = pd.read_pickle( join(kpdirectory, filename) )
    kpdf.set_index('Datetime')

    df = pd.merge(omnidf, smdf, left_index=True, right_index=True)
    df = pd.merge(df,     kpdf, left_index=True, right_index=True)

    # Drop highly-correlated OMNI variables
    df = _remove_correlated_omni(df)
    
    # Drop highly-correlated SuperMAG variables
    df = _remove_correlated_supermag(df)
        
    # Special test for uselog and includedXdt
    # Can't have both true because dXdt variables are positive and negative
    # so we can't take logs of them.
    if uselog and includedXdt: 
        import sys
        sys.exit('Error: Either uselog True or includedXdt True, but not both.')
    
    # If we fit to log(variables) (uselog is True),
    # drop variables positive and negative variables
    # and we can't take log of them
    if uselog:
        df = _remove_pos_neg( df )
   
    # if includedXdt is false or uselog is True drop these variables
    # We're either not using them (includedXdt False) or they are positive and 
    # negative and we can't take the log of them (uselog is True)
    # Note, don't need check on uselog here, since uselog can only be true
    # if includedXdt is false.
    if not includedXdt:
        df = _remove_dXdt( df )

    # Drop rows with 0 values and, if uselog is true, take log(variables)
    df = _remove_zeros_add_logs( df, uselog, includedXdt )
    
    # if kp != None, drop rows with Kp values below kp
    if kp is not None: 
        df = _remove_kp( df, kp )
    
    # Add circular response for mlt. Not done earlier in case uselog is true.  
    # We don't want the log of the sine and cosine.
    df['cosmlt'] = np.cos( df.mlt*np.pi/180. )
    df['sinmlt'] = np.sin( df.mlt*np.pi/180. )
    
    # Add circular response for mcolat if we use get_data_all. 
    # aka run_info["info"] == 'all'. Not done earlier in case uselog is true.  
    if run_info["info"] == 'all':
        df['cosmcolat'] = np.cos( df.mcolat*np.pi/180. )
        df['sinmcolat'] = np.sin( df.mcolat*np.pi/180. )
    
    # Drop variables that we won't use them in fit     
    df = _remove_unused( df )
    
    return df

def _remove_std( df ):
    """Removes STD variables from a dataframe.  The input dataframe contains a 
    collection of variables including the Mean value and Std Dev of the variables.  
    The STD for a given variable in in 'variable STD' column and the Mean is 
    in the 'variable Mean' column. 
    
    Inputs:
        df = contains the Means and STDs (std dev)
        
    Outputs:
        dataframe without the STD columns
    """
    
    columns = df.columns
    for col in columns:
        if 'STD' in col:
            df = df.drop([col], axis=1)

    return df
    
def _remove_mean( df ):
    """Removes Mean variables from a dataframe.  The input dataframe contains 
    a collection of variables including the Mean value and Std Dev of the 
    variables.  The STD for a given variable in in 'variable STD' column and
    the Mean is in the 'variable Mean' column. 
    
    Inputs:
        df = contains the Means and STDs (std dev)
        
    Outputs:
        dataframe without the Mean columns
    """
    
    columns = df.columns
    for col in columns:
        if 'Mean' in col:
            df = df.drop([col], axis=1)

    return df

def get_data_one( file_info, run_info, random_state=42, test_size=0.2 ):
    """Merges the SuperMAG and OMNI data, processes them based on the options, 
    and returns training and test dataframes with B_H Mean as the response variable.
    In this version, the dataframe is for a specific station during a specific
    year.
    
    Inputs:
        file_info = information, such as paths to directories

        run_info = information on flag settings, etc. for this run.  Includes
            year, number, distance, station, uselog, etc.
            
        random_state = train_test_split parameter, controls the shuffling applied 
            to the data before applying the split. Pass an int for reproducible 
            output across multiple function calls
        
        test_size = train_test_split parameter, if float, should be between 0.0 
            and 1.0 and represent the proportion of the dataset to include in 
            the test split. If int, represents the absolute number of test samples.
            
     Outputs:
        train_set and test_set = training and test dataframes
        
        scaler = Standard scalar used to scale data, can be used to reverse 
            scaling
    """
    
    if run_info['uselog'] and run_info['uselogbh']: 
        import sys
        sys.exit('Error: Either USELOG True or USELOGBH True, but not both.')

    # Merge the SuperMAG and OMNI files into a single dataframe
    df = _merge_files( file_info, run_info ) 
    
    # # Skip files with less than 50% of measurements per year
    # # 0.5 * 365 days * 24 hours * 60 min = 262800 obsservations
    # # we combine "number" observations to get mean and std dev.
    # if len(df) < 262800/run_info['number']:
    #     return None, None
    
    # Skip files with less than 50 data points
    if len(df) < 50:
        return None, None
                   
    # Determine whether we fit on Mean only or Mean and STD variables
    # Regardless, the response variable is B_H Mean, so we drop B_H STD
    if run_info['includestd']:
        df = df.drop(['B_H STD'], axis=1)
    else:
        df = _remove_std(df)

    if run_info['uselogbh']:
        tmp = np.log( df['B_H Mean'] )
        df['B_H Mean'] = tmp

    # Determine whether the fit uses squares of some variables 
    if run_info['addsquares'] is not None:
        for name in run_info['addsquares']:
            df[name + ' Squared'] = df[name] * df[name]

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
        
    return train_set, test_set, scaler

def get_data_all( file_info, flag_info, random_state=42, test_size=0.2 ):
    """Merges the SuperMAG and OMNI data, processes them based on the options, 
    and returns training and test dataframes with B_H Mean as the response variable.
    In this version, we collect all years and all stations together into one
    dataframe
    
    Inputs:
        file_info = information, such as paths to directories

        flag_info = information on flag settings, etc. for this run.  Includes
            number, distance, uselog, etc.  This is distinct from the run_info
            in get_data.  flag_info does not have the year or the station.
            
        random_state = train_test_split parameter, controls the shuffling applied 
            to the data before applying the split. Pass an int for reproducible 
            output across multiple function calls
        
        test_size = train_test_split parameter, if float, should be between 0.0 
            and 1.0 and represent the proportion of the dataset to include in 
            the test split. If int, represents the absolute number of test samples.
            
     Outputs:
        train_set and test_set = training and test dataframes
        
        scaler = Standard scalar used to scale data, can be used to reverse 
            scaling
    """
    
    if flag_info['uselog'] and flag_info['uselogbh']: 
        import sys
        sys.exit('Error: Either USELOG True or USELOGBH True, but not both.')

    if flag_info['Kp'] is not None and flag_info['|B| threshold'] is not None: 
        import sys
        sys.exit('Error: Either KP or BTHRESHOLD is not None, but not both.')

    # Loop through all the years and the stations to combine all the data into
    # a single dataframe
    
    # Initialize list of dataframes that we will combine
    df_list = []
    
    # We will need to add year and station to flag_info to create 
    # run_info like dict
    run_info = flag_info
    
    for yr in file_info['years']:
            
        run_info['year'] = yr
        stations = stations_list( yr, file_info['SuperMAG Directory'] )
        
        for stat in stations:
            # print(yr, stat)
            run_info['station'] = stat
            
            # Merge the SuperMAG and OMNI files into a single dataframe
            tmp = _merge_files( file_info, run_info ) 
            
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

    # Determine whether we fit on Mean only or Mean and STD variables
    # Regardless, the response variable is B_H Mean, so we drop B_H STD
    if flag_info['includestd']:
        df = df.drop(['B_H STD'], axis=1)
    else:
        df = _remove_std(df)

    if flag_info['uselogbh']:
        tmp = np.log( df['B_H Mean'] )
        df['B_H Mean'] = tmp

    # Determine whether the fit uses squares of some variables 
    if flag_info['addsquares'] is not None:
        for name in flag_info['addsquares']:
            df[name + ' Squared'] = df[name] * df[name]
            
    # Determine whether |B| threshold is specified.  If so, drop rows with 
    # |B| less than that specified.  Threshold is num. of std deviations.
    # e.g., Drop everything below num. std deviations above the mean.
    # We use this to isolate rows with strong storm conditions
    if flag_info['|B| threshold'] is not None:
        mean = df['B_H Mean'].mean()
        std  = df['B_H Mean'].std()
        num  = flag_info['|B| threshold']
        df = df.drop(df[df['B_H Mean'] < (mean + std*num)].index)
        
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
    if flag_info['standardize']:
        scaler = StandardScaler()
        scaler.set_output(transform="pandas")
        
        # Skip categorical variable station if its in dataframe
        # station is the last column
        names = list(train_set.columns)
        if 'station' in names: 
            names.remove('station')
        
        train_set[names] = scaler.fit_transform(train_set[names])
        test_set[names] = scaler.transform(test_set[names])
        
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
            year, number, distance, station, uselog, etc.
        
     Outputs:
        prefix = a string
    """
    
    # Change titles and filenames based on options
    # Prefix added to beginning of titles and filenames
    prefix = ''
    
    if run_info['standardize']: prefix = prefix + 'Standardized '
    if run_info['includedXdt']: prefix = prefix + 'dXdt '
    if run_info['includestd']:  prefix = prefix + 'STD '
    if run_info['uselog']:      prefix = prefix + 'Log '
    if run_info['uselogbh']:    prefix = prefix + 'LogBH '    
    if run_info['addsquares'] is not None:    prefix = prefix + 'AddSqs '
    if run_info['Kp'] is not None:            
        prefix = prefix + 'Kp' + str(run_info['Kp']) + ' '   
    if run_info['|B| threshold'] is not None: 
        prefix = prefix + 'BThres' + str(run_info['|B| threshold']) + ' '   
    
    return prefix
 
def get_suffix( run_info, base='Autogluon' ):
    """Determine suffix based on fit options.  Used to determine directory
    names, etc.
    
    Inputs:
        run_info = information on flag settings, etc. for this run.  Includes
            year, number, distance, station, uselog, etc.
        
        base = base of directory name
        
     Outputs:
        suffix = a string
    """

    # Used to determine directory names
    suffix = base
    if run_info['standardize']: suffix = suffix + '_Standardize'
    if run_info['includedXdt']: suffix = suffix + '_dXdt '
    if run_info['includestd']:  suffix = suffix + '_STD'
    if run_info['uselog']:      suffix = suffix + '_Log'
    if run_info['uselogbh']:    suffix = suffix + '_LogBH'
    if run_info['addsquares'] is not None:    suffix = suffix + '_AddSqs'
    if run_info['Kp'] is not None:            
        suffix = suffix + '_Kp' + str(run_info['Kp'])  
    if run_info['|B| threshold'] is not None: 
        suffix = suffix + '_B_Thres' + str(run_info['|B| threshold']) 

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

def pearson_cc( x, y ):
    """Calculates the Pearson correlation coefficient for x and y
    
    Inputs:
        x = numpy array of x values
        
        y = numpy array of y values
               
     Outputs:
        pcc = Pearson correlation coefficient 
    """
    xmean = np.mean(x)
    ymean = np.mean(y)
    
    top = np.sum((x-xmean)*(y-ymean))
    bot = np.sqrt( np.sum((x-xmean)**2) )*np.sqrt( np.sum((y-ymean)**2) )
    
    pcc = top/bot
    
    return pcc

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