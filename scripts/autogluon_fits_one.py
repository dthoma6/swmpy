#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 12:42:43 2026

@author: Dean Thomas
"""
import swmpy as swm
from file_info import file_info, data_dir

if __name__ == "__main__":
    
    # Use logs of variables for fit
    USELOG = False
    USELOGBH = False 
    if USELOG and USELOGBH: 
        import sys
        sys.exit('Error: Either USELOG True or USELOGBH True, but not both.')
    
    # Include dB/dt, dV/dt, dn/dt in fit
    INCLUDEDXDT = False
    if USELOG and INCLUDEDXDT: 
        import sys
        sys.exit('Error: Either USELOG True or INCLUDEDXDT True, but not both.')
   
    # Include STD (std deviation) variables in fit
    INCLUDESTD = False
    
    # Kp threshold, we keep only data with Kp above this.
    # Use None, if we want to keep all data
    KP = None
    
    # |B| threshold, we keep on data with "|B| mean" above the threshold
    # defined by number of std deviations.  Drop data with less than
    # mean + num. of std deviations. Use None, if we want to keep all data
    #
    # Didn't work as well as Kp cut-off
    #
    BTHRESHOLD = None  

    # Include list of variables that we want to square in the fit
    # e.g., we want '|V| Mean' and '|V| Mean Square' in the fit, add
    # '|V| Mean' to list.  Otherwise, set to None
    ADDSQUARES = None

    # Should data be standardized, data ==> data = (data-mean)/(std dev)
    STANDARDIZE = True

    # Whether to generate regression fit or quantile fit
    REGRESSION = False
    
    # Whether to generate fits
    FITS = True
    
    # Whether to generate plots
    PLOTS = True
    
    # Whether to copy plots to single directory
    COPY = False
    
    # Get run_info on flags and other run parameters
    # Note, there are a few differences between run_info for all data
    # and run_info for single station in a single year.  Single stations and
    # single year parameters are included in that run_info
    def get_run_info_one( station, year, number, distance ): 
        
        run_info = {
            "info": "single",  # one year, one station
            "uselog": USELOG,
            "uselogbh": USELOGBH,
            "standardize": STANDARDIZE,
            "includedXdt": INCLUDEDXDT,
            "includestd": INCLUDESTD,
            "station": station,
            "year": year,
            "number": number,
            "distance": distance,
            "Kp": KP,
            "|B| threshold": BTHRESHOLD,
            "addsquares": None
            }
        
        return run_info

                    
    ################# autogluon fit

    if FITS:
        for yr in file_info['years']:
            # get list of stations
            # stats = swm.stations_list(yr, file_info['SuperMAG Directory'])
            stats = ['MEA',]
    
            for num in file_info['numbers']:
                for dist in file_info['distances']:
                    for stat in stats:
                        print( '######### ', stat, yr, dist, num, ' #########' )
                        
                        run_info = get_run_info_one( stat, yr, num, dist )
    
                        # Do modeling
                        if REGRESSION:
                            swm.autogluon_regression(file_info, run_info)
                        else:
                            swm.autogluon_quantile(file_info, run_info)

    ################# autogluon plots
    
    if PLOTS:
        for yr in file_info['years']:
            # get list of stations
            # stats = swm.stations_list(yr, file_info['SuperMAG Directory'])
            stats = ['MEA',]
    
            for num in file_info['numbers']:
                for dist in file_info['distances']:
                    for stat in stats:
                        print( '######### ', stat, yr, dist, num, ' #########' )
    
                        run_info = get_run_info_one( stat, yr, num, dist )
    
                        if REGRESSION:
                            # Plot predicted vs measured results from previously 
                            # fit autogluon model
                            swm.autogluon_predict_measured_plot(file_info, run_info, 
                                                                full=False)
                            
                            
                            # Plot residuals vs prediction plots based on results from 
                            # previously fit autogluon model
                            swm.autogluon_residuals_predict_plot(file_info, run_info, 
                                                                 full=False)
        
                            # Plot QQ plots based on results from previously fit 
                            # autogluon model
                            swm.autogluon_qq_plot(file_info, run_info, full=False)
        
                            # Plot premutation importance based on results from 
                            # previously fit autogluon model
                            swm.autogluon_permutation_plot(file_info, run_info,
                                                           quantile=False, full=False)
                        else:
                            # Plot premutation importance based on results from 
                            # previously fit autogluon model
                            swm.autogluon_permutation_plot(file_info, run_info,
                                                           quantile=True)

                            # Plot quantiles to show model uncertainty
                            swm.autogluon_quantile_plot(file_info, run_info)

    ################# autogluon copy plots into one directory
    
    if COPY:
        from os import makedirs, rename
        from os.path import join, exists
        from shutil import copy
                
        destination = join( data_dir, "All_Plots")
        if not exists( destination ):
            makedirs( destination )
        cnt = 0
        
        for yr in file_info['years']:
            stats = swm.stations_list(yr, file_info['SuperMAG Directory'])
    
            for num in file_info['numbers']:
                for dist in file_info['distances']:
                    for stat in stats:
    
                        run_info = get_run_info_one( stat, yr, num, dist )
                        # Directory where we will store plots
                        path = swm.dir_path( file_info, run_info, quantile=not REGRESSION )
                        
                        if REGRESSION:
                            prefix = swm.get_prefix(run_info)
                            srcpath = join( path, 'Fit vs Measured ' + prefix.strip() +
                                           '.png')
                            if exists( srcpath ):
                                # Quantiles plots
                                dstpath = copy(srcpath, destination)
                                newpath = join(destination, 'Fit vs Measured ' + 
                                               prefix.strip() + '_' + str(cnt) +'.png')
                                rename(dstpath, newpath)

                            srcpath = join( path, 'QQ Plot ' + prefix.strip() +
                                           '.png')
                            if exists( srcpath ):
                                # Quantiles plots
                                dstpath = copy(srcpath, destination)
                                newpath = join(destination, 'QQ Plot ' + 
                                               prefix.strip() + '_' + str(cnt) +'.png')
                                rename(dstpath, newpath)

                            srcpath = join( path, 'Features ' + prefix.strip() + '.png')
                            if exists( srcpath ):                                
                                # Features plots
                                dstpath = copy(srcpath, destination)
                                newpath = join(destination, 'Features ' + 
                                               prefix.strip() + '_' + str(cnt) +'.png')
                                rename(dstpath, newpath)
                        else:
                            prefix = swm.get_prefix(run_info)
                            srcpath = join( path, 'Quantile ' + prefix.strip() +
                                           '.png')
                            if exists( srcpath ):
                                # Quantiles plots
                                dstpath = copy(srcpath, destination)
                                newpath = join(destination, 'Quantile ' + 
                                               prefix.strip() + '_' + str(cnt) +'.png')
                                rename(dstpath, newpath)

                            srcpath = join( path, 'Features ' + prefix.strip() + '.png')
                            if exists( srcpath ):                                
                                # Features plots
                                dstpath = copy(srcpath, destination)
                                newpath = join(destination, 'Features ' + 
                                               prefix.strip() + '_' + str(cnt) +'.png')
                                rename(dstpath, newpath)
                                
                        cnt += 1
