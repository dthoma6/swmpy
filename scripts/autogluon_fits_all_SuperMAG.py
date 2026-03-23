#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 12:42:43 2026

@author: Dean Thomas
"""
import swmpy as swm

if __name__ == "__main__":
    
    #########################################################################
    # Generate autogluon fits across all combinations of number and distance 
    # parameters specified in file info. Fits are across all SuperMAG sites.
    #########################################################################

    # Whether to generate fits
    FITS = True
    
    # Whether to generate plots
    PLOTS = True
    
    # Whether to generate regression fit or quantile fit
    REGRESSION = True
    
    #########################################################################
    # Generic information, such as paths to directories, for multiple runs
    #########################################################################
    
    from file_info import file_info

    #########################################################################
    # Specific parameters for the this run
    #########################################################################
    
    # "info" ==> "all" for all years and all stations, or "single" for one year, one station
    # "uselogy" ==> True/False, use log of y for fit
    # "standardize" ==> True/False, standardize data => (data-mean)/stddev
    # "number" ==> number of minutes over which OMNI and SuperMAG data average, 
    #               None if we want raw 1-minute cadence data
    # "distance" ==> distance from Earth in Re that we ballistic propagate OMNI 
    #               data to, None if we want raw OMNI data
    # "Kp Lower" ==> keep OMNI/SuperMAG data for events with Kp above this, or 
    #               None if keep all data
    # "Kp Upper" ==> keep OMNI/SuperMAG data for events with Kp below this, or 
    #               None if keep all data

    def get_run_info_all( number, distance ): 
        
        if number is None and distance is not None: 
            import sys
            sys.exit('Error: If number is None, distance must be None.')

        run_info = {
            "info": 'all',  # all years, all stations
            "uselogy": True,
            "standardize": False,
            "number": number,
            "distance": distance,
            "Kp Lower": 7.0,
            "Kp Upper": None,
            }
        
        return run_info

    ################# autogluon fit

    if FITS:
        for num in file_info['numbers']:
           for dist in file_info['distances']:
               
               # if num is None, dist must also be None
               if num is None and dist is not None:
                   break
               
               # Update run_info
               run_info = get_run_info_all( num, dist )

               # Do modeling
               if REGRESSION:
                   swm.autogluon_regression(file_info, run_info, full=True)
               else:
                   swm.autogluon_quantile(file_info, run_info, full=True)

    ################# autogluon plots
    
    if PLOTS:
        for num in file_info['numbers']:
            for dist in file_info['distances']:

                # if num is None, dist must also be None
                if num is None and dist is not None:
                    break
              
                # Update run_info
                run_info = get_run_info_all( num, dist )

                if REGRESSION:
                    # Plot predicted vs measured results from previously 
                    # fit autogluon model
                    swm.autogluon_predict_measured_plot(file_info, run_info,
                                             full=True)
                    
                    
                    # Plot residuals vs prediction plots based on results from 
                    # previously fit autogluon model
                    swm.autogluon_residuals_predict_plot(file_info, run_info, 
                                                         full=True)

                    # Plot QQ plots based on results from previously fit 
                    # autogluon model
                    swm.autogluon_qq_plot(file_info, run_info, full=True)

                    # Plot premutation importance based on results from 
                    # previously fit autogluon model
                    swm.autogluon_permutation_plot(file_info, run_info,
                                                   quantile=False, full=True)
                else:
                    # # Plot premutation importance based on results from 
                    # # previously fit autogluon model
                    swm.autogluon_permutation_plot(file_info, run_info,
                                                   quantile=True, full=True)

                    # Plot quantiles to show model uncertainty
                    swm.autogluon_quantile_plot(file_info, run_info, 
                                                zoom=None, full=True)
                    

