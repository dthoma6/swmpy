#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 12:42:43 2026

@author: Dean Thomas
"""
import swmpy as swm
from file_info_all_SuperMAG import file_info

if __name__ == "__main__":
    
    # Whether to fit B_H (True) or dB_H/dt (False) as dependent variable
    USEBH = True
    
    # Use logs of variables for fit
    # USELOG True, log of all independent and dependent variables
    # USELOGY True, log of dependent variable only
    USELOG = False
    USELOGY = True 
    if USELOG and USELOGY: 
        import sys
        sys.exit('Error: Either USELOG True or USELOGY True, but not both.')
    
    # Include dB/dt, dV/dt, dn/dt in fit
    # Since dX/dt variables are positive and negative, can't use with USELOG
    INCLUDEDXDT = False
    if USELOG and INCLUDEDXDT: 
        import sys
        sys.exit('Error: Either USELOG True or INCLUDEDXDT True, but not both.')
    
    # Kp threshold, we keep only data with Kp above this.
    # Use None, if we want to keep all data
    KP = 7.0
    
    # |B| threshold, we keep on data with "|B| mean" (OMNI) above the threshold
    # defined by number of std deviations.  Drop data with less than
    # mean + num. of std deviations. Use None, if we want to keep all data
    #
    # Doesn't work as well as the Kp threshold
    BTHRESHOLD = None 
    
    if KP is not None and BTHRESHOLD is not None: 
        import sys
        sys.exit('Error: Either KP or BTHRESHOLD is not None, but not both.')

    # Include list of variables that we want to square in the fit
    # e.g., we want '|V| Mean' and '|V| Mean Square' in the fit, add
    # '|V| Mean' to list.  Otherwise, set to None
    #
    # Adding squares of ['|B| Mean', '|V| Mean', 'Proton Density, n/cc Mean']
    # made no meaningful change.  For Kp=7.0, LogBH regression, the added
    # features were near 0 in importance
    ADDSQUARES = None 

    # Whether data is standardized, data ==> data = (data-mean)/(std dev)
    STANDARDIZE = False
        
    # Whether to generate fits
    FITS = True
    
    # Whether to generate plots
    PLOTS = True
    
    # Whether to generate regression fit or quantile fit
    REGRESSION = True
    

    # Get run_info on flags and other run parameters
    # Note, there are a few differences between run_info for all data
    # and run_info for single station in a single year.  Single stations and
    # single year parameters are included in that run_info
    def get_run_info_all( number, distance ): 
        
        if number is None and distance is not None: 
            import sys
            sys.exit('Error: If number is None, distance must be None.')

        run_info = {
            "info": 'all',  # all years, all stations
            "usebh": USEBH,
            "uselog": USELOG,
            "uselogy": USELOGY,
            "standardize": STANDARDIZE,
            "includedXdt": INCLUDEDXDT,
            "number": number,
            "distance": distance,
            "Kp": KP,
            '|B| threshold': BTHRESHOLD,
            "addsquares": ADDSQUARES
            }
        
        return run_info

                    
    ################# autogluon fit

    if FITS:
        for num in file_info['numbers']:
           for dist in file_info['distances']:
               
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
                    

