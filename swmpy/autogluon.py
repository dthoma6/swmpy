#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 20:12:04 2026

@author: Dean Thomas
"""

from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.metrics import r2_score, root_mean_squared_error
import statsmodels.api as sm
from pickle import dump, load

from swmpy.utils import get_data_one, get_data_all, get_prefix, get_suffix, \
    set_plot_rcParams, nse

def _dir_path( file_info, run_info, quantile=False, full=False ):
    """Generate path to directory where autogluon models and plots will be stored.
    The path is dependent on the run options.
    
    Inputs:
        file_info = information, such as paths to directories, for run
        
        run_info = information on flag settings, etc. for this run.  Includes
            year, number, distance, station, uselog, etc.
            
        quantile = whether the premutation is for a quantile fit (True) or
            a regression fit (False)

        full = whether to combine all data (all years and all stations) for the
            fit or just single year and station
                
    Outputs:
        path to directory
    """
    # Folder where we will store dataa. Folder name starts with run flags
    # which we format with get_suffix
    folder = get_suffix( run_info )
    
    if quantile: 
        folder = folder + '_Quantile'
    
    # If data is all years, all stations use short path          
    if full:
        path = join( file_info['Fit Directory'], 
                    folder, 
                    'num=' + str(run_info['number']), 
                    'dist=' + str(run_info['distance']) )
    # If data is single year and single station use longer path
    else:
        path = join( file_info['Fit Directory'], 
                    folder, 
                    run_info['station'], 
                    str(run_info['year']), 
                    'num=' + str(run_info['number']), 
                    'dist=' + str(run_info['distance']) )

    return path

def _get_title( kind, prefix, run_info, full ):
    """Creates a title with information on run flags and parameters
    
    Inputs:
        kind = string describing type of plot, e.g., Mean or Quantile
        
        prefix = information returned from get_prefix on run flags
        
        run_info = information on flag settings, etc. for this run.  Includes
            year, number, distance, station, uselog, etc.
            
        full = whether to combine all data (all years and all stations) for the
            fit or just single year and station
                
    Outputs:
        title = title for use in plots, etc.
    """
    if full:
        title = prefix + kind + ' ' + \
                 str(run_info['distance']) + 'Re ' + \
                 str(run_info['number']) + 'min' 
    else:
        title = prefix + kind + ' ' + \
                 run_info['station'] + ' ' + \
                 str(run_info['distance']) + 'Re ' + \
                 str(run_info['number']) + 'min ' + \
                 str(run_info['year']) 

    return title
    
def autogluon_permutation_plot( file_info, run_info, quantile=False, full=False ):
    """Takes a previously developed autogluon model for combined SuperMAG and 
    OMNI data and generates permutation importance plot for each model that 
    autogluon used (autogluon fits multiple machine learning models to the data.)
    
    Inputs:
        file_info = information, such as paths to directories, for run
        
        run_info = information on flag settings, etc. for this run.  Includes
            year, number, distance, station, uselog, etc.
            
        quantile = whether the premutation is for a quantile fit (True) or
            a regression fit (False)
        
        full = whether to combine all data (all years and all stations) for the
            fit or just single year and station
                
    Outputs:
        feature importance plot for each autogluon model
    """
   
    # Directory where we will retrieve data and store plots
    path = _dir_path( file_info, run_info, quantile=quantile, full=full )
    
    # Load test data
    test_set = load( open(join( path, 'test_data.pkl'), 'rb') )
 
    # Load autogluon model
    predictor = TabularPredictor.load(path)
    
    # We'll make a grid of subplots with one plot per model
    # s1 and s2 give us the size of the grid
    models = predictor.model_names()
    s1 = int( np.sqrt( len(models) ) )
    s2 = int( np.ceil( len(models)/s1 ) )

    set_plot_rcParams( )
    fig, ax = plt.subplots(s1,s2,figsize=(6*s2,3*s1+1) ) 
    axes = ax.reshape(-1)
    
    # Use test data tto generate permutation importance plot  
    test_data = TabularDataset(test_set)
    for i in range( len(models) ):
        model = models[i]
        # Calculates feature importance scores for the given model via 
        # permutation importance.
        importance_df = predictor.feature_importance(test_data, 
                                                     model=model).reset_index()
        importance_df.plot.scatter( x='importance', y='index', 
                                   xerr='stddev', ax=axes[i] )
        axes[i].set_title( model )
         
    # Make unused subplots invisible
    for j in range( len(axes) - len(models) ):
        axes[j+len(models)].set_visible(False)   
        
    # Adjust titles and filenames based on options
    # Prefix added to beginning of titles and filenames
    prefix = get_prefix( run_info )

    fig.suptitle( _get_title( 'Features ', prefix, run_info, full ) )
    fig.tight_layout() 
    
    plt.savefig( join( path, 'Features ' + prefix.strip() + '.png') )
    # plt.close( )
    return

def autogluon_residuals_predict_plot( file_info, run_info, full=False ):
    """Takes a previously developed autogluon model for combined SuperMAG and 
    OMNI data and generates a residual vs predict plot based on the residuals to 
    determine if the residual variance looks constant (autogluon fits multiple 
    machine learning models to the data.)
    
    Inputs:
        file_info = information, such as paths to directories, for run
        
        run_info = information on flag settings, etc. for this run.  Includes
            year, number, distance, station, uselog, etc.
                
        full = whether to combine all data (all years and all stations) for the
            fit or just single year and station
            
    Outputs:
        residuals vs prediction plot for each autogluon model
    """
   
    # Directory where we will retrieve data and store plots
    path = _dir_path( file_info, run_info, full=full )

    # Load test data
    test_set = load( open(join( path, 'test_data.pkl'), 'rb') )
 
    # Select dependent variable
    if run_info['usebh']:
        dependent = 'B_H Mean'
    else:
        dependent = 'dB_H/dt Mean'

    # Load autogluon model
    predictor = TabularPredictor.load(path)
    
    # We'll make a grid of subplots with one plot per model
    # s1 and s2 give us the size of the grid
    models = predictor.model_names()
    s1 = int( np.sqrt( len(models) ) )
    s2 = int( np.ceil( len(models)/s1 ) )
    
    set_plot_rcParams()
    fig, ax = plt.subplots(s1,s2,figsize=(3*s2,3*s1+1) ) 
    axes = ax.reshape(-1)
    
    # Use test data to create residuals plot  
    data = TabularDataset(test_set)
    y_data = test_set[dependent]
        
    for i, model in enumerate( models ):
        print( 'Residuals vs Predict plot: ', model )
        y_pred = predictor.predict(data.drop(columns=[dependent]), model=model)
        residuals = y_pred - y_data
     
        axes[i].scatter(y_pred, residuals, s=3)
        axes[i].set_ylabel("Residuals")
        axes[i].set_xlabel("Predicted")
        
        # Change labels based on options
        if run_info['uselog'] or run_info['uselogy']:
            axes[i].set_ylabel(r'Residuals')
            if run_info['usebh']:
                axes[i].set_xlabel(r'$log_{10}(\overline {B_H})$ Predict')
            else:
                axes[i].set_xlabel(r'$log_{10}(\overline {dB_{H}/dt})$ Predict')
        else:
            axes[i].set_ylabel(r'Residuals')
            if run_info['usebh']:
                axes[i].set_xlabel(r'$\overline {B_H}$ Predict')
            else:
                axes[i].set_xlabel(r'$\overline {dB_{H}/dt}$ Predict')
            

        axes[i].set_title( model )

    # Make unused subplots invisible
    for j in range( len(axes) - len(models) ):
        axes[j+len(models)].set_visible(False)   
        
    # Change titles and filenames based on options
    # Prefix added to beginning of titles and filenames
    prefix = get_prefix( run_info )

    fig.suptitle( _get_title( 'Residuals Plot ', prefix, run_info, full ) )            
    fig.tight_layout() 
    
    plt.savefig( join( path, 'Residuals Plot ' + prefix.strip() + '.png') )
    # plt.close( )
    return

def autogluon_qq_plot( file_info, run_info, full=False):
    """Takes a previously developed autogluon model for combined SuperMAG and 
    OMNI data and generates a qq plot based on the residuals to determine if 
    the residuals are normal (autogluon fits multiple machine learning models 
    to the data.)
    
    Inputs:
        file_info = information, such as paths to directories, for run
        
        run_info = information on flag settings, etc. for this run.  Includes
            year, number, distance, station, uselog, etc.
                
        full = whether to combine all data (all years and all stations) for the
            fit or just single year and station

    Outputs:
        qq plot for each autogluon model
    """
   
    # Directory where we will retrieve data and store plots
    path = _dir_path( file_info, run_info, full=full )
    
    # Load test data
    test_set = load( open(join( path, 'test_data.pkl'), 'rb') )
 
    # Select dependent variable
    if run_info['usebh']:
        dependent = 'B_H Mean'
    else:
        dependent = 'dB_H/dt Mean'

    # Load autogluon model
    predictor = TabularPredictor.load(path)
    
    # We'll make a grid of subplots with one plot per model
    # s1 and s2 give us the size of the grid
    models = predictor.model_names()
    s1 = int( np.sqrt( len(models) ) )
    s2 = int( np.ceil( len(models)/s1 ) )
    
    set_plot_rcParams()
    fig, ax = plt.subplots(s1,s2,figsize=(3*s2,3*s1+1) ) 
    axes = ax.reshape(-1)
    
    # Use test data to create qq plot  
    data = TabularDataset(test_set)
    y_data = test_set[dependent]
        
    for i, model in enumerate( models ):
        print( 'QQ plot: ', model )
        y_pred = predictor.predict(data.drop(columns=[dependent]), model=model)
        residuals = y_pred - y_data
     
        sm.qqplot(residuals, line='s', ax=axes[i])
        axes[i].set_title( model )

    # Make unused subplots invisible
    for j in range( len(axes) - len(models) ):
        axes[j+len(models)].set_visible(False)   
        
    # Change titles and filenames based on options
    # Prefix added to beginning of titles and filenames
    prefix = get_prefix( run_info )

    fig.suptitle( _get_title( 'QQ Plot ', prefix, run_info, full ) )        
    fig.tight_layout() 
    
    plt.savefig( join( path, 'QQ Plot ' + prefix.strip() + '.png') )
    # plt.close( )
    return

def autogluon_predict_measured_plot( file_info, run_info, full=False ):
    """Takes a previously developed autogluon model for combined SuperMAG and 
    OMNI data and generates a plot of predicted vs measured values for each 
    autogluon model. (autogluon fits multiple machine learning models to the data.)
    
    Inputs:
        file_info = information, such as paths to directories, for run
        
        run_info = information on flag settings, etc. for this run.  Includes
            year, number, distance, station, uselog, etc.
            
        full = whether to combine all data (all years and all stations) for the
            fit or just single year and station
                
    Outputs:
        plot of predicted vs measured values for each autogluon model
    """
   
    # Directory where we will retrieve data and store plots
    path = _dir_path( file_info, run_info, full=full )
    
    # Load train and test data
    train_set = load( open(join( path, 'train_data.pkl'), 'rb') )
    test_set  = load( open(join( path, 'test_data.pkl'), 'rb') )
 
    # Select dependent variable
    if run_info['usebh']:
        dependent = 'B_H Mean'
    else:
        dependent = 'dB_H/dt Mean'

    # Load autogluon model
    predictor = TabularPredictor.load(path)
    
    # We'll make a grid of subplots with one plot per model
    # s1 and s2 give us the size of the grid
    models = predictor.model_names()
    s1 = int( np.sqrt( len(models) ) )
    s2 = int( np.ceil( len(models)/s1 ) )

    set_plot_rcParams(fontsize=7)
    fig, ax = plt.subplots(s1,s2,figsize=(3*s2,3*s1+1) ) 
    axes = ax.reshape(-1)
    
    # Prep train and test data 
    train_data = TabularDataset(train_set)
    test_data  = TabularDataset(test_set)
    y_train    = train_set[dependent]
    y_test     = test_set[dependent]
    
    for i, model in enumerate( models ):
        print( 'Predict vs Measured plot: ', model )
        y_fit  = predictor.predict(train_data.drop(columns=[dependent]), 
                                   model=model)
        y_pred = predictor.predict(test_data.drop(columns=[dependent]), 
                                   model=model)
        
        # r2 based on training data
        r2 = r2_score(y_train, y_fit)
        # Nash-Sutcliff model efficiency (aka prediction efficiency)
        pe = nse(y_test, y_pred)
        # Root mean square error
        rmse = root_mean_squared_error( y_test, y_pred )
        
        # Plot data
        axes[i].scatter(y_test, y_pred, s=3, label='Test')
        
        # Change labels based on options
        if run_info['uselog'] or run_info['uselogy']:
            if run_info['usebh']:
                axes[i].set_ylabel(r'$log_{10}(\overline {B_H})$ Predict')
                axes[i].set_xlabel(r'$log_{10}(\overline {B_H})$ Measured')
            else:
                axes[i].set_ylabel(r'$log_{10}(\overline {dB_{H}/dt})$ Predict')
                axes[i].set_xlabel(r'$log_{10}(\overline {dB_{H}/dt})$ Measured')
        else:
            if run_info['usebh']:
                axes[i].set_ylabel(r'$\overline {B_H}$ Predict')
                axes[i].set_xlabel(r'$\overline {B_H}$ Measured')
            else:
                axes[i].set_ylabel(r'$\overline {dB_{H}/dt}$ Predict')
                axes[i].set_xlabel(r'$\overline {dB_{H}/dt}$ Measured')
            
        axes[i].set_title( model + r' $r^2$: ' + str(round(r2, 2)) + r' $pe$: ' + 
                          str(round(pe, 2)) + r' $RMSE$: ' + str(round(rmse, 2)))
        
        # Use the same x and y limits
        xmin, xmax = axes[i].get_xlim()
        ymin, ymax = axes[i].get_ylim()
        amin = np.floor(min(xmin, ymin))
        amax = np.ceil(max(xmax, ymax))
        # axes[i].set_xticks(np.arange(amin,amax))
        # axes[i].set_yticks(np.arange(amin,amax))
        axes[i].set_xlim(amin, amax)
        axes[i].set_ylim(amin, amax)
        axes[i].legend()

    # Make unused subplots invisible
    for j in range( len(axes) - len(models) ):
        axes[j+len(models)].set_visible(False)   
        
    # Change titles and filenames based on options
    # Prefix added to beginning of titles and filenames
    prefix = get_prefix( run_info )

    fig.suptitle( _get_title( 'Fit vs Measured ', prefix, run_info, full ) )
    fig.tight_layout() 
    
    plt.savefig( join( path, 'Fit vs Measured ' + prefix.strip() + '.png') )
    # plt.close( )
    return

def autogluon_quantile_plot( file_info, run_info, alpha=0.05, zoom=None, full=False ):
    """Takes a previously developed autogluon model for combined SuperMAG and 
    OMNI data and generates a plot of predicted and measured values for each 
    autogluon model, along with perdiction interval. (autogluon fits multiple 
    machine learning models to the data.)
    
    Inputs:
        file_info = information, such as paths to directories, for run
        
        run_info = information on flag settings, etc. for this run.  Includes
            year, number, distance, station, uselog, etc.
                
        alpha = specifes quantiles, e.g., if alpha=0.05, the 0.05 and 0.95 
            quantiles are modeled, plus the 0.5 quantile (aka median) that is 
            always modeled.
            
        zoom = if None, do nothing.  If list, e.g., [650,850], set x-axis limits
            to [650,850]
        
        full = whether to combine all data (all years and all stations) for the
            fit or just single year and station
                
    Outputs:
        Quantile plot for each autogluon model with prediction interval
    """
   
    # Directory where we will retrieve data and store plots
    path = _dir_path( file_info, run_info, quantile=True, full=full )
    
    # Load test data
    test_set = load( open(join( path, 'test_data.pkl'), 'rb') )
 
     # For the plot below, we sort the test_set from smallest to largest
    test_set = test_set.sort_values(by=["B_H Mean"])
    test_set = test_set.reset_index(drop=True)

    # Select dependent variable
    if run_info['usebh']:
        dependent = 'B_H Mean'
    else:
        dependent = 'dB_H/dt Mean'

    # Load autogluon model
    predictor = TabularPredictor.load(path)
    
    # We'll make a grid of subplots with one plot per model
    # s1 and s2 give us the size of the grid
    models = predictor.model_names()
    s1 = int( np.sqrt( len(models) ) )
    s2 = int( np.ceil( len(models)/s1 ) )

    set_plot_rcParams(fontsize=7)
    fig, ax = plt.subplots(s1,s2,figsize=(3*s2,3*s1+1) ) 
    axes = ax.reshape(-1)
    
    # Use test data in autogluon model  
    test_data  = TabularDataset(test_set)
    y_test     = test_set[dependent]
    
    # Size of prediction interval
    pct = 100*(1-2*alpha)
    
    for i, model in enumerate( models ):
        print( 'Quantile plot: ', model )
        y_pred = predictor.predict(test_data.drop(columns=[dependent]), 
                                   model=model)
        
        # Plot data
        axes[i].scatter(test_set.index, y_pred[0.5], c='r', s=3, label='Predicted Median')           
        axes[i].fill_between( test_set.index, y_pred[alpha], y_pred[1-alpha], 
                             alpha=0.4, label=str(pct) + "% Prediction Interval")
        axes[i].scatter(test_set.index, y_test, c='g', s=3, label='Test Observations')
        if zoom is not None:
            axes[i].set_xlim(zoom)

        # Change labels based on options
        if run_info['uselog'] or run_info['uselogy']:
            if run_info['usebh']:
                axes[i].set_ylabel(r'$log_{10}(\overline {B_H})$ Predict')
            else:
                axes[i].set_ylabel(r'$log_{10}(\overline {dB_{H}/dt})$ Predict')
            axes[i].set_xlabel(r'Index')
        else:
            if run_info['usebh']:
                axes[i].set_ylabel(r'$\overline {B_H}$ Predict')
            else:
                axes[i].set_ylabel(r'$\overline {dB_{H}/dt}$ Predict')
            axes[i].set_xlabel(r'Index')
            
        axes[i].set_title( model )
        
        # Use the same x and y limits
        axes[i].legend()

    # Make unused subplots invisible
    for j in range( len(axes) - len(models) ):
        axes[j+len(models)].set_visible(False)   
        
    # Change titles and filenames used below based on options
    # Prefix added to beginning of titles and filenames
    prefix = get_prefix( run_info )

    fig.suptitle( _get_title( 'Quantile ', prefix, run_info, full ) )
    fig.tight_layout() 
    
    plt.savefig( join( path, 'Quantile ' + prefix.strip() + '.png') )
    # plt.close( )
    return

def autogluon_regression( file_info, run_info, full=False ):
    """Uses autogluon to model the data using regression. Model uses SuperMAG 
    and OMNI data, specified in run_info. B_H Mean is the response variable.
    
    Inputs:
        file_info = information, such as paths to directories, for run
        
        run_info = information on flag settings, etc. for this run.  Includes
            year, number, distance, station, uselog, etc.
                
        full = whether to combine all data (all years and all stations) for the
            fit or just single year and station
                
    Outputs:
        pickle files for models and scaler
    """
 
    # Get the training and test dataframes for all data (all years, all stations)
    # or for one year and one station
    if full:
        train_set, test_set, scaler = get_data_all( file_info, run_info )
    else:
        train_set, test_set, scaler = get_data_one( file_info, run_info )
    
    # Return if train_set and test_set are None.  This happens
    # when the SuperMAG data set is small
    if train_set is None and test_set is None: return

    # Select dependent variable
    if run_info['usebh']:
        dependent = 'B_H Mean'
        train_set = train_set.drop(['dB_H/dt Mean'], axis=1)
        test_set  = test_set.drop(['dB_H/dt Mean'], axis=1)
    else:
        dependent = 'dB_H/dt Mean'
        train_set = train_set.drop(['B_H Mean'], axis=1)
        test_set  = test_set.drop(['B_H Mean'], axis=1)
        
    # Directory where we will store plots, model data, etc.
    path = _dir_path( file_info, run_info, full=full )

    # Train autogluon model
    problem_type = 'regression'
    eval_metric = 'root_mean_squared_error'
    train_data = TabularDataset(train_set)
    predictor = TabularPredictor(label=dependent,
                                 problem_type=problem_type,
                                 eval_metric=eval_metric,
                                 path=path).fit(train_data) 
    
    # Test autogluon model    
    test_data = TabularDataset(test_set)
    predictor.evaluate(test_data)
    # leaderboard = predictor.leaderboard(test_data)
    # print( leaderboard )
    predictor.fit_summary()
    
    # Save the standard scaler
    if scaler is not None:
        dump( scaler, open(join( path, 'StandardScaler.pkl'), 'wb') )
    
    # Save training and test data sets
    dump( train_set, open(join( path, 'train_data.pkl'), 'wb') )
    dump( test_set,  open(join( path, 'test_data.pkl'), 'wb') )
    return

def autogluon_quantile( file_info, run_info, alpha=0.05, full=False ):
    """Uses autogluon to model the quantiles of the data. Model uses SuperMAG 
    and OMNI data for the given year, specified in run_info. B_H Mean is the
    response variable.
    
    Inputs:
        file_info = information, such as paths to directories, for run
        
        run_info = information on flag settings, etc. for this run.  Includes
            year, number, distance, station, uselog, etc.
        
        alpha = specifes quantiles to be modeled, e.g., if alpha=0.05, then
            the 0.05 and 0.95 quantiles will be modeled, plus the 0.5 quantile
            (aka median) that is always modeled.
        
        full = whether to combine all data (all years and all stations) for the
            fit or just single year and station
                
    Outputs:
        pickle files for models and scaler
    """
 
    # Get the training and test dataframes for all data (all years, all stations)
    # or for one year and one station
    if full:
        train_set, test_set, scaler = get_data_all( file_info, run_info )
    else:
        train_set, test_set, scaler = get_data_one( file_info, run_info )
    
    # Return if train_set and test_set are None.  This happens
    # when the SuperMAG data set is small
    if train_set is None and test_set is None: return
        
    # Select dependent variable, and drop the one we're not using
    if run_info['usebh']:
        dependent = 'B_H Mean'
        train_set = train_set.drop(['dB_H/dt Mean'], axis=1)
        test_set  = test_set.drop(['dB_H/dt Mean'], axis=1)
    else:
        dependent = 'dB_H/dt Mean'
        train_set = train_set.drop(['B_H Mean'], axis=1)
        test_set  = test_set.drop(['B_H Mean'], axis=1)
        
    # Directory where we will store plots, model data, etc.
    path = _dir_path( file_info, run_info, quantile=True, full=full )
 
    # Train autogluon model
    problem_type = 'quantile'
    quantile_levels=[alpha, 0.5,1.0-alpha]
    train_data = TabularDataset(train_set)
    predictor = TabularPredictor(label=dependent,
                                 problem_type=problem_type,
                                 quantile_levels = quantile_levels,
                                 path=path).fit(train_data) 
    
    # Test autogluon model    
    test_data = TabularDataset(test_set)
    predictor.evaluate(test_data)
    # leaderboard = predictor.leaderboard(test_data)
    # print( leaderboard )
    predictor.fit_summary()
    
    # Save the standard scaler
    if scaler is not None:
        dump( scaler, open(join( path, 'StandardScaler.pkl'), 'wb') )
    
    # Save training and test data sets
    dump( train_set, open(join( path, 'train_data.pkl'), 'wb') )
    dump( test_set,  open(join( path, 'test_data.pkl'), 'wb') )

    return

