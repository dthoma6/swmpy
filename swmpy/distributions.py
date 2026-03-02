#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 07:08:28 2026

@author: Dean Thomas
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from swmpy.utils import stations_list, set_plot_rcParams

def supermag_distribution( supermagdirectory, year, number, uselog=False ):
    """Plots the distribution for a subset of the variables associated with
    each SuperMAG station.  
    
    Inputs:
        supermagdirectory = directory path where SuperMAG files exist
        
        year = all files for specified year are plotted
        
        number = number of minutes over which statistics gathered (e.g., number = 30
             examine 30 minutes windows)
        
        uselog = whether to use log10(variable) in distribution
        
     Outputs:
        distribution plot for each station
    """
    # Set matplotlib rcParams
    set_plot_rcParams(fontsize=6, figsize=[12,6])
    
    # get list of stations
    stations = stations_list(year, supermagdirectory)
    
    # names of variables that we want to plot
    names = ['B_mag Mean', 'B_H Mean', 'B_mag STD', 'B_H STD']

    for station in stations:
        
        filename = station + '-stats-' + str(number) + 'min-' + str(year) + '.pkl'
        smdf = pd.read_pickle( supermagdirectory + filename )
        
        data = {}
        for name in names:
            if uselog:
                data[name] = np.log10( smdf[name].values )
            else:
                data[name] = smdf[name].values
    
        # Create histogram
        fig, ax = plt.subplots(1,len(names), sharey=True)
        for i in range(len(names)):
            name = names[i]
            ax[i].hist(data[name], bins=100, alpha=0.5, color='blue', edgecolor='blue')
            if uselog:
                ax[i].set_xlabel('ln(' + name + ')')
            else:
                ax[i].set_xlabel(name)
        ax[0].set_ylabel('Frequency')
        
        fig.suptitle('Histogram of ' + station + '-stats-' + str(number) + \
                     'min-' + str(year) + ' Data')    
        plt.show()

    return

def omni_distribution( omnidirectory, year, number, distance, uselog=False ):
    """Plots the distribution for a subset of the variables associated with 
    each OMNI data set.
    
    Inputs:
        omnidirectory = directory path where OMNI files exist
        
        year = all files for specified year are plotted
        
        number = number of minutes over which statistics gathered (e.g., number = 30
             examine 30 minutes windows)

        distance = dist from earth (toward sun) at which solar wind data is valid.
            solar wind data will be ballistically propagated from bow shock nose
            to this point on the GSE x axis.

        uselog = whether to use log10(variable) in distribution
        
     Outputs:
        distribution plot for each station
    """
    # Set matplotlib rcParams
    set_plot_rcParams(fontsize=6, figsize=[12,6])
  
    filename = 'OMNI-stats-' + str(distance) + 'Re-' + str(number) + 'min-' +\
        str(year) + '.pkl'
    omnidf = pd.read_pickle( omnidirectory + filename )  
    
    # names  of variables that we plot, based on scatter_matrix.py reduction in parameters
    names = ['|B| Mean', '|V| Mean',
            'Bx, nT (GSE, GSM) Mean', 'Bz, nT (GSE) Mean',
            'Vy Velocity, km/s, GSE Mean', 'Vz Velocity, km/s, GSE Mean',
            'Proton Density, n/cc Mean']
    
    data = {}
    for name in names:
        if uselog:
            data[name] = np.log10( omnidf[name].values )
        else:
            data[name] = omnidf[name].values

    # Create histogram
    fig, ax = plt.subplots(1,len(names), sharey=True)
    for i in range(len(names)):
        name = names[i]
        ax[i].hist(data[name], bins=100, alpha=0.5, color='blue', edgecolor='blue')
        if uselog:
            ax[i].set_xlabel('ln(' + name + ')')
        else:
            ax[i].set_xlabel(name)
    ax[0].set_ylabel('Frequency')
    
    fig.suptitle('Histogram of ' + str(distance) + 'Re-' + str(number) + \
                 'min-' + str(year) + ' Data')    
    plt.show()
    
    return

