#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 10:13:57 2026

@author: Dean Thomas
"""

import pandas as pd
import numpy as np
from os.path import join

from swmpy.utils import set_plot_rcParams

# get_redundant_pairs and get_top_abs_correlations borrowed from:
# https://stackoverflow.com/questions/17778394/list-highest-correlation-pairs-from-a-large-correlation-matrix-in-pandas

def _get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def _get_top_abs_correlations(df, n=5):
    '''Top correlation coefficients in descending order'''
    au_corr = df.corr().abs().unstack()
    labels_to_drop = _get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

def scatter_matrix( omnidirectory, year, number, distance, level=0 ):
    """Reads OMNI solar wind pickle file, and generates a scatter matrix.  From 
    this, we identify which variables are highly correlated and will be removed 
    from the data.  Highly-correlated variables can create problems, for example,
    in linear regressions.
    
    Inputs:
        omnidirectory = directory path where OMNI files exist
        
        year = all files for specified year are downloaded
        
        number = number of minutes over which statistics gathered (e.g., number = 30
             examine 30 minutes windows)

        distance = dist from earth (toward sun) at which solar wind data is valid.
            solar wind data will be ballistically propagated from bow shock nose
            to this point on the GSE x-axis.

        level = defines which variables are removed.  Level 0: include all 
            variables.  Level 1: Remove derived parameters as defined in OMNI
            documentation, see below.  Level 2: Remove remaining parameters that 
            have a correlation coefficient over 0.9
            
     Outputs:
        scatter matrix for OMNI solar wind data
    """
 
    filename = 'OMNI-stats-' + str(distance) + 'Re-' + str(number) + 'min-' + str(year) + '.pkl'
    df = pd.read_pickle( join(omnidirectory, filename) )  
    df.set_index('Datetime')
    
    # OMNI documentation at 
    # https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/modified/hro_modified_format.txt
    # states that OMNI data contains...
    #
    # Derived parameters are obtained from the following equations.
    #
    # Flow pressure = (2*10**-6)*Np*Vp**2 nPa (Np in cm**-3, 
    # Vp in km/s, subscript "p" for "proton")
    #
    # Electric field = -V(km/s) * Bz (nT; GSM) * 10**-3
    #
    # Plasma beta = [(T*4.16/10**5) + 5.34] * Np / B**2 (B in nT)
    #
    # Alfven Mach number = (V * Np**0.5) / (20 * B)
    #
    # Magnetosonic speed = [(sound speed)**2 + (Alfv speed)**2]**0.5
    
    # ===> Level 0
    
    # Include all OMNI variables
    
    # Not surprisingly, this leads to high correlation coefficients (r) for 
    # some parameters, see <== immediately below.  Therefore, we remove all
    # of the derived parameters
    
    # Note, the correlations shown below are for year = 2025, number = 30, 
    # distance = 2

    # ===> Level 1: Remove derived parameters
    
    # Top Absolute Correlations
    # Bz, nT (GSE) Mean            Electric field, mV/m Mean        0.988477  <==
    # Bx, nT (GSE, GSM) Mean       By, nT (GSE) Mean                0.940812
    # Vx Velocity, km/s, GSE Mean  Temperature, K Mean              0.906755
    # Plasma beta Mean             Alfven mach number Mean          0.898749  <==
    # Alfven mach number Mean      Magnetosonic mach number Mean    0.866896  <==
    # Vy Velocity, km/s, GSE Mean  Temperature, K Mean              0.866720
    # Vx Velocity, km/s, GSE Mean  Vy Velocity, km/s, GSE Mean      0.852968
    # Proton Density, n/cc Mean    Flow pressure, nPa Mean          0.791687  <==
    # Vx Velocity, km/s, GSE Mean  Proton Density, n/cc Mean        0.667176
    # Vy Velocity, km/s, GSE Mean  Proton Density, n/cc Mean        0.639041
    # Plasma beta Mean             Magnetosonic mach number Mean    0.577850
    # Proton Density, n/cc Mean    Temperature, K Mean              0.482432
    # Temperature, K Mean          Plasma beta Mean                 0.344113
    #                              Alfven mach number Mean          0.321276
    # Vy Velocity, km/s, GSE Mean  Plasma beta Mean                 0.269572
    # dtype: float64
    
    # ===> Level 2: Remove Temperature and By
    
    # After removing the derived parameters, we find the following correlations.
    # Two are over 0.9, so we remove Temperature and By
    
    # Top Absolute Correlations
    # Vx Velocity, km/s, GSE Mean  Temperature, K Mean            0.955621  <==
    # Bx, nT (GSE, GSM) Mean       By, nT (GSE) Mean              0.949614  <==
    # Vy Velocity, km/s, GSE Mean  Temperature, K Mean            0.885310
    # Vx Velocity, km/s, GSE Mean  Vy Velocity, km/s, GSE Mean    0.882467
    #                              Proton Density, n/cc Mean      0.784262
    # Vy Velocity, km/s, GSE Mean  Proton Density, n/cc Mean      0.748267
    # Proton Density, n/cc Mean    Temperature, K Mean            0.677409
    # Vz Velocity, km/s, GSE Mean  Proton Density, n/cc Mean      0.159379
    # Bx, nT (GSE, GSM) Mean       Proton Density, n/cc Mean      0.153550
    #                              Vz Velocity, km/s, GSE Mean    0.145243
    # By, nT (GSE) Mean            Vz Velocity, km/s, GSE Mean    0.138534
    # Bx, nT (GSE, GSM) Mean       Bz, nT (GSE) Mean              0.126020
    # Bz, nT (GSE) Mean            Vz Velocity, km/s, GSE Mean    0.111594
    # Vx Velocity, km/s, GSE Mean  Vz Velocity, km/s, GSE Mean    0.087828
    # By, nT (GSE) Mean            Vy Velocity, km/s, GSE Mean    0.079556
    # dtype: float64
        
    # After dropping the above parameters, we're left with:
        
    # ===> Level 3
        
    # Top Absolute Correlations
    # Vx Velocity, km/s, GSE Mean  Vy Velocity, km/s, GSE Mean    0.887378
    #                              Proton Density, n/cc Mean      0.760650
    # Vy Velocity, km/s, GSE Mean  Proton Density, n/cc Mean      0.728685
    # Bx, nT (GSE, GSM) Mean       Proton Density, n/cc Mean      0.333264
    #                              Bz, nT (GSE) Mean              0.289630
    #                              Vz Velocity, km/s, GSE Mean    0.256184
    # Vz Velocity, km/s, GSE Mean  Proton Density, n/cc Mean      0.243867
    # Vx Velocity, km/s, GSE Mean  Vz Velocity, km/s, GSE Mean    0.235424
    # Bz, nT (GSE) Mean            Vx Velocity, km/s, GSE Mean    0.206401
    # Bx, nT (GSE, GSM) Mean       Vx Velocity, km/s, GSE Mean    0.152206
    # Bz, nT (GSE) Mean            Vz Velocity, km/s, GSE Mean    0.151330
    #                              Proton Density, n/cc Mean      0.098432
    # Vy Velocity, km/s, GSE Mean  Vz Velocity, km/s, GSE Mean    0.042098
    # Bz, nT (GSE) Mean            Vy Velocity, km/s, GSE Mean    0.024963
    # Bx, nT (GSE, GSM) Mean       Vy Velocity, km/s, GSE Mean    0.010044
    # dtype: float64
    
    ###########################################################################
    
    # Level 0: For all levels, remove these.  We don't use them in fits.
    
    # Don't use time, date, sample size, glon, glat, mlt, or mcolat
    df = df.drop(['tval'], axis=1)
    df = df.drop(['Datetime'], axis=1)
    df = df.drop(['Sample Size'], axis=1)
    
    # Same as By, Bz in GSE.  We dropped these before calculating correlation 
    # coefficents
    df = df.drop(['By, nT (GSM) Mean'], axis=1)
    df = df.drop(['Bz, nT (GSM) Mean'], axis=1)
    
    # Level 1: Drop derived parameters
    
    if level > 0:
        # Flow pressure 
        df = df.drop(['Flow pressure, nPa Mean'], axis=1)
    
        # Electric field 
        df = df.drop(['Electric field, mV/m Mean'], axis=1)
    
        # Plasma beta
        df = df.drop(['Plasma beta Mean'], axis=1)
    
        # Alfven mach correlated with Plasma Beta
        df = df.drop(['Alfven mach number Mean'], axis=1)
    
        # Magnetosonic mach correlated with |B| (in second run)
        df = df.drop(['Magnetosonic mach number Mean'], axis=1)

    # Level 2: Drop other parameters that are highly correlated

    if level > 1:
        # By correlated with Bx
        df = df.drop(['By, nT (GSE) Mean'], axis=1)
       
        # Temperature correlated with Vx
        df = df.drop(['Temperature, K Mean'], axis=1)

    columns = df.columns
    cc_mean = pd.DataFrame.corr( df[columns] )
    
    print()
    print("Top Absolute Correlations")
    print(_get_top_abs_correlations(cc_mean, n=15))
    print()
    
    set_plot_rcParams(fontsize=6, figsize = (12,12))
    axes = pd.plotting.scatter_matrix(df[columns])
    for i in range(np.shape(axes)[0]):
        for j in range(np.shape(axes)[1]):
            if i < j:
                axes[i,j].set_visible(False)    
    return

