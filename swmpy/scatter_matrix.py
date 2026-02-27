#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 10:13:57 2026

@author: Dean Thomas
"""

import pandas as pd
import numpy as np

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
    df = pd.read_pickle( omnidirectory + filename )  
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
    # of these derived parameters

    # ===> Level 1
    
    # Top Absolute Correlations - cc_mean using all parameters
    # Top Absolute Correlations - cc_mean
    # |V| Mean                     Vx Velocity, km/s, GSE Mean      0.999980
    # Bz, nT (GSE) Mean            Electric field, mV/m Mean        0.986960  <==
    # Bx, nT (GSE, GSM) Mean       By, nT (GSE) Mean                0.941966
    # |V| Mean                     Temperature, K Mean              0.916204
    # Vx Velocity, km/s, GSE Mean  Temperature, K Mean              0.914193
    # Plasma beta Mean             Alfven mach number Mean          0.901133  <==
    # Vy Velocity, km/s, GSE Mean  Temperature, K Mean              0.868849
    # |B| Mean                     Magnetosonic mach number Mean    0.867556  <==
    # Alfven mach number Mean      Magnetosonic mach number Mean    0.864986  <==
    # |V| Mean                     Vy Velocity, km/s, GSE Mean      0.850932
    # Vx Velocity, km/s, GSE Mean  Vy Velocity, km/s, GSE Mean      0.850705
    # Proton Density, n/cc Mean    Flow pressure, nPa Mean          0.764023  <==
    # |B| Mean                     Alfven mach number Mean          0.753960  <==
    #                              Flow pressure, nPa Mean          0.727607
    # Vx Velocity, km/s, GSE Mean  Proton Density, n/cc Mean        0.724736
    # dtype: float64
    
    # After dropping the derived parameters, we see a few remaining r over 0.9,
    # see <== below.  Therefore, we drop Vx (high r), By (high r), and
    # Temperature (high r)
    
    # ===> Level 2
    
    # Top Absolute Correlations - cc_mean
    # |V| Mean                     Vx Velocity, km/s, GSE Mean    0.999992  <==
    # Bx, nT (GSE, GSM) Mean       By, nT (GSE) Mean              0.947255  <==
    # |V| Mean                     Temperature, K Mean            0.946953  <==
    # Vx Velocity, km/s, GSE Mean  Temperature, K Mean            0.946113  <==
    # Vy Velocity, km/s, GSE Mean  Temperature, K Mean            0.882275 
    # Vx Velocity, km/s, GSE Mean  Vy Velocity, km/s, GSE Mean    0.858781 
    # |V| Mean                     Vy Velocity, km/s, GSE Mean    0.858621
    # Vx Velocity, km/s, GSE Mean  Proton Density, n/cc Mean      0.810726
    # |V| Mean                     Proton Density, n/cc Mean      0.809248
    # Vy Velocity, km/s, GSE Mean  Proton Density, n/cc Mean      0.767028
    # Proton Density, n/cc Mean    Temperature, K Mean            0.691547
    # |B| Mean                     Proton Density, n/cc Mean      0.440756
    #                              Bz, nT (GSE) Mean              0.204954
    #                              Bx, nT (GSE, GSM) Mean         0.178071
    #                              Temperature, K Mean            0.152553
    # dtype: float64
    
    # After dropping the above parameters, we're left with:
        
    # ===> Level 3
        
    # Top Absolute Correlations - cc_mean
    # |V| Mean                     Proton Density, n/cc Mean      0.865322
    #                              Vy Velocity, km/s, GSE Mean    0.776010
    # Vy Velocity, km/s, GSE Mean  Proton Density, n/cc Mean      0.752563
    # |B| Mean                     Proton Density, n/cc Mean      0.589828
    #                              Bx, nT (GSE, GSM) Mean         0.362391
    #                              Vy Velocity, km/s, GSE Mean    0.336019
    #                              |V| Mean                       0.288616
    #                              Bz, nT (GSE) Mean              0.270602
    # Bx, nT (GSE, GSM) Mean       Bz, nT (GSE) Mean              0.270151
    #                              Vz Velocity, km/s, GSE Mean    0.231749
    #                              Proton Density, n/cc Mean      0.227188
    # Bz, nT (GSE) Mean            Vz Velocity, km/s, GSE Mean    0.163759
    # |B| Mean                     Vz Velocity, km/s, GSE Mean    0.148066
    # Bz, nT (GSE) Mean            |V| Mean                       0.135806
    #                              Vy Velocity, km/s, GSE Mean    0.126451
    # dtype: float64
    
    # and for STD (std deviation variables):
        
    # Top Absolute Correlations - cc_std
    # Vz Velocity, km/s, GSE STD  Proton Density, n/cc STD      0.833405
    # |B| STD                     Bx, nT (GSE, GSM) STD         0.764237
    # Vy Velocity, km/s, GSE STD  Proton Density, n/cc STD      0.760308
    # |B| STD                     Vz Velocity, km/s, GSE STD    0.737249
    # Bx, nT (GSE, GSM) STD       Vz Velocity, km/s, GSE STD    0.624127
    # Vy Velocity, km/s, GSE STD  Vz Velocity, km/s, GSE STD    0.620639
    # |B| STD                     Proton Density, n/cc STD      0.593903
    #                             Vy Velocity, km/s, GSE STD    0.593720
    # Bz, nT (GSE) STD            |V| STD                       0.586776
    # |V| STD                     Proton Density, n/cc STD      0.568643
    #                             Vy Velocity, km/s, GSE STD    0.457519
    # Bx, nT (GSE, GSM) STD       Vy Velocity, km/s, GSE STD    0.456055
    # Bz, nT (GSE) STD            Vy Velocity, km/s, GSE STD    0.353019
    # |B| STD                     |V| STD                       0.308919
    # Bx, nT (GSE, GSM) STD       Proton Density, n/cc STD      0.299857
    # dtype: float64
      
    # Level 0: For all levels, remove these
    
    # Don't use time, date, sample size, glon, glat, mlt, or mcolat
    df = df.drop(['tval'], axis=1)
    df = df.drop(['Datetime'], axis=1)
    df = df.drop(['Sample Size'], axis=1)
    
    # Same as By, Bz in GSE.  We dropped these before calculating correlation 
    # coefficents
    df = df.drop(['By, nT (GSM) Mean'], axis=1)
    df = df.drop(['By, nT (GSM) STD'], axis=1)
    df = df.drop(['Bz, nT (GSM) Mean'], axis=1)
    df = df.drop(['Bz, nT (GSM) STD'], axis=1)
    
    # Level 1: Drop derived parameters
    
    if level > 0:
        # Flow pressure 
        df = df.drop(['Flow pressure, nPa Mean'], axis=1)
        df = df.drop(['Flow pressure, nPa STD'], axis=1)
    
        # Electric field 
        df = df.drop(['Electric field, mV/m Mean'], axis=1)
        df = df.drop(['Electric field, mV/m STD'], axis=1)
    
        # Plasma beta
        df = df.drop(['Plasma beta Mean'], axis=1)
        df = df.drop(['Plasma beta STD'], axis=1)
    
        # Alfven mach correlated with Plasma Beta
        df = df.drop(['Alfven mach number Mean'], axis=1)
        df = df.drop(['Alfven mach number STD'], axis=1)
    
        # Magnetosonic mach correlated with |B| (in second run)
        df = df.drop(['Magnetosonic mach number Mean'], axis=1)
        df = df.drop(['Magnetosonic mach number STD'], axis=1)

    # Level 2: Drop other parameters that are highly correlated

    if level > 1:
        # Vx correlated with |V|
        df = df.drop(['Vx Velocity, km/s, GSE Mean'], axis=1)
        df = df.drop(['Vx Velocity, km/s, GSE STD'], axis=1)
       
        # Bx and By correlated
        df = df.drop(['By, nT (GSE) Mean'], axis=1)
        df = df.drop(['By, nT (GSE) STD'], axis=1)
    
        # Temperature correlated with |V| and Vx
        df = df.drop(['Temperature, K Mean'], axis=1)
        df = df.drop(['Temperature, K STD'], axis=1)

    columns = df.columns
    std_columns  = []
    mean_columns = []
    for i in range(len(columns)):
        if 'STD' in columns[i]:
            std_columns.append( columns[i] )
        if 'Mean' in columns[i]:
            mean_columns.append( columns[i] )

    cc_std = pd.DataFrame.corr( df[std_columns] )
    cc_mean = pd.DataFrame.corr( df[mean_columns] )
    cc_all = df.corr()
    
    print("Top Absolute Correlations - cc_std")
    print(_get_top_abs_correlations(cc_std, n=15))
    print()
    print("Top Absolute Correlations - cc_mean")
    print(_get_top_abs_correlations(cc_mean, n=15))
    print()
    print("Top Absolute Correlations - cc_all")
    print(_get_top_abs_correlations(cc_all, n=15))
    
    set_plot_rcParams()
    axes = pd.plotting.scatter_matrix(df[std_columns])
    for i in range(np.shape(axes)[0]):
        for j in range(np.shape(axes)[1]):
            if i < j:
                axes[i,j].set_visible(False)    
    
    axes = pd.plotting.scatter_matrix(df[mean_columns])
    for i in range(np.shape(axes)[0]):
        for j in range(np.shape(axes)[1]):
            if i < j:
                axes[i,j].set_visible(False)    

    axes = pd.plotting.scatter_matrix(df)
    for i in range(np.shape(axes)[0]):
        for j in range(np.shape(axes)[1]):
            if i < j:
                axes[i,j].set_visible(False)    
    return

