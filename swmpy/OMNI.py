#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 9 Dec 2025

@author: Dean Thomas
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from datetime import datetime, timedelta, timezone
from os.path import join, basename, dirname
import fortranformat as ff

from swmpy.utils import set_plot_rcParams

# Parses OMNI solar wind data and creates pickle file with the data
#
# OMNI data downloaded from:
# https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/modified/

# The format for 1-min and 5-min* modified (Level-3) OMNI data sets is the follwoing:

# Word                                  Format  Fill val.

#  1  Year			                    I4	    1995 ... 2006
#  2  Day			                    I4	    1 ... 365 or 366
#  3  Hour			                    I3	    0 ... 23
#  4  Minute			                    I3	    0 ... 59 at start of average
#  5  ID for IMF spacecraft	            I3	    99         See  footnote D below
#  6  ID for SW Plasma spacecraft	    I3	    99         See  footnote D below
#  7  # of points in IMF averages	    I4      999
#  8  # of points in Plasma averages 	I4      999
#  9  Percent interp		                I4      999        See  footnote A below
# 10  Timeshift, sec		                I7      999999
# 11  RMS, Timeshift		                I7      999999
# 12  RMS, Phase front normal	        F6.2.   99.99      See Footnotes E, F below
# 13  Time btwn observations, sec	    I7	    9999999    DBOT1, See Footnote C below
# 14  Field magnitude average, nT	    F8.2    9999.99
# 15  Bx, nT (GSE, GSM)		            F8.2    9999.99
# 16  By, nT (GSE)		                F8.2    9999.99
# 17  Bz, nT (GSE)		                F8.2    9999.99
# 18  By, nT (GSM)	                    F8.2	    9999.99    Determined from post-shift GSE components
# 19  Bz, nT (GSM)	                    F8.2	    9999.99    Determined from post-shift GSE components
#                                                          See  footnote "gsm" below
# 20  RMS SD B scalar, nT	            F8.2	    9999.99                
# 21  RMS SD field vector, nT	        F8.2	    9999.99    See  footnote E below
# 22  Flow speed, km/s		            F8.1
# 23  Vx Velocity, km/s, GSE	            F8.1    99999.9
# 24  Vy Velocity, km/s, GSE	            F8.1    99999.9
# 25  Vz Velocity, km/s, GSE	            F8.1    99999.9
# 26  Proton Density, n/cc		        F7.2    999.99
# 27  Temperature, K		                F9.0    9999999.
# 28  Flow pressure, nPa		            F6.2	    99.99      See  footnote G below		
# 29  Electric field, mV/m		        F7.2	    999.99     See  footnote G below
# 30  Plasma beta		                F7.2	    999.99     See  footnote G below
# 31  Alfven mach number		            F6.1	    999.9      See  footnote G below
# 32  X(s/c), GSE, Re		            F8.2    9999.99
# 33  Y(s/c), GSE, Re		            F8.2    9999.99
# 34  Z(s/c), GSE, Re		            F8.2    9999.99
# 35  BSN location, Xgse, Re	            F8.2	    9999.99    BSN = bow shock nose
# 36  BSN location, Ygse, Re	            F8.2    9999.99
# 36  BSN location, Zgse, Re 	        F8.2    9999.99

# Ancilary Data

# 38  AE-index, nT                      I6      99999     See  footnote H below
# 39  AL-index, nT                      I6      99999     See  footnote H below
# 40  AU-index, nT                      I6      99999     See  footnote H below
# 41  SYM/D index, nT                   I6      99999     See  footnote H below
# 42  SYM/H index, nT                   I6      99999     See  footnote H below
# 43  ASY/D index, nT                   I6      99999     See  footnote H below
# 44  ASY/H index, nT                   I6      99999     See  footnote H below
# 45  Na/Np Ratio                       F7.3    9.999     See  footnote I below
# 46  Magnetosonic mach number          F5.1    99.9      See  Footnote K below

# The data may be read with the format statement
# (2I4,4I3,3I4,2I7,F6.2,I7, 8F8.2,4F8.1,F7.2,F9.0,F6.2,2F7.2,F6.1,6F8.2,7I6,F7.3, F5.1)

# * Note For 5-min data  we added fluxes from GOES at the end of each record
#  in format:
#   Proton Flux >10 MeV, 1/(cm**2-sec-ster)  F9.2 99999.99  See  footnote J below
#   Proton Flux >30 MeV, 1/(cm**2-sec-ster)  F9.2 99999.99
#   Proton Flux >60 MeV, 1/(cm**2-sec-ster)  F9.2 99999.99

# Footnote A:

# Percent interp: The percent (0-100) of the points contributing to 
# the 1-min magnetic field averages whose phase front normal (PFN) 
# was interpolated because neither the MVAB-0 nor Cross Product 
# shift techniques yielded a PFN that satisfied its respective tests 
# (see detailed documentation for these).

# Footnote C: 

# The DBOT (Duration Between Observing Times) words: For a 
# given record, we take the 1-min average time shift and estimate, 
# using the solar wind velocity and the location of the observing 
# spacecraft, the time at which the corresponding observation 
# would have been made at the spacecraft. Then we take the 
# difference between this time and the corresponding time of 
# the preceding 1-min record and define this as DBOT1. This 
# difference would be one minute in the absence of PFN and/or 
# flow velocity variations. When this difference becomes negative, 
# we have apparent out-of- sequence arrivals of phase planes. 
# That is, if plane A is observed before plane B at the spacecraft, 
# plane B is predicted to arrive at the target before plane A. 
# Searching for negative DBOT enables finding of such cases.

# DBOT2 is like DBOT1 except that the observation time for 
# the current 1-min record is compared to the latest (most time-
# advanced) previous observation time and not to the observation 
# time of the previous record. Use of DBOT2 helps to find 
# extended intervals of out-of-sequence arrivals.

# We do not capture out-of-sequence-arrival information at 15-s 
# resolution but only at 1-min resolution. The standard deviation 
# in the 1-min averaged time shifts may be used to help find cases 
# of out-of-sequence 15-s data. 

# Footnote gsm:

# The computation of standard By and Bz, GSM is taken from the GEOPACK-2008 
# at http://geo.phys.spbu.ru/~tsyganenko/Geopack-2008.html)
# software package developed by Drs. Nikolai Tsyganenko. 

# Footnote D: 

# The following spacecraft ID's are used:
# 	ACE	71
# 	Geotail	60
# 	IMP 8	50
# 	Wind	52 ( we use definitive plasma data)

# Footnote E: 

# Note that standard deviations for the two vectors are given
# as the square roots of the sum of squares of the standard deviations
# in the component averages.  The component averages are given in
# the records but not their individual standard deviations.

# Footnote F: 

# There are no phase front normal standard deviations in the 5-min
# records.  This word has fill (99.99) for such records.

# Footnote G: 

# Derived parameters are obtained from the following equations.

# Flow pressure = (2*10**-6)*Np*Vp**2 nPa (Np in cm**-3, 
# Vp in km/s, subscript "p" for "proton")

# Electric field = -V(km/s) * Bz (nT; GSM) * 10**-3

# Plasma beta = [(T*4.16/10**5) + 5.34] * Np / B**2 (B in nT)

# Alfven Mach number = (V * Np**0.5) / (20 * B)

# For details on these, see http://omniweb.sci.gsfc.nasa.gov/ftpbrowser/bow_derivation.html

# Footnote H:
# Provisional high res. Indices where taken from World Data Center for Geomagnetism, 
# Kyoto: http://swdcwww.kugi.kyoto-u.ac.jp/aeasy/

# Footnote I:
# The new parameter Na/Np Ratio is included using Wind SWE definitive data.

# Footnote J: 
# Proton fluxes from GOES were taken from http://satdat.ngdc.noaa.gov/sem/goes/data/new_avg/

# Footnote K: Magnetosonic Mach Number = V/Magnetosonic_speed
#             Magnetosonic speed = [(sound speed)**2 + (Alfv speed)**2]**0.5
#             The Alfven speed = 20. * B / N**0.5 
#             The sound speed = 0.12 * [T + 1.28*10**5]**0.5 
#             About Magnetosonic speed check  https://omniweb.sci.gsfc.nasa.gov/bow_derivation1.html 

#############################################################################

def omni_read( info, year ):
    """Parses OMNI solar wind data text files, and stores the data in a pickle
    file for future analyses.  

    Inputs:
        info = infomation, such as paths to directories, for run
        
        year = OMNI text file for specified year is read
                
    Outputs:
        pickle file with parsed OMNI solar wind data
    """
    # path to OMNI solar wind file
    filepath = join(info["OMNI Directory"], 
                         "omni_min" + str(year) + ".asc.txt" )

    # See comment block above with Fortran format for reading OMNI data
    reader = ff.FortranRecordReader('(2I4,4I3,3I4,2I7,F6.2,I7,8F8.2,4F8.1,F7.2,F9.0,F6.2,2F7.2,F6.1,6F8.2,7I6,F7.3,F5.1)')
        
    # Names of OMNI variables plus datetime generated from OMNI variables
    names = ["Year", "Day", "Hour", "Minute", "ID for IMF spacecraft", 
             "ID for SW Plasma spacecraft", "# of points in IMF averages",
            "# of points in Plasma averages", "Percent interp", "Timeshift, sec", 
            "RMS, Timeshift", "RMS, Phase front normal",
            "Time btwn observations, sec", "Field magnitude average, nT", 
            "Bx, nT (GSE, GSM)", "By, nT (GSE)", "Bz, nT (GSE)",
            "By, nT (GSM)", "Bz, nT (GSM)", "RMS SD B scalar, nT", 
            "RMS SD field vector, nT", "Flow speed, km/s", "Vx Velocity, km/s, GSE",
            "Vy Velocity, km/s, GSE", "Vz Velocity, km/s, GSE", "Proton Density, n/cc", 
            "Temperature, K", "Flow pressure, nPa", "Electric field, mV/m", 
            "Plasma beta", "Alfven mach number", "X(s/c), GSE, Re", 
            "Y(s/c), GSE, Re", "Z(s/c), GSE, Re","BSN location, Xgse, Re", 
            "BSN location, Ygse, Re", "BSN location, Zgse, Re", "AE-index, nT", 
            "AL-index, nT", "AU-index, nT", "SYM/D index, nT", "SYM/H index, nT", 
            "ASY/D index, nT", "ASY/H index, nT", "Na/Np Ratio", 
            "Magnetosonic mach number", "Datetime"]
    
    # Read OMNI date into a list
    with open(filepath, 'r') as file:
        
        # Empty list for data
        omni = [] 
        cnt = 0
    
        # Read OMNI data from file
        for line in file:
            tmp = reader.read(line)
    
            # Generate datetime from OMNI data
            yy = tmp[0] # year
            dd = tmp[1] # day
            hh = tmp[2] # hour
            mm = tmp[3] # minute
    
            date = datetime(yy, 1, 1, 0, 0, tzinfo=timezone.utc) + \
                    timedelta( days=dd-1, hours=hh, minutes=mm )
            tmp.append(date)
    
            # Append OMNI date to list
            omni.append(tmp)
            cnt += 1
            if cnt % 10000 == 0: print( 'Processed line: ', cnt )
    
    # Create dataframe with OMNI data    
    omnidf = pd.DataFrame( omni, columns = names )
    
    # Change fill values to NaNs
    omnidf.loc[omnidf['ID for IMF spacecraft']          == 99,  'ID for IMF spacecraft'] = np.nan
    omnidf.loc[omnidf['ID for SW Plasma spacecraft']    == 99,  'ID for SW Plasma spacecraft'] = np.nan
    omnidf.loc[omnidf['# of points in IMF averages']    == 999, '# of points in IMF averages'] = np.nan
    omnidf.loc[omnidf['# of points in Plasma averages'] == 999, '# of points in Plasma averages'] = np.nan
    omnidf.loc[omnidf['Percent interp'] == 999,    'Percent interp'] = np.nan
    omnidf.loc[omnidf['Timeshift, sec'] == 999999, 'Timeshift, sec'] = np.nan
    omnidf.loc[omnidf['RMS, Timeshift'] == 999999, 'RMS, Timeshift'] = np.nan
    omnidf.loc[omnidf['RMS, Phase front normal']     == 99.99,   'RMS, Phase front normal'] = np.nan
    omnidf.loc[omnidf['Time btwn observations, sec'] == 9999999, 'Time btwn observations, sec'] = np.nan
    omnidf.loc[omnidf['Field magnitude average, nT'] == 9999.99, 'Field magnitude average, nT'] = np.nan
    omnidf.loc[omnidf['Bx, nT (GSE, GSM)'] == 9999.99, 'Bx, nT (GSE, GSM)'] = np.nan
    omnidf.loc[omnidf['By, nT (GSE)']      == 9999.99, 'By, nT (GSE)'] = np.nan
    omnidf.loc[omnidf['Bz, nT (GSE)']      == 9999.99, 'Bz, nT (GSE)'] = np.nan
    omnidf.loc[omnidf['By, nT (GSM)']      == 9999.99, 'By, nT (GSM)'] = np.nan
    omnidf.loc[omnidf['Bz, nT (GSM)']      == 9999.99, 'Bz, nT (GSM)'] = np.nan
    omnidf.loc[omnidf['RMS SD B scalar, nT']     ==	9999.99, 'RMS SD B scalar, nT'] = np.nan                
    omnidf.loc[omnidf['RMS SD field vector, nT'] ==	9999.99, 'RMS SD field vector, nT'] = np.nan
    # Flow speed, km/s no fill value listed in comment block above
    omnidf.loc[omnidf['Vx Velocity, km/s, GSE'] == 99999.9, 'Vx Velocity, km/s, GSE'] = np.nan
    omnidf.loc[omnidf['Vy Velocity, km/s, GSE'] == 99999.9, 'Vy Velocity, km/s, GSE'] = np.nan
    omnidf.loc[omnidf['Vz Velocity, km/s, GSE'] == 99999.9, 'Vz Velocity, km/s, GSE'] = np.nan
    omnidf.loc[omnidf['Proton Density, n/cc'] == 999.99,  'Proton Density, n/cc'] = np.nan
    omnidf.loc[omnidf['Temperature, K']       == 9999999.,'Temperature, K'] = np.nan
    omnidf.loc[omnidf['Flow pressure, nPa']   ==	99.99,    'Flow pressure, nPa'] = np.nan		
    omnidf.loc[omnidf['Electric field, mV/m'] ==	999.99,   'Electric field, mV/m'] = np.nan
    omnidf.loc[omnidf['Plasma beta']          ==	999.99,   'Plasma beta'] = np.nan
    omnidf.loc[omnidf['Alfven mach number']   ==	999.9,    'Alfven mach number'] = np.nan
    omnidf.loc[omnidf['X(s/c), GSE, Re'] == 9999.99, 'X(s/c), GSE, Re'] = np.nan
    omnidf.loc[omnidf['Y(s/c), GSE, Re'] == 9999.99, 'Y(s/c), GSE, Re'] = np.nan
    omnidf.loc[omnidf['Z(s/c), GSE, Re'] == 9999.99, 'Z(s/c), GSE, Re'] = np.nan
    omnidf.loc[omnidf['BSN location, Xgse, Re'] == 9999.99, 'BSN location, Xgse, Re'] = np.nan
    omnidf.loc[omnidf['BSN location, Ygse, Re'] == 9999.99, 'BSN location, Ygse, Re'] = np.nan
    omnidf.loc[omnidf['BSN location, Zgse, Re'] == 9999.99, 'BSN location, Zgse, Re'] = np.nan
    omnidf.loc[omnidf['AE-index, nT'] == 99999, 'AE-index, nT'] = np.nan
    omnidf.loc[omnidf['AL-index, nT'] == 99999, 'AL-index, nT'] = np.nan
    omnidf.loc[omnidf['AU-index, nT'] == 99999, 'AU-index, nT'] = np.nan
    omnidf.loc[omnidf['SYM/D index, nT'] == 99999, 'SYM/D index, nT'] = np.nan
    omnidf.loc[omnidf['SYM/H index, nT'] == 99999, 'SYM/H index, nT'] = np.nan
    omnidf.loc[omnidf['ASY/D index, nT'] == 99999, 'ASY/D index, nT'] = np.nan
    omnidf.loc[omnidf['ASY/H index, nT'] == 99999, 'ASY/H index, nT'] = np.nan
    omnidf.loc[omnidf['Na/Np Ratio']     == 9.999, 'Na/Np Ratio'] = np.nan
    omnidf.loc[omnidf['Magnetosonic mach number'] == 99.9, 'Magnetosonic mach number'] = np.nan
        
    # Save data to pickle file
    base = basename( filepath )
    dire = dirname( filepath )
    newfile = join( dire, base + '.pkl' )
    omnidf.to_pickle( newfile )
    return

def _test_calc_dXdt( filepath ):
    """ Test calcdXdt subroutine using arrays with known derivatives
    
    Inputs:
        filepath: path to OMNI file
        
    Outputs:
        stdout on whether test succeeded or no
    """
    omni = pd.read_pickle( filepath )
    
    # Remove lines with no data and get UNIX timestamp in ms
    omnidf = omni.dropna( subset=['Year', 'Day', 'Hour', 'Minute'] )
    omnidf['tval'] = omnidf['Datetime'].astype('int64') / 10**9

    # Create column with with known derivative
    print( 'Test constant, non-zero slope')
    omnidf['test1'] = 10*omnidf['tval']
    dXdt = _calc_dXdt( omnidf['test1'].to_numpy(), omnidf['tval'].to_numpy() )
    for x in dXdt:
        if not np.isclose( x, 10 ):
            print( 'Error in calcdXdt' )
    print( 'Non-zero slope test complete')
    
    print( 'Test constant, zero slope')
    omnidf['test2'] = 10
    dXdt = _calc_dXdt( omnidf['test2'].to_numpy(), omnidf['tval'].to_numpy() )
    for x in dXdt:
        if not np.isclose( x, 0 ):
            print( 'Error in calcdXdt' )
    print( 'Zero slope test complete')

    print( 'Test parabolic slope')
    omnidf['test3'] = omnidf['tval']**2
    dXdt = _calc_dXdt( omnidf['test3'].to_numpy(), omnidf['tval'].to_numpy() )
    for i in range(len(dXdt)):
        x = dXdt[i]
        t = omnidf['tval'][i]
        if not np.isclose( x, 2*t ):
            print( 'Error in calcdXdt' )
    print( 'Parabolic slope test complete')
    return
    
@njit
def _calc_dXdt(X, t):
    """ Subroutine for omnistats that allows numba accelleration. It  
    calculates time derivative of X using data from OMNI file.
    
    Inputs:
        X = numpy array with OMNI variable for which we want the time derivative
        
        t = numpy array with time from OMNI file (tval)
                               
    Outputs:
        dXdt = numpy array with time derivative of X with respect to t
    """

    nX = len(X)
    nt = len(t)
    assert nX == nt 
    
    # Create array to put dXdt into
    dXdt = np.full( nX, np.nan, dtype=float )
    
    # Use stencils to calculate derivatives.
    # We may have unequal intervals, so we must use the correct stencils
    #
    # Singh, Ashok K., and B. S. Bhadauria. "Finite difference formulae for 
    # unequal sub-intervals using Lagrange’s interpolation formula." Int. J. 
    # Math. Anal 3.17 (2009): 815.
    
    for i in range(nX):
        if i > 0 and i < nX-1: # in interior of array
            h1 = t[i]   - t[i-1]
            h2 = t[i+1] - t[i]
            f0 = X[i-1]
            f1 = X[i]
            f2 = X[i+1]
            dXdt[i] = - h2/h1/(h1+h2)*f0 - (h1-h2)/h1/h2*f1 + h1/h2/(h1+h2)*f2
        elif i == 0: # at beginning of array
            h1 = t[i+1] - t[i]
            h2 = t[i+2] - t[i+1]
            f0 = X[i]
            f1 = X[i+1]
            f2 = X[i+2]
            dXdt[i] = - h1/h2/(h1+h2)*f2 + (h1+h2)/h1/h2*f1 - (2*h1+h2)/h1/(h1+h2)*f0        
        else: # i == nX-1: at end of array
            h1 = t[i-1] - t[i-2]
            h2 = t[i]   - t[i-1]
            f0 = X[i-2]
            f1 = X[i-1]
            f2 = X[i]
            dXdt[i] =   h2/h1/(h1+h2)*f0 - (h1+h2)/h1/h2*f1 + (2*h2+h1)/h2/(h1+h2)*f2   
    
    return dXdt

def omni_stats(info, year, number, distance):
    """Calculates statistics (mean and std deviation) for parameters in the
    OMNI solar wind files.  

    Inputs:
        info = information, such as paths to directories, for run

        year = year for associated OMNI solar wind file
        
        number = number of minutes over which statistics gathered (e.g., number = 30
             examine 30 minutes windows)
        
        distance = dist from earth (toward sun) at which solar wind data is valid.
            solar wind data will be ballistically propagated from bow shock nose
            to this point on the GSE x axis.
        
     Outputs:
        pickle file with statistics
    """
    # path to OMNI data directory, files will be read and saved from there
    omnidirectory = info["OMNI Directory"]
    
    # path to OMNI solar wind pickle file
    filepath = join(omnidirectory, "omni_min" + str(year) + ".asc.txt.pkl" )
    omni = pd.read_pickle( filepath )
    
    # Remove lines with no data and get UNIX timestamp in ms
    omnidf = omni.dropna( subset=['Year', 'Day', 'Hour', 'Minute'] )
    omnidf['tval'] = omnidf['Datetime'].astype('int64') / 10**9
    
    
    # Ballistically propagate the solar wind conditions.  That is, add delay
    # for solar wind to travel from OMNI BSN (Re) to DIST (Re) along GSE x axis
    
    # *6378.1 convert to km
    omnidf['Distance'] = (omnidf['BSN location, Xgse, Re'] - distance)*6378.1 
    # *1000 to get ms
    omnidf['tval']     = omnidf['tval'] + \
                        1000.*omnidf['Distance']/omnidf['Vx Velocity, km/s, GSE'] 
    omnidf['Datetime'] = pd.to_datetime( omnidf['tval'], unit='s' )
    omnidf = omnidf.sort_values(by=['tval'])
    omnidf = omnidf.reset_index()
    
    # Get the magnitude of vectors for stats
    omnidf['B_mag'] = np.sqrt( omnidf['Bx, nT (GSE, GSM)']**2 + 
                              omnidf['By, nT (GSE)'] **2 + 
                              omnidf['Bz, nT (GSE)']**2 )
    omnidf['V_mag'] = np.sqrt( omnidf['Vx Velocity, km/s, GSE']**2 + 
                              omnidf['Vy Velocity, km/s, GSE']**2 + 
                              omnidf['Vz Velocity, km/s, GSE'] **2 )
    
    # Get tval values. Useful below.
    tval = omnidf['tval'].values
    
    # Create lists to store stats in
    B_mag_mean = []
    B_mag_std  = []
    
    Bx_GSE_GSM_mean = []
    By_GSE_mean = []
    Bz_GSE_mean = []
    Bx_GSE_GSM_std = []
    By_GSE_std = []
    Bz_GSE_std = []
    
    By_GSM_mean = []
    Bz_GSM_mean = []
    By_GSM_std = []
    Bz_GSM_std = []
    
    V_mag_mean = []
    V_mag_std  = []
    
    Vx_GSE_mean = []
    Vy_GSE_mean = []
    Vz_GSE_mean = []
    Vx_GSE_std = []
    Vy_GSE_std = []
    Vz_GSE_std = []
    
    n_mean = []
    n_std  = []
    
    T_mean = []
    T_std  = []
    
    P_mean = []
    P_std  = []
    
    E_mean = []
    E_std  = []
    
    beta_mean = []
    beta_std  = []
    
    Alfven_mean = []
    Alfven_std  = []
    
    # Commented out because frequently too few samples in data
    # NaNp_mean = []
    # NaNp_std  = []
    
    Mach_mean = []
    Mach_std  = []
    
    tval_std   = []
    date_std   = []
    cnt_std    = []
    
    dBdt_mean  = []
    dBdt_std   = []
    dVdt_mean  = []
    dVdt_std   = []
    dndt_mean  = []
    dndt_std   = []
    
    maxidx = omnidf.shape[0]
    
    # Below we determine startdate and enddate for each chuck of data that is 
    # number minutes long.
    
    # Get time since UNIX epoch for first day of year, the SuperMAG data uses these dates
    # Note, we start at 0 minutes into the year.  See comment for number variable
    startdate = datetime(year, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp()

    # Find where startdate is in the ballistically propagated dataframe
    # This defines startidx, which we use below to march through the data
    for i in range(maxidx):
        if tval[i] > startdate: break
    startidx = i
    
    # Loop through SuperMAG data, find data between startdate and enddate.
    for i in range(maxidx): # This for-loop is too long, but we break out of it below
    
        # To determine enddate, number-1 is needed.  See comment for number variable.
        # If number=30, we want times at 0, 1, 2, 3, ... 28, 29 minutes
        # So if startdate = 0, we add number-1 to get enddate
        #
        # (number-1)*60 because startdate and enddate are in seconds
        enddate = startdate + (number-1)*60  
        
        # print(datetime.fromtimestamp(startdate), datetime.fromtimestamp(enddate))
        
        # Counter to keep track of where we are in reading chunks of data.
        # Each chunk is nominally number data points long.  So in the loops below, we
        # skip number forward, unless some values are missing, then we skip j forward
        cnt = number
        
        # Arrays to temporarily store values for calculating stats
        B_mag_tmp      = np.full( number, np.nan, dtype=float )
        Bx_GSE_GSM_tmp = np.full( number, np.nan, dtype=float )
        By_GSE_tmp     = np.full( number, np.nan, dtype=float )
        Bz_GSE_tmp     = np.full( number, np.nan, dtype=float )
        By_GSM_tmp     = np.full( number, np.nan, dtype=float )
        Bz_GSM_tmp     = np.full( number, np.nan, dtype=float )
        V_mag_tmp      = np.full( number, np.nan, dtype=float )
        Vx_GSE_tmp     = np.full( number, np.nan, dtype=float )
        Vy_GSE_tmp     = np.full( number, np.nan, dtype=float )
        Vz_GSE_tmp     = np.full( number, np.nan, dtype=float )
        n_tmp          = np.full( number, np.nan, dtype=float )
        T_tmp          = np.full( number, np.nan, dtype=float )
        P_tmp          = np.full( number, np.nan, dtype=float )
        E_tmp          = np.full( number, np.nan, dtype=float )
        beta_tmp       = np.full( number, np.nan, dtype=float )
        Alfven_tmp     = np.full( number, np.nan, dtype=float )
        # Commented out because frequently too few samples in data
        # NaNp_tmp      = np.full( number, np.nan, dtype=float )
        Mach_tmp       = np.full( number, np.nan, dtype=float )
        tval_tmp       = np.full( number, np.nan, dtype=float )
        cnt_tmp        = 0 # counter to see if we have enough samples to find std dev
        
        # Loop through array to get number samples for stats        
        for j in range(number):
    
            # Don't go past end of array
            if j+startidx < maxidx:
                
                # Make sure data is past the startdate, if not we have a problem
                if tval[j+startidx] >= startdate:
                    
                    # Make sure we don't run past the enddate.
                    # If this happens, some time steps were skipped in the SuperMAG data
                    if tval[j+startidx] <= enddate:
                        B_mag_tmp[j]      = omnidf['B_mag'][j+startidx]
                        Bx_GSE_GSM_tmp[j] = omnidf['Bx, nT (GSE, GSM)'][j+startidx]
                        By_GSE_tmp[j]     = omnidf['By, nT (GSE)'][j+startidx]
                        Bz_GSE_tmp[j]     = omnidf['Bz, nT (GSE)'][j+startidx]
                        By_GSM_tmp[j]     = omnidf['By, nT (GSM)'][j+startidx]
                        Bz_GSM_tmp[j]     = omnidf['Bz, nT (GSM)'][j+startidx]
                        
                        V_mag_tmp[j]      = omnidf['V_mag'][j+startidx]
                        Vx_GSE_tmp[j]     = omnidf['Vx Velocity, km/s, GSE'][j+startidx]
                        Vy_GSE_tmp[j]     = omnidf['Vy Velocity, km/s, GSE'][j+startidx]
                        Vz_GSE_tmp[j]     = omnidf['Vz Velocity, km/s, GSE'][j+startidx]
                        
                        n_tmp[j]      = omnidf['Proton Density, n/cc'][j+startidx]
                        T_tmp[j]      = omnidf['Temperature, K'][j+startidx]
                        P_tmp[j]      = omnidf['Flow pressure, nPa'][j+startidx]
                        E_tmp[j]      = omnidf['Electric field, mV/m'][j+startidx]
                        
                        beta_tmp[j]   = omnidf['Plasma beta'][j+startidx]
                        Alfven_tmp[j] = omnidf['Alfven mach number'][j+startidx]
                        # Commented out because frequently too few samples in data
                        # NaNp_tmp[j]   = omnidf['Na/Np Ratio'][j+startidx]
                        Mach_tmp[j]   = omnidf['Magnetosonic mach number'][j+startidx]
                        
                        tval_tmp[j]   = omnidf['tval'][j+startidx]
                        cnt_tmp        += 1
                    else:
                        cnt = j
                        break
                
                # Easy to get to the else because ballistic propagation moves
                # a lot of data before the beginning of the year.
                # else:
                #     print( 'Time problem ', i, j, tval[j+startidx], startdate)
               
        # If we have enough points to determine mean, std deviation, store stats
        if cnt_tmp > 1:
            B_mag_mean.append( np.nanmean(B_mag_tmp) )
            B_mag_std.append( np.nanstd(B_mag_tmp) )
           
            Bx_GSE_GSM_mean.append( np.nanmean(Bx_GSE_GSM_tmp) )
            By_GSE_mean.append( np.nanmean(By_GSE_tmp) )
            Bz_GSE_mean.append( np.nanmean(Bz_GSE_tmp) )
            Bx_GSE_GSM_std.append( np.nanstd(Bx_GSE_GSM_tmp) )
            By_GSE_std.append( np.nanstd(By_GSE_tmp) )
            Bz_GSE_std.append( np.nanstd(Bz_GSE_tmp) )
            
            By_GSM_mean.append( np.nanmean(By_GSM_tmp) )
            Bz_GSM_mean.append( np.nanmean(Bz_GSM_tmp) )
            By_GSM_std.append( np.nanstd(By_GSM_tmp) )
            Bz_GSM_std.append( np.nanstd(Bz_GSM_tmp) )
            
            V_mag_mean.append( np.nanmean(V_mag_tmp) )
            V_mag_std.append( np.nanstd(V_mag_tmp) )
            
            Vx_GSE_mean.append( np.nanmean(Vx_GSE_tmp) )
            Vy_GSE_mean.append( np.nanmean(Vy_GSE_tmp) )
            Vz_GSE_mean.append( np.nanmean(Vz_GSE_tmp) )
            Vx_GSE_std.append( np.nanstd(Vx_GSE_tmp) )
            Vy_GSE_std.append( np.nanstd(Vy_GSE_tmp) )
            Vz_GSE_std.append( np.nanstd(Vz_GSE_tmp) )
            
            n_mean.append( np.nanmean(n_tmp) )
            n_std.append( np.nanstd(n_tmp) )
            
            T_mean.append( np.nanmean(T_tmp) )
            T_std.append( np.nanstd(T_tmp) )
            
            P_mean.append( np.nanmean(P_tmp) )
            P_std.append( np.nanstd(P_tmp) )
            
            E_mean.append( np.nanmean(E_tmp) )
            E_std.append( np.nanstd(E_tmp) )
            
            beta_mean.append( np.nanmean(beta_tmp) )
            beta_std.append( np.nanstd(beta_tmp) )
            
            Alfven_mean.append( np.nanmean(Alfven_tmp) )
            Alfven_std.append( np.nanstd(Alfven_tmp) )
            
            # Commented out because frequently too few samples in data
            # NaNp_mean.append( np.nanmean(NaNp_tmp) )
            # NaNp_std.append( np.nanstd(NaNp_tmp) )
            
            Mach_mean.append( np.nanmean(Mach_tmp) )
            Mach_std .append( np.nanstd(Mach_tmp) )
            
            tval_std.append( startdate )
            date_std.append( datetime.fromtimestamp(startdate, tz=timezone.utc) )
            cnt_std.append( cnt_tmp )
            
            # Calculate time derivatives of |B|, |V|, and n
            dBdt = _calc_dXdt(B_mag_tmp, tval_tmp)
            dVdt = _calc_dXdt(V_mag_tmp, tval_tmp)
            dndt = _calc_dXdt(n_tmp, tval_tmp)
            
            dBdt_mean.append( np.nanmean(dBdt) )
            dBdt_std.append( np.nanstd(dBdt) )
            dVdt_mean.append( np.nanmean(dVdt) )
            dVdt_std.append( np.nanstd(dVdt) )
            dndt_mean.append( np.nanmean(dndt) )
            dndt_std.append( np.nanstd(dndt) )
        
        # Update startdate for next loop
        # It's one minute (60 seconds) past last enddate
        startdate = enddate + 60
        startidx += cnt
        cnt       = number
        if startidx >= maxidx: break
    
    # We're done with the loop, store the stats in a pickle file
    statsdf = pd.DataFrame( ) 
    statsdf['tval']        = tval_std
    statsdf['Datetime']    = date_std
    statsdf['Sample Size'] = cnt_std
    
    statsdf['|B| Mean'] = B_mag_mean 
    statsdf['|B| STD']  = B_mag_std  
    
    statsdf['Bx, nT (GSE, GSM) Mean'] = Bx_GSE_GSM_mean 
    statsdf['By, nT (GSE) Mean']      = By_GSE_mean 
    statsdf['Bz, nT (GSE) Mean']      = Bz_GSE_mean 
    statsdf['Bx, nT (GSE, GSM) STD']  = Bx_GSE_GSM_std 
    statsdf['By, nT (GSE) STD']       = By_GSE_std 
    statsdf['Bz, nT (GSE) STD']       = Bz_GSE_std 
    statsdf['By, nT (GSM) Mean']      = By_GSM_mean 
    statsdf['Bz, nT (GSM) Mean']      = Bz_GSM_mean 
    statsdf['By, nT (GSM) STD']       = By_GSM_std 
    statsdf['Bz, nT (GSM) STD']       = Bz_GSM_std 
    
    statsdf['|V| Mean'] = V_mag_mean 
    statsdf['|V| STD']  = V_mag_std  
    
    statsdf['Vx Velocity, km/s, GSE Mean'] = Vx_GSE_mean 
    statsdf['Vy Velocity, km/s, GSE Mean'] = Vy_GSE_mean 
    statsdf['Vz Velocity, km/s, GSE Mean'] = Vz_GSE_mean 
    statsdf['Vx Velocity, km/s, GSE STD']  = Vx_GSE_std 
    statsdf['Vy Velocity, km/s, GSE STD']  = Vy_GSE_std 
    statsdf['Vz Velocity, km/s, GSE STD']  = Vz_GSE_std 
    
    statsdf['Proton Density, n/cc Mean']   = n_mean 
    statsdf['Proton Density, n/cc STD']    = n_std  
    
    statsdf['Temperature, K Mean']         = T_mean 
    statsdf['Temperature, K STD']          = T_std  
    
    statsdf['Flow pressure, nPa Mean']     = P_mean 
    statsdf['Flow pressure, nPa STD']      = P_std  
    
    statsdf['Electric field, mV/m Mean']   = E_mean 
    statsdf['Electric field, mV/m STD']    = E_std  
    
    statsdf['Plasma beta Mean']            = beta_mean 
    statsdf['Plasma beta STD']             = beta_std  
    
    statsdf['Alfven mach number Mean']     = Alfven_mean 
    statsdf['Alfven mach number STD']      = Alfven_std  
    
    # Commented out because frequently too few samples in data
    # statsdf['Na/Np Ratio Mean']            = NaNp_mean 
    # statsdf['Na/Np Ratio STD']             = NaNp_std  
    
    statsdf['Magnetosonic mach number Mean'] = Mach_mean 
    statsdf['Magnetosonic mach number STD']  = Mach_std  
    
    statsdf['d|B|/dt Mean'] = dBdt_mean 
    statsdf['d|B|/dt STD']  = dBdt_std 
    statsdf['d|V|/dt Mean'] = dVdt_mean 
    statsdf['d|V|/dt STD']  = dVdt_std 
    statsdf['dn/dt Mean']   = dndt_mean 
    statsdf['dn/dt STD']    = dndt_std 

    file = 'OMNI-stats-' + str(distance) + 'Re-' + str(number) + 'min-' + \
        str(year) + '.pkl' 
    statsdf.to_pickle( join(omnidirectory, file) )    
    return

def omni_plots(info, year, number, distance):
    """Generates plots of the statistics (mean and std deviation) for a 
       subset of the parameters in the OMNI solar wind files.  Used as a
       quick check of results.

    Inputs:
        info = information, such as paths to directories, for run

        year = year for associated OMNI solar wind file
        
        number = number of minutes over which statistics gathered (e.g., number = 30
             examine 30 minutes windows)
        
        distance = dist from earth (toward sun) at which solar wind data is valid.
            solar wind data will be ballistically propagated from bow shock nose
            to this point on the GSE x axis.
        
     Outputs:
        png files of plotted data
    """
    # path to OMNI data directory, files will be read and saved from there
    omnidirectory = info["OMNI Directory"]
    
    # path to OMNI solar wind pickle file
    filepath = join(omnidirectory, "omni_min" + str(year) + ".asc.txt.pkl" )
    omni = pd.read_pickle( filepath )
    
    # Remove lines with no data and get UNIX timestamp in ms
    omnidf = omni.dropna( subset=['Year', 'Day', 'Hour', 'Minute'] )
    omnidf['tval'] = omnidf['Datetime'].astype('int64') / 10**9
    
    # Get the magnitude of vectors for stats
    omnidf['B_mag'] = np.sqrt( omnidf['Bx, nT (GSE, GSM)']**2 + 
                              omnidf['By, nT (GSE)'] **2 + 
                              omnidf['Bz, nT (GSE)']**2 )
    omnidf['V_mag'] = np.sqrt( omnidf['Vx Velocity, km/s, GSE']**2 + 
                              omnidf['Vy Velocity, km/s, GSE']**2 + 
                              omnidf['Vz Velocity, km/s, GSE'] **2 )

    # path to OMNI solar wind stats pickle file        
    file = 'OMNI-stats-' + str(distance) + 'Re-' + str(number) + 'min-' + \
        str(year) + '.pkl' 
    statsdf = pd.read_pickle( join(omnidirectory, file) )    
    
    # Create array showing when data is available to compare to cnt_std.
    # A check to verify that we're doing everything correctly.
    # 
    # dataavail is number+2 when a point is non-nan, and nan otherwise
    #
    # Plot along side cnt_std below to see if they match up.
    dataavail = np.full(len(omnidf['B_mag']), np.nan, dtype=float)    
    for i in range(len(omnidf['B_mag'])): 
        if not np.isnan( omnidf['B_mag'][i] ): 
            dataavail[i] = number+2
            
    # Plot some of the stats for quality control
    set_plot_rcParams( fontsize=5 )
    fig, ax = plt.subplots(3, sharex=True)
    ax[0].scatter( statsdf['Datetime'], statsdf['|B| STD'], 
                  label=r'$|B|$ STD', s=3 )
    ax[0].scatter( statsdf['Datetime'], statsdf['|B| Mean'], 
                  label=r'$|B|$ Mean', s=3 )
    ax[0].legend()
    ax[1].scatter( statsdf['Datetime'], statsdf['|V| STD'], 
                  label=r'$|V|$ STD', s=3 )
    ax[1].scatter( statsdf['Datetime'], statsdf['|V| Mean'], 
                  label=r'$|V|$ Mean', s=3 )
    ax[1].legend()
    ax[2].scatter( statsdf['Datetime'], statsdf['Sample Size'], 
                  label=r'number. of Samples', s=3 )
    ax[2].scatter( omnidf['Datetime'], dataavail, 
                  label=r'Data Available', s=3 )
    ax[0].set_ylabel(r'$|B|$ STD and Mean')
    ax[1].set_ylabel(r'$|V|$ STD and Mean')
    ax[2].set_ylabel(r'Sample Size')
    ax[2].legend()
    ax[2].set_xlabel("Date")
    fig.suptitle(str(distance) + 'Re ' + str(number) + 'min '+ str(year))
    plt.show()
    file = 'OMNI-stats-' + str(distance) + 'Re-' + \
                str(number) + 'min-' + str(year) + '.png'
    fig.savefig( join(omnidirectory, file) )
    
    # Plot some of the stats for quality control
    fig, ax = plt.subplots(3, sharex=True)
    ax[0].scatter( statsdf['Datetime'], np.log(statsdf['d|B|/dt STD']), 
                  label=r'$log(d|B|/dt$ STD)', s=3 )
    ax[0].scatter( statsdf['Datetime'], np.log(statsdf['d|B|/dt Mean']), 
                  label=r'$log(d|B|/dt$ Mean)', s=3 )
    ax[0].legend()
    ax[1].scatter( statsdf['Datetime'], np.log(statsdf['d|V|/dt STD']), 
                  label=r'$log(d|V|/dt$ STD)', s=3 )
    ax[1].scatter( statsdf['Datetime'], np.log(statsdf['d|V|/dt Mean']), 
                  label=r'$log(d|V|/dt$ Mean)', s=3 )
    ax[1].legend()
    ax[2].scatter( statsdf['Datetime'], np.log(statsdf['dn/dt STD']), 
                  label=r'$log(dn/dt$ STD)', s=3 )
    ax[2].scatter( statsdf['Datetime'], np.log(statsdf['dn/dt Mean']), 
                  label=r'$log(dn/dt$ Mean)', s=3 )
    ax[2].legend()
    ax[0].set_ylabel(r'$log((d|B|/dt)$ STD and Mean')
    ax[1].set_ylabel(r'$log(d|V|/dt)$ STD and Mean')
    ax[2].set_ylabel(r'$log(dn/dt)$ STD and Mean')
    ax[2].set_xlabel("Date")
    fig.suptitle(str(distance) + 'Re ' + str(number) + 'min '+ str(year))
    plt.show()
    file = 'OMNI-stats-dXdt-' + str(distance) + 'Re-' + \
                str(number) + 'min-' + str(year) + '.png'
    fig.savefig( join(omnidirectory, file) )

    return

if __name__ == "__main__":

    # Test calc_dXdt

    # Year for OMNI data
    yr=2024
    
    # Directory where the OMNI files are stored
    directory = "/Volumes/PhysicsHD/swmpy/OMNI/"

    fpath = directory + "omni_min" + str(yr) + ".asc.txt.pkl"
    _test_calc_dXdt( fpath )
