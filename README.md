
This package examines machine learning analysis of OMNI solar wind and
SuperMAG magnetometer data.  The analysis uses autogluon, sklearn, and other 
statistical and machine learning packages to fit SuperMAG magnetic field data
to the OMNI solar wind data.

# Core Functionality

The software is structured to handle three types of data:
- SuperMAG Data: Downloads and processes geomagnetic data from magnetometer stations 
located across the Earth. It includes functionality to calculate statistics like 
means over specified time windows.
- OMNI Data: Parses and processes high-resolution solar wind data (e.g., magnetic 
field vectors, plasma velocity, and proton density). It includes ballistic 
propagation to translate solar wind data to specific distances from Earth.
- Kp Index: Reads and parses the Kp index, a measure of geomagnetic activity, 
and maps these values to higher-resolution time periods for analysis.

# Data Analysis and Visualization

The package provides several tools for exploring and preparing the data:
- Correlation Analysis: A scatter matrix tool identifies highly correlated 
variables to avoid redundancy in statistical models.
- Distribution Plotting: Generates histograms for key variables from both 
SuperMAG and OMNI datasets to visualize data distributions and outliers.
- Utility Functions: Contains common tasks such as merging OMNI and SuperMAG 
datasets, filtering data by Kp index thresholds, and data cleansing.

# Machine Learning Integration

The package uses the AutoGluon package for predictive modeling:
- Regression and Quantile Fits: Trains machine learning models to predict magnetic 
field variables (specifically $B_H$ mean values) based on solar wind inputs.
- Model Evaluation: Includes tools for generating permutation importance plots 
(to see which features most affect the model), residual plots, predicted versus 
measured comparisons, and calculating prediction efficiency (aka Nash-Sutcliffe 
Efficiency) to evaluate model performance.

# Install

```
git clone https://github.com/dthoma6/swmpy
cd sympy
pip install --editable .
```

# SuperMAG API Requirements

This package uses a fork of the SuperMAG API, available at:

https://supermag.jhuapl.edu/mag/lib/content/api/supermag-api.py

The API was downloaded in February 2026, and a small change was needed for the
most recent version of Python.  As noted in supermag-api.py, cafile is deprecated 
past python 3.5.  Hence,

```
    with urllib.request.urlopen(fetchurl,cafile=cafile) as response:
      longstring = response.read()
```
is replaced with

```
    with urllib.request.urlopen(fetchurl) as response:
      longstring = response.read()
```

# Download data

Data used in this analysis was downloaded from various sites.  

OMNI 1-minute, high-resolution data was downloaded from:

https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/modified/

The SuperMAG station list was downloaded from:

https://supermag.jhuapl.edu/mag/?fidelity=high&tab=stationinfo&start=2023-01-01T00%3A00%3A00.000Z&interval=00%3A23%3A59

Slight editing of list was needed to give every column a name. The last two columns
in the CSV file did not have names.  We named the O1 and O2.  

Additional SuperMAG magnetometer data was downloaded using code in SuperMAG.py 
(see download_SuperMAG.py).

The Kp index was downloaded from:

https://kp.gfz.de/en/data

along with information on the data format

# Scripts

The scripts directory contains the scripts needed to execute the analysis. Users
should edit file_info.py for their computer system.  file_info.py specifies where
the data will be stored on your computer and other details of the analysis. The
scripts also use a run_info structure to specify run details such which variables
to include in the fit, the Kp cut-off, etc.

