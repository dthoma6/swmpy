
This package examines machine learning analysis of OMNI solar wind and
SuperMAG magnetometer data.  The analysis uses autogluon, sklearn, and other 
statistical and machine learning packages to fit SuperMAG magnetic field data
to the OMNI solar wind data.

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

