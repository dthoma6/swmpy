#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 10:29:42 2026

@author: Dean Thomas
"""

from setuptools import setup, find_packages

install_requires = [
                        "pandas", 
                        "numpy", 
                        "numba", 
                        "statsmodels", 
                        "scikit-learn", 
                        "matplotlib", 
                        "cartopy", 
                        "autogluon", 
                        # "supermag-api", We fork supermag-api, minor change
                        "fortranformat"
                     ]

setup(
    name='swmpy',
    version='0.5',
    author='Dean Thomas',
    author_email='dean.thomas@physics123.net',
    packages=find_packages(),
    description='Machine learning analysis of OMNI solar wind data and SuperMAG magnetometer data',
    install_requires=install_requires
)
