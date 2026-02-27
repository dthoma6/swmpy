#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 10:29:42 2026

@author: Dean Thomas
"""

import logging
logging.basicConfig(
    format='%(filename)s:%(funcName)s(): %(message)s',
    level=logging.INFO,
    datefmt='%S')

from .SuperMAG import supermag_download, supermag_download_all, supermag_stats,\
    supermag_plots

from .OMNI import omni_read, omni_stats, omni_plots

from .SPUD import spud_supermag_match, spud_supermag_appres

from .Kp import kp_read, kp_stats

from .scatter_matrix import scatter_matrix

from .distributions import supermag_distribution, omni_distribution

from .utils import stations_list, get_data_one, get_data_all, \
    set_plot_rcParams, get_prefix, get_suffix, pearson_cc, nse
    
from .autogluon import autogluon_permutation_plot, autogluon_qq_plot, \
    autogluon_residuals_predict_plot, autogluon_predict_measured_plot, \
    autogluon_quantile_plot, autogluon_regression, autogluon_quantile
    

