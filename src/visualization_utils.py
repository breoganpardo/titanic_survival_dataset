# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 18:51:49 2021

@author: breog
"""
from matplotlib import cm
import pandas as pd



def pair_plot(X,c,marker='o',n_bins=15,figsize=(18,18)):
    cmap = cm.get_cmap('gnuplot')
    scatter = pd.plotting.scatter_matrix(X, c= c, marker = marker,
                                         s=40, hist_kwds={'bins':n_bins},
                                         figsize=figsize, cmap=cmap)    