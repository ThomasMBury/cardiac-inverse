#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 10:49:51 2020

Compute VV-VN, and NV-NN regression over short time windows

@author: tbury
"""

import numpy as np
import pandas as pd
import os

import plotly.express as px
import plotly.graph_objects as go

import scipy.stats as stats

from plotly.subplots import make_subplots
import plotly.graph_objects as go


def compute_regressions_vv_vn(df_vv_stats, pvc_thresh, sample_rate):
    '''
    Compute linear regression of VV-VN for each segment in df_vv_stats

    '''
    
    df_vv_stats = df_vv_stats.copy()
    df_vv_stats['NN'] = df_vv_stats['NN']/sample_rate
    df_vv_stats['NV'] = df_vv_stats['NV']/sample_rate
    df_vv_stats['VN'] = df_vv_stats['VN']/sample_rate
    df_vv_stats['VV'] = df_vv_stats['VV']/sample_rate

    
    list_reg = []
    segments = df_vv_stats['segment'].unique()
    for segment in segments:
        df_seg = df_vv_stats[df_vv_stats['segment']==segment]
        
        
        # Comute average NN interval
        nn_mean = df_seg['nn_avg'].mean()
        
        # Compute linear regression of VV-VN in this segment
        x = df_seg['VN'].values
        y = df_seg['VV'].values
        
        xmin = df_seg['VN'].min()
        xmax = df_seg['VN'].max()
        
        # Only keep segments with sufficiently many PVCs
        if len(x) > pvc_thresh:
            out = stats.linregress(x,y)
            b,c,r,p,se1 = out
            se2 = out.intercept_stderr            
            regression_text='y={}x+{},  R^2={}'.format(round(b,2),round(c,2),round(r**2,2))          
            
            # Total variance in y
            total_variance = y.var()
            
            # Approximate te based on linear regression and guess for t_in
            tin = 0.48
            te_approx = (c-b*tin)/(1-b)
    
            # Store values for linear regression
            d = {'b':b, 'c':c, 'r':r, 'te_approx':te_approx,
                 'se_slope':se1, 'se_intercept':se2, ''
                 'xmin':xmin, 'xmax':xmax, 'nn_mean':nn_mean, 'segment':segment,
                 'total_variance':total_variance}
            
            list_reg.append(d)
        
    df_reg = pd.DataFrame(list_reg) 

    return df_reg


def compute_regressions_nv_nn(df_vv_stats, pvc_thresh, sample_rate):
    '''
    Compute linear regression of NV-NN for each segment in df_vv_stats

    '''
    
    df_vv_stats = df_vv_stats.copy()
    df_vv_stats['NN'] = df_vv_stats['NN']/sample_rate
    df_vv_stats['NV'] = df_vv_stats['NV']/sample_rate
    df_vv_stats['VN'] = df_vv_stats['VN']/sample_rate
    df_vv_stats['VV'] = df_vv_stats['VV']/sample_rate
    
    list_reg = []
    segments = df_vv_stats['segment'].unique()
    for segment in segments:
        df_seg = df_vv_stats[df_vv_stats['segment']==segment]
        
        # Comute average NN interval
        nn_mean = df_seg['nn_avg'].mean()
        
        # Compute linear regression of VV-VN in this segment
        x = df_seg['NN'].values
        y = df_seg['NV'].values
        
        xmin = df_seg['NN'].min()
        xmax = df_seg['NN'].max()
        
        # Only keep segments with sufficiently many PVCs
        if len(x) > pvc_thresh:
            out = stats.linregress(x,y)
            m,c,r,p,se1 = out
            se2 = out.intercept_stderr            
            regression_text='y={}x+{},  R^2={}'.format(round(m,2),round(c,2),round(r**2,2))          
            
            # Total variance in y
            total_variance = y.var()
    
            # Store values for linear regression
            d = {'m':m, 'c':c, 'r':r,
                 'se_slope':se1, 'se_intercept':se2,
                 'xmin':xmin, 'xmax':xmax, 'nn_mean':nn_mean, 'segment':segment,
                 'total_variance':total_variance}
            
            list_reg.append(d)
        
    df_reg = pd.DataFrame(list_reg) 

    return df_reg



#----------------
# Compute for patient
#---------------

record_id = '18'
sample_rate = 180

# df_vv_stats = get_vv_stats(record_id)

# df_vv_stats = df_vv_stats.query('nib==1').copy()

# # Add col for segment number
# seg_length = 5*60*sample_rate # (mins, seconds, Hz)
# df_vv_stats['segment'] = df_vv_stats['sample'].apply(lambda x: int(x//seg_length))

# # Get NN avg using VV/2 (since all bigeminy)
# df_vv_stats['NN'] = df_vv_stats['VV']/2
# df_vv_stats.to_csv('output/df_vv_stats.csv', index=False)


# Import vv stats
df_vv_stats = pd.read_csv('output/df_vv_stats.csv')

# # Get NN avg using VV/2 (since all bigeminy)
# df_vv_stats['NN'] = df_vv_stats['VV']/2

# Compute VV-VN regressions over each segment
df_reg = compute_regressions_vv_vn(df_vv_stats, pvc_thresh=40, sample_rate=sample_rate)
df_reg.to_csv('output/df_reg_vv_vn.csv', index=False)


# Compute NV-NN regressions over each segment
df_reg = compute_regressions_nv_nn(df_vv_stats, pvc_thresh=40, sample_rate=sample_rate)
df_reg.to_csv('output/df_reg_nv_nn.csv', index=False)






