#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 22:10:33 2023

Make figure 6
Stacked linear regression plot for reentry+cond model simulation
 

@author: tbury
"""



import numpy as np
import pandas as pd
import os

import plotly.express as px
import plotly.graph_objects as go

import scipy.stats as stats

import matplotlib.pyplot as plt


def compute_regressions(df_vv_stats, pvc_thresh):
    '''
    Compute linear regression of VV-VN for each segment in df_vv_stats

    '''
    
    list_reg = []
    segments = df_vv_stats['segment'].unique()
    for segment in segments:
        df_seg = df_vv_stats[df_vv_stats['segment']==segment]
        
        # Comute average NN interval
        nn_mean = df_seg['NN'].mean()
        
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


def compute_regressions_nv_nn(df_vv_stats, pvc_thresh):
    '''
    Compute linear regression of VV-VN for each segment in df_vv_stats

    '''
    
    list_reg = []
    segments = df_vv_stats['segment'].unique()
    for segment in segments:
        df_seg = df_vv_stats[df_vv_stats['segment']==segment]
        
        # Comute average NN interval
        nn_mean = df_seg['NN'].mean()
        
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


sample_rate = 180

# Import data
df_vv = pd.read_csv('../output/df_vv_stats.csv')
df_reg = pd.read_csv('../output/df_reg_vv_vn.csv')
df_reg['unex_var'] = (1-df_reg['r']**2)*df_reg['total_variance']

# Convert to s
df_vv['NN'] = df_vv['NN']/sample_rate
df_vv['NV'] = df_vv['NV']/sample_rate
df_vv['VN'] = df_vv['VN']/sample_rate
df_vv['VV'] = df_vv['VV']/sample_rate
df_vv['nn_avg'] = df_vv['nn_avg']/sample_rate


#-----------
# Simulate reentry model with conductio
#------------

# Get parameters
b = df_reg['b'].mean()
c = df_reg['c'].mean()
sigma = np.sqrt(df_reg['unex_var'].mean())

# Simulate model using VN values of patient (in bigeminy)
df_sim = df_vv[['VN', 'segment']].copy()

def model_reentry_cond(vn):
    vv = b*vn + c + sigma*np.random.normal(0,1)
    return vv


df_sim['VV'] = df_sim.apply(
    lambda row: model_reentry_cond(row['VN']), 
    axis=1)

df_sim['NV'] = df_sim['VV'] - df_sim['VN']
df_sim['NN'] = df_sim['VV']/2



#------------
# Stacked plot for VV-VN linear regressions
#-------------


# Compute linear regression of each segment
df_reg = compute_regressions(df_sim, pvc_thresh=40)

# Assign color based on nn_mean
colors = px.colors.sample_colorscale('Turbo', 100, low=0.0, high=1.0, colortype='rgb')[::-1]
nn_min = df_reg['nn_mean'].min()
nn_max = df_reg['nn_mean'].max()+0.01

def get_col(nn_mean):
    prop = (nn_mean-nn_min)/(nn_max-nn_min)
    assert prop>=0
    idx = int(100*prop)
    col = colors[idx]
    return col

df_reg['color'] = df_reg['nn_mean'].apply(get_col)

# color map based on nn_mean
color_discrete_map = dict()
for segment in df_reg['segment'].unique():
    col = df_reg[df_reg['segment']==segment]['color'].iloc[0]
    color_discrete_map[segment] = col

list_df = []
for segment in df_reg['segment'].unique():
    dic = df_reg.query('segment==@segment').iloc[0]
    xvals = np.linspace(dic['xmin'], dic['xmax'], 100)
    yvals = dic['b']*xvals + dic['c']
    df = pd.DataFrame({'x':xvals, 'y':yvals, 'b':dic['b'], 'c':dic['c'], 'nn_mean':dic['nn_mean'], 
                        'segment':dic['segment'], 'r':dic['r']})
    list_df.append(df)

df_plot = pd.concat(list_df)


fig = px.line(df_plot, x='x', y='y', color='segment',
              hover_data=['b','c','nn_mean','r'],
              color_discrete_map=color_discrete_map,
              )
fig.update_layout(showlegend=False)

fig.update_traces(line=dict(width=1))

fig.update_xaxes(title='VN (s)', 
                    range=[0.45,1.75],
                    tick0=0.6,
                    dtick=0.2,
                    )
fig.update_yaxes(title='VV (s)', 
                    range=[0.89,2.19],
                    tick0=1,
                    dtick=0.2,
                    )

fig.update_layout(height=300,
                  width=300,
                  margin=dict(l=10, r=10, t=10, b=10)
                  )


# fig.write_image('temp.png', scale=2,)
fig.write_image('../../results/figure6d.png', scale=2,)



#------------
# Stacked plot for NV-NN linear regressions
#-------------

df_reg = compute_regressions_nv_nn(df_sim, pvc_thresh=40)

# Assign color based on nn_mean
colors = px.colors.sample_colorscale('Turbo', 100, low=0.0, high=1.0, colortype='rgb')[::-1]
nn_min = df_reg['nn_mean'].min()
nn_max = df_reg['nn_mean'].max()+0.01
    
def get_col(nn_mean):
    prop = (nn_mean-nn_min)/(nn_max-nn_min)
    assert prop>=0
    idx = int(100*prop)
    col = colors[idx]
    return col

df_reg['color'] = df_reg['nn_mean'].apply(get_col)

# color map based on nn_mean
color_discrete_map = dict()
for segment in df_reg['segment'].unique():
    col = df_reg[df_reg['segment']==segment]['color'].iloc[0]
    color_discrete_map[segment] = col

list_df = []
for segment in df_reg['segment'].unique():
    dic = df_reg.query('segment==@segment').iloc[0]
    xvals = np.linspace(dic['xmin'], dic['xmax'], 100)
    yvals = dic['m']*xvals + dic['c']
    df = pd.DataFrame({'x':xvals, 'y':yvals, 'm':dic['m'], 'c':dic['c'], 'nn_mean':dic['nn_mean'], 
                        'segment':dic['segment'], 'r':dic['r']})
    list_df.append(df)


df_plot = pd.concat(list_df)

fig = px.line(df_plot, x='x', y='y', color='segment',
              hover_data=['m','c','nn_mean','r'],
              color_discrete_map=color_discrete_map,
              )
fig.update_layout(showlegend=False)

fig.update_traces(line=dict(width=1))

fig.update_xaxes(title='t_s (s)', 
                    range=[0.45,1.1],
                  )
fig.update_yaxes(title='NV (s)', 
                    range=[0.36,0.63],
                  )

fig.update_layout(height=300,
                  width=300,
                  margin=dict(l=10, r=10, t=10, b=10)
                  )

fig.write_image('../../results/figure6c.png', scale=2,)



















