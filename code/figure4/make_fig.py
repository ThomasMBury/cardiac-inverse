#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 22:10:33 2023

Make figure 4
Histogram showing slope of linear regression for
- patient data
- simulation of reentry
- simulation of reentry with conduction delay

@author: tbury
"""



import numpy as np
import pandas as pd
import os

import plotly.express as px
import plotly.graph_objects as go

import scipy.stats as stats

import matplotlib.pyplot as plt


sample_rate = 180

# Import data
df = pd.read_csv('../output/df_vv_stats.csv')
df_reg = pd.read_csv('../output/df_reg_vv_vn.csv')

df_reg['unex_var'] = (1-df_reg['r']**2)*df_reg['total_variance']

# Convert to s
df['NN'] = df['NN']/sample_rate
df['NV'] = df['NV']/sample_rate
df['VN'] = df['VN']/sample_rate
df['VV'] = df['VV']/sample_rate


#----------------
# Model for reentry - fixed CI with noise
# VV = VN + t_c + eps
#----------------

# Get an approximation for t_c (conduction time)
tc = (df['VV']-df['VN']).mean()

# Get approximation for noise term
sigma = np.sqrt(df_reg['unex_var'].mean())

list_b_m1 = []
list_b_patient = []
list_c_m1 = []
list_c_patient = []

list_p_slopes_m1 = []
list_p_intercept_m1 = []


for segment in df_reg['segment'].unique():
    
    ## Patient data
    df_patient = df.query('segment==@segment')
    # Lin regression of segment
    b_patient = df_reg.query('segment==@segment')['b'].iloc[0]
    c_patient = df_reg.query('segment==@segment')['c'].iloc[0]
    r_patient = df_reg.query('segment==@segment')['r'].iloc[0]
    se_slope_patient = df_reg.query('segment==@segment')['se_slope'].iloc[0]
    se_inter_patient = df_reg.query('segment==@segment')['se_intercept'].iloc[0]
    total_variance = df_reg.query('segment==@segment')['total_variance'].iloc[0]
    
    
    # Simulate model
    vn_vals = df_patient['VN']
    noise_vals = np.random.normal(0, sigma, len(vn_vals))
    vv_vals = vn_vals + tc + noise_vals
    df_null = pd.DataFrame({'VN':vn_vals, 'VV':vv_vals})
    
    # Compute linear regression 
    out = stats.linregress(vn_vals, vv_vals)
    b_null, c_null, r_null, p_null, se_slope_null = out
    se_inter_null = out.intercept_stderr
    reg_text_null = "y={}x+{},  R^2={}".format(round(b_null, 4), round(c_null, 4), round(r_null**2, 4))
    # print(reg_text_null)
    
    list_b_m1.append(b_null)
    list_c_m1.append(c_null)

    list_b_patient.append(b_patient)
    list_c_patient.append(c_patient)
    

#----------------
# Model:
# VV = b*VN + c + eps
#----------------

# Obtain b as mean slope of linear regressions
b = df_reg['b'].mean()

# Obtain c as mean intercept of linear regressions
c = df_reg['c'].mean()

# Get approximation for noise term
sigma = np.sqrt(df_reg['unex_var'].mean())

list_b_m2 = []
list_b_patient = []
list_c_m2 = []
list_c_patient = []

list_p_slopes_m2 = []
list_p_intercept_m2 = []


for segment in df_reg['segment'].unique():
    
    ## Patient data
    df_patient = df.query('segment==@segment')
    # Lin regression of segment
    b_patient = df_reg.query('segment==@segment')['b'].iloc[0]
    c_patient = df_reg.query('segment==@segment')['c'].iloc[0]
    r_patient = df_reg.query('segment==@segment')['r'].iloc[0]
    se_slope_patient = df_reg.query('segment==@segment')['se_slope'].iloc[0]
    se_inter_patient = df_reg.query('segment==@segment')['se_intercept'].iloc[0]
    total_variance = df_reg.query('segment==@segment')['total_variance'].iloc[0]
    
    # Simulate model
    vn_vals = df_patient['VN']
    noise_vals = np.random.normal(0, sigma, len(vn_vals))
    vv_vals = b*vn_vals + c + noise_vals
    df_null = pd.DataFrame({'VN':vn_vals, 'VV':vv_vals})
    
   
    # Compute linear regression
    out = stats.linregress(vn_vals, vv_vals)
    b_null, c_null, r_null, p_null, se_slope_null = out
    se_inter_null = out.intercept_stderr
    reg_text_null = "y={}x+{},  R^2={}".format(round(b_null, 4), round(c_null, 4), round(r_null**2, 4))
    # print(reg_text_null)
    
    list_b_m2.append(b_null)
    list_c_m2.append(c_null)

    list_b_patient.append(b_patient)
    list_c_patient.append(c_patient)



########### Plot to compare b values
df_plot = pd.DataFrame(
    {'reentry':list_b_m1,
      'reentry+variable conduction':list_b_m2,
      'patient data':list_b_patient
      }
    )

df_plot = df_plot.melt(value_vars=['patient data','reentry', 'reentry+variable conduction'])

fig = px.histogram(df_plot, x='value', color='variable', nbins=60)
fig.update_layout(barmode='overlay')

# Reduce opacity to see both histograms
fig.update_traces(opacity=0.75)

fig.update_layout(width=400,
                  height=300,
                  legend=dict(
                      yanchor="top",
                      y=0.99,
                      xanchor="left",
                      x=0.01,
                      title='',
                      ),
                  margin=dict(l=0, r=10, t=10, b=10)
                  )

fig.update_xaxes(title='slope of linear regression')
fig.update_yaxes(title='count')

fig.write_image('../../results/figure4.png', scale=2)

print('Figure 4 exported')









