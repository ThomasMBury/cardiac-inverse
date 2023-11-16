#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 22:10:33 2023

Make figure 3
(a) Stacked plot of linear regressions for NV-NN
(b) Stacked plot of linear regressions for VV-VN
(c) Histogram showing slopes for NV-NN
(d) Histogram showing slopes for VV-VN

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


#------------
# Stacked plot for VV-VN linear regressions
#-------------

# Import data
df_reg = pd.read_csv('../output/df_reg_vv_vn.csv')

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
                    range=[0.5,1.7],
                  )
fig.update_yaxes(title='VV (s)', 
                    range=[0.9,2.35],
                  )


fig.update_layout(height=300,
                  width=300,
                  margin=dict(l=10, r=10, t=10, b=10)
                  )

fig.write_image('../../results/figure3b.png', scale=2,)



#------------
# Make histogram of gradient values for VV-VN regression
#-------------

df_plot = df_reg.melt(id_vars='segment', 
                      value_vars=['b'],
                      var_name='feature', 
                      value_name='val',
                      )

fig = px.histogram(df_plot, x='val',
                    nbins=60,)

fig.update_xaxes(matches=None)
fig.for_each_xaxis(lambda xaxis: xaxis.update(showticklabels=True))

fig.update_xaxes(title='slope')
fig.update_yaxes(title='count')

fig.update_layout(height=200,
                  width=300,
                  margin=dict(l=10, r=10, t=10, b=10)
                  )
fig.write_image('../../results/figure3d.png', scale=2,)



#------------
# Stacked plot for NV-NN linear regressions
#-------------

# Import data
df_reg = pd.read_csv('../output/df_reg_nv_nn.csv')


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
                    range=[0.3,0.65],
                  )


fig.update_layout(height=300,
                  width=300,
                  margin=dict(l=10, r=10, t=10, b=10)
                  )

# fig.write_html('temp.html')
fig.write_image('../../results/figure3a.png', scale=2,)


#------------
# Make histogram of gradient values for NV-NN regression
#-------------

df_plot = df_reg.melt(id_vars='segment', 
                      value_vars=['m'],
                      var_name='feature', 
                      value_name='val',
                      )

fig = px.histogram(df_plot, x='val',
                    nbins=60,)

fig.update_xaxes(matches=None)
fig.for_each_xaxis(lambda xaxis: xaxis.update(showticklabels=True))

fig.update_xaxes(
    # range=[0.6,1.15], 
    title='slope',
                  )

fig.update_yaxes(title='count')

fig.update_layout(height=200,
                  width=300,
                  margin=dict(l=10, r=10, t=10, b=10)
                  )
fig.write_image('../../results/figure3c.png', scale=2,)

print('Figure 3 exported')


