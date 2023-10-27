#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 22:10:33 2023

Make figure 2
(a) NV-NN scatter plot over whole recording
(b) VV-VN scatterp lot over whole recording
(a) NV-NN scatter plot over 5 min segment
(b) VV-VN scatterp lot over 5 min segment


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


# Import data
df = pd.read_csv('../output/df_vv_stats.csv')
# df_reg = pd.read_csv('output/df_reg.csv')
sample_rate = 180

#--------------
# Scatter plot for VV-VN over entire record
#--------------

df_plot = df[['VV','VN','NV','NN','segment']].copy()
df_plot['segment'] = df_plot['segment'].astype(str)

# Add noise to values to remove aliasing
df_plot['VV'] = df_plot['VV'] + np.random.normal(loc=0, scale=0.5, size=len(df_plot))
df_plot['VN'] = df_plot['VN'] + np.random.normal(loc=0, scale=0.5, size=len(df_plot))
df_plot['NV'] = df_plot['NV'] + np.random.normal(loc=0, scale=0.5, size=len(df_plot))
df_plot['NN'] = df_plot['NN'] + np.random.normal(loc=0, scale=0.5, size=len(df_plot))


##### Plot for VV vs VN
fig = go.Figure()

fig.add_trace(
    go.Scatter(x=df_plot['VN']/sample_rate,
                y=df_plot['VV']/sample_rate,
                mode='markers',
                marker=dict(size=1),
                showlegend=False,
                )
    )

# Do linear regressions
x = df_plot['VN'].values/sample_rate
y = df_plot['VV'].values/sample_rate
lr = stats.linregress(x,y)
m,c,r,p,se1 = lr
regression_text='y={}x+{},  R^2={}'.format(round(m,2),round(c,2),round(r**2,2))          
xvals = df_plot['VN']/sample_rate
yvals = m*xvals + c
fig.add_trace(
    go.Scatter(x=xvals,
                y=yvals,
                mode='lines',
                showlegend=False,
                line = dict(color = 'black', width= 1, 
                            # dash = 'dash',
                            ),
                )
    )

fig.add_annotation(x=0.95, y=0.05,
                    xref='paper',
                    yref='paper',
                    text=regression_text,
                    font=dict(size=12),
                    showarrow=False)

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

fig.write_image('../../results/figure2b.png', scale=2)



#--------------
# Scatter plot for NV-NN over entire record
#--------------


fig = go.Figure()

fig.add_trace(
    go.Scatter(x=df_plot['NN']/sample_rate,
                y=df_plot['NV']/sample_rate,
                mode='markers',
                marker=dict(size=1),
                showlegend=False,
                )
    )

# Do linear regressions
x = df_plot['NN'].values/sample_rate
y = df_plot['NV'].values/sample_rate
lr = stats.linregress(x,y)
m,c,r,p,se1 = lr
regression_text='y={}x+{},  R^2={}'.format(round(m,2),round(c,2),round(r**2,2))          
xvals = df_plot['NN']/sample_rate
yvals = m*xvals + c
fig.add_trace(
    go.Scatter(x=xvals,
                y=yvals,
                mode='lines',
                showlegend=False,
                line = dict(color = 'black', width= 1, 
                            # dash = 'dash',
                            ),
                )
    )

fig.add_annotation(x=0.95, y=0.05,
                    xref='paper',
                    yref='paper',
                    text=regression_text,
                    font=dict(size=12),
                    showarrow=False)

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

fig.write_image('../../results/figure2a.png', scale=2,)





#--------------
# Scatter plot for VV-VN for a particular segment
#--------------

segment = 40

df_plot = df[['VV','VN','NV','NN','segment']].copy()
df_plot['segment'] = df_plot['segment'].astype(str)

df_plot = df.query('segment==@segment')[['VV','VN','NV','NN','segment']].copy()


# Add noise to values to remove aliasing
df_plot['VV'] = df_plot['VV'] + np.random.normal(loc=0, scale=0.5, size=len(df_plot))
df_plot['VN'] = df_plot['VN'] + np.random.normal(loc=0, scale=0.5, size=len(df_plot))
df_plot['NV'] = df_plot['NV'] + np.random.normal(loc=0, scale=0.5, size=len(df_plot))
df_plot['NN'] = df_plot['NN'] + np.random.normal(loc=0, scale=0.5, size=len(df_plot))

fig = go.Figure()

fig.add_trace(
    go.Scatter(x=df_plot['VN']/sample_rate,
                y=df_plot['VV']/sample_rate,
                mode='markers',
                marker=dict(size=2),
                showlegend=False,
                )
    )

# Do linear regressions
x = df_plot['VN'].values/sample_rate
y = df_plot['VV'].values/sample_rate
lr = stats.linregress(x,y)
m,c,r,p,se1 = lr
regression_text='y={}x+{},  R^2={}'.format(round(m,2),round(c,2),round(r**2,2))          
xvals = df_plot['VN']/sample_rate
yvals = m*xvals + c
fig.add_trace(
    go.Scatter(x=xvals,
                y=yvals,
                mode='lines',
                showlegend=False,
                line = dict(color = 'black', width= 1, 
                            # dash = 'dash',
                            ),
                )
    )

fig.add_annotation(x=0.95, y=0.05,
                    xref='paper',
                    yref='paper',
                    text=regression_text,
                    font=dict(size=12),
                    showarrow=False)


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

# fig.write_html('temp.html')
fig.write_image('../../results/figure2d.png', scale=2)






#--------------
# Scatter plot for NV vs NN for a particular segment
#--------------

fig = go.Figure()

fig.add_trace(
    go.Scatter(x=df_plot['NN']/sample_rate,
                y=df_plot['NV']/sample_rate,
                mode='markers',
                marker=dict(size=2),
                showlegend=False,
                )
    )

# Do linear regressions
x = df_plot['NN'].values/sample_rate
y = df_plot['NV'].values/sample_rate
lr = stats.linregress(x,y)
m,c,r,p,se1 = lr
regression_text='y={}x+{},  R^2={}'.format(round(m,2),round(c,2),round(r**2,2))          
xvals = df_plot['NN']/sample_rate
yvals = m*xvals + c
fig.add_trace(
    go.Scatter(x=xvals,
                y=yvals,
                mode='lines',
                showlegend=False,
                line = dict(color = 'black', width= 1, 
                            # dash = 'dash',
                            ),
                )
    )

fig.add_annotation(x=0.95, y=0.05,
                    xref='paper',
                    yref='paper',
                    text=regression_text,
                    font=dict(size=12),
                    showarrow=False)

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
fig.write_image('../../results/figure2c.png', scale=2)







