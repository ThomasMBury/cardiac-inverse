#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 22:10:33 2023

Make figure 5
(a) AP for Torord cell simulation at different BCL
(b) Latency vs BCL

@author: tbury
"""



import numpy as np
import pandas as pd
import os

import plotly.express as px
import plotly.graph_objects as go

import scipy.stats as stats

import matplotlib.pyplot as plt


#----------
# Make AP fig
#------------

df = pd.read_csv('../../data/torord_voltage.csv')
df['time'] = df['time']/1000
df['bcl'] = df['bcl']/1000

bcl_values = np.arange(600,2001,200)/1000
df_plot = df.query('bcl in @bcl_values')

fig = px.line(df_plot, x='time', y='voltage', color='bcl',
              color_discrete_sequence=px.colors.sequential.Plasma_r,
)

fig.update_xaxes(title='Time (s)', range=[-0.02,1])
fig.update_yaxes(title='Voltage (mV)')

fig.update_layout(height=400, width=400,
                  legend=dict(x=0.77, y=0.98, title='BCL (s)'),
                  margin=dict(l=5,r=5,t=5,b=5),
                  )

fig.write_image('../../results/figure5a.png', scale=2)



#----------
# Make latency fig
#-----------
df = pd.read_csv('../../data/torord_ap_specs.csv')
df['latency'] = df['latency']/1000
df['bcl'] = df['bcl']/1000
df = df.query('bcl > 1.1')

x = df['bcl']
y = df['latency']

out = stats.linregress(x,y)
b,c,r,p,se1 = out
regression_text='y={}x+{},  R^2={}'.format(round(b,2),round(c,2),round(r**2,2))          

fig = px.scatter(df, x='bcl', y='latency')

xvals = df['bcl']
yvals = b*xvals + c
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
                    font=dict(size=15),
                    showarrow=False)

fig.update_yaxes(title='Latency (s)')
fig.update_xaxes(title='BCL (s)')

fig.update_layout(height=400, width=400,
                  legend=dict(x=0.75, y=0.98, title='BCL'),
                  margin=dict(l=5,r=5,t=5,b=5),
                  )

fig.write_image('../../results/figure5b.png', scale=2)

print('Figure 5 exported')
















