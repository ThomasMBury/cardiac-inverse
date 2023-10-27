#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 12:57:34 2023

Compute the AIC scores for each model on each segment of data

@author: tbury
"""


import numpy as np
import pandas as pd
import os

import plotly.express as px
import plotly.graph_objects as go

import scipy.stats as stats

import matplotlib.pyplot as plt


# Get the VV, VN data of patient
def import_vv_stats(record_id):

    # Cornell data
    fileroot='/Users/tbury/research_storage/holter_data/cornell/data_output/'

    # Check existance of data file
    filepath = fileroot+'nn_avg/{}_nnavg_rw4.csv'.format(record_id)
    if not os.path.exists(filepath):
        print('No data for record {}'.format(record_id))
    
    # Import data
    df_nnavg = pd.read_csv(fileroot + 'nn_avg/{}_nnavg_rw4.csv'.format(record_id))
    df_vv_stats = pd.read_csv(fileroot + 'vv_stats/{}_vv_stats.csv'.format(record_id))
    
    # Merge df_nnavg with VV stats data
    df_vv_stats = df_vv_stats.merge(df_nnavg,
                            how='left',
                            on='Time (s)')    
    print('Data imported for record {}'.format(record_id))
    
    # Get VV data where NIB=1
    df_vv_stats = df_vv_stats[df_vv_stats['NIB']==1][['Time (s)', 'NV','VN','VV','NN avg']].copy()

    return df_vv_stats

record_id= '18'
df_vv_stats = import_vv_stats(record_id)
    
# Add col for segment number
seg_length = 5*60 # seconds
df_vv_stats['segment'] = df_vv_stats['Time (s)'].apply(lambda x: int(x//seg_length))



# Compute AIC score for reentry model in each segment

def model_reentry(vn):
    tlag = 0.47
    vv = 1*vn + tlag
    return vv


def model_reentry_cond(vn):
    alpha=0.6
    beta = 0.11
    vv = (1-beta)*vn + alpha
    return vv
    

def model_ead(vn, ts):
    b = 0.89
    gamma = 0.42
    c = 0.26
    # c = 0.27
    vv = b*vn + c + gamma*ts
    return vv
    

    
def compute_aic_reentry(df):
    ''' Compute AIC in a single segment'''
    
    # Get predicitons form model
    df['preds'] = df['VN'].apply(model_reentry)
    
    # Get the SSE (sum of squared errors)
    sse = sum((df['preds'] - df['VV'])**2)

    n = len(df)
    
    # AIC score in OLS framework
    # https://stats.stackexchange.com/questions/261273/how-can-i-apply-akaike-information-criterion-and-calculate-it-for-linear-regress
    k = 1 # number of parameters
    aic = n * np.log(sse/n) + 2*k
    
    return aic, sse


def compute_aic_reentry_cond(df):
    ''' Compute AIC in a single segment'''
    
    # Get predicitons form model
    df['preds'] = df['VN'].apply(model_reentry_cond)
    
    # Get the SSE (sum of squared errors)
    sse = sum((df['preds'] - df['VV'])**2)
    
    n = len(df)
    
    # AIC score in OLS framework
    # https://stats.stackexchange.com/questions/261273/how-can-i-apply-akaike-information-criterion-and-calculate-it-for-linear-regress
    k = 2 # number of parameters
    aic = n * np.log(sse/n) + 2*k
    
    return aic, sse


def compute_aic_ead(df):
    ''' Compute AIC in a single segment'''
    
    # Get predicitons form model
    nn_avg = df['NN avg'].mean()
    
    df['preds'] = df.apply(lambda x: model_ead(x['VN'], nn_avg), axis=1)
    

    
    # Get the SSE (sum of squared errors)
    sse = sum((df['preds'] - df['VV'])**2)
    
    n = len(df)

    # AIC score in OLS framework
    # https://stats.stackexchange.com/questions/261273/how-can-i-apply-akaike-information-criterion-and-calculate-it-for-linear-regress
    k = 3 # number of parameters
    aic = n * np.log(sse/n) + 2*k
    
    return aic, sse



# Compute AIC for every segment
list_dict = []

list_segments = df_vv_stats['segment'].unique()

for segment in list_segments:
    
    df = df_vv_stats.query('segment==@segment').copy()
    
    
    aic_reentry, sse_reentry = compute_aic_reentry(df)
    aic_reentry_cond, sse_reentry_cond = compute_aic_reentry_cond(df)
    aic_ead, sse_ead = compute_aic_ead(df)


    d = {'segment':segment,
         'aic_reentry':aic_reentry,
         'aic_reentry_cond':aic_reentry_cond,
         'aic_ead':aic_ead,
         'sse_reentry':sse_reentry,
         'sse_reentry_cond':sse_reentry_cond,
         'sse_ead':sse_ead,
         'n': len(df)
         }
    list_dict.append(d)

    
df_aic = pd.DataFrame(list_dict)
    

# counts
df_aic[df_aic['aic_ead']<df_aic['aic_reentry']]


# Winner for each segment
df_aic['winner'] = df_aic[['aic_reentry','aic_reentry_cond','aic_ead']].idxmin(axis=1)

# Compute AIC score (probability that this model minimises the info loss)
df_aic['min_aic'] = df_aic[['aic_reentry','aic_reentry_cond','aic_ead']].min(axis=1)
df_aic['aicS_reentry'] = df_aic.apply(lambda x: np.exp((x['min_aic'] - x['aic_reentry'])/2), axis=1)
df_aic['aicS_reentry_cond'] = df_aic.apply(lambda x: np.exp((x['min_aic'] - x['aic_reentry_cond'])/2), axis=1)
df_aic['aicS_ead'] = df_aic.apply(lambda x: np.exp((x['min_aic'] - x['aic_ead'])/2), axis=1)



print(df_aic['winner'].value_counts())



# Example plot of segment
segment = 30
df = df_vv_stats.query('segment==@segment').copy()
nn_avg = df['NN avg'].mean()
df['preds_reentry'] = df['VN'].apply(model_reentry)
df['preds_reentry_cond'] = df['VN'].apply(model_reentry_cond)
df['preds_ead'] = df.apply(lambda x: model_ead(x['VN'], nn_avg), axis=1)


# print(df_aic.query('segment==@segment')['aic_reentry'])
# print(df_aic.query('segment==@segment')['aic_reentry_cond'])
# print(df_aic.query('segment==@segment')['aic_ead'])


import plotly.express as px
df_plot = df[['VN','VV','preds_reentry','preds_reentry_cond', 'preds_ead']]
df_plot = df_plot.melt(id_vars ='VN')
fig = px.scatter(df_plot, x='VN', y='value', color='variable')
fig.write_html('temp.html')












