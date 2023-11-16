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


record_id= '18'
sample_rate = 180

df_vv_stats = pd.read_csv('output/df_vv_stats.csv')


df_vv_stats['NN'] = df_vv_stats['NN']/sample_rate
df_vv_stats['NV'] = df_vv_stats['NV']/sample_rate
df_vv_stats['VN'] = df_vv_stats['VN']/sample_rate
df_vv_stats['VV'] = df_vv_stats['VV']/sample_rate

# Functions to compute AIC score for each model

def model_reentry(vn):
    tlag = 0.47
    vv = 1*vn + tlag
    return vv


def model_parasystole(vn):
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


def compute_aic_parasystole(df):
    ''' Compute AIC in a single segment'''
    
    # Get predicitons form model
    df['preds'] = df['VN'].apply(model_parasystole)
    
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
    nn_avg = df['NN'].mean()
    
    df['preds'] = df.apply(lambda x: model_ead(x['VN'], nn_avg), axis=1)
    
    # Get the SSE (sum of squared errors)
    sse = sum((df['preds'] - df['VV'])**2)
    
    n = len(df)

    # AIC score in OLS framework
    # https://stats.stackexchange.com/questions/261273/how-can-i-apply-akaike-information-criterion-and-calculate-it-for-linear-regress
    k = 3 # number of parameters
    aic = n * np.log(sse/n) + 2*k
    
    return aic, sse


# Compute AIC for every segment where >=40 PVCs
list_dict = []

list_segments = df_vv_stats['segment'].unique()

for segment in list_segments:
    
    df = df_vv_stats.query('segment==@segment').copy()
    
    # If less than 40 PVCs do not include
    if len(df)<40:
        continue
    
    aic_reentry, sse_reentry = compute_aic_reentry(df)
    aic_parasystole, sse_parasystole = compute_aic_parasystole(df)
    aic_ead, sse_ead = compute_aic_ead(df)


    d = {'segment':segment,
         'aic_reentry':aic_reentry,
         'aic_parasystole':aic_parasystole,
         'aic_ead':aic_ead,
         'sse_reentry':sse_reentry,
         'sse_reentry_cond':sse_parasystole,
         'sse_ead':sse_ead,
         'n': len(df)
         }
    list_dict.append(d)

    
df_aic = pd.DataFrame(list_dict)
    

# counts
df_aic[df_aic['aic_ead']<df_aic['aic_reentry']]


# Winner for each segment
df_aic['winner'] = df_aic[['aic_reentry','aic_parasystole','aic_ead']].idxmin(axis=1)

# Compute AIC score (probability that this model minimises the info loss)
df_aic['min_aic'] = df_aic[['aic_reentry','aic_parasystole','aic_ead']].min(axis=1)
df_aic['aicS_reentry'] = df_aic.apply(lambda x: np.exp((x['min_aic'] - x['aic_reentry'])/2), axis=1)
df_aic['aicS_parasystole'] = df_aic.apply(lambda x: np.exp((x['min_aic'] - x['aic_parasystole'])/2), axis=1)
df_aic['aicS_ead'] = df_aic.apply(lambda x: np.exp((x['min_aic'] - x['aic_ead'])/2), axis=1)


print(df_aic['winner'].value_counts())








