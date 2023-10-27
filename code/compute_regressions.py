#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 10:49:51 2020

Compute VV-VN, and NV-NN regression over short time windows

Make scatter plots of VV-VN for selected segments

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




def get_nib_values(list_labels, verbose=0):
    '''
    Compute the NIB values for list of beat annotations list_labels

    A value of -1 means that NIB could not be computed due to interuption from
    noise values.

    Parameters:
        list_labels: list of 'N','V','Q','S' corresponding to beats

    '''
    nib = np.nan
    count = 0
    list_nib = []

    for idx, label in enumerate(list_labels):

        if label=='N':
            nib+=1
            count+=1

        if label=='V':

            # Convert Nan to -1 to keep all values integers
            nib = -1 if np.isnan(nib) else nib
            list_nib.extend([nib]*(count+1))

            # Reset counts
            nib=0
            count=0

        if label in ['Q', 'S']:
            nib=np.nan
            count+=1

        if verbose:
            if idx%10000==0:
                print('Complete for index {}'.format(idx))

    # Add the final labels (must be -1 if remaining)
    l_remaining = len(list_labels)-len(list_nib)
    if l_remaining > 0:
        list_nib.extend([-1]*l_remaining)

    return list_nib


def compute_rr_nib_nnavg(df_beats):
    '''
    Compute RR intervals, NIB values, and NN avg
    Places into input dataframe and returns

    Parameters
    ----------
    df_beats : pd.DataFrame
        Beat annotation labels.
        Cols ['sample', 'type']

    Returns
    -------
    df_beats : pd.DataFrame

    '''

    #------------
    # Compute RR intervals
    #--------------

    # Remove + annotation which indicates rhythm change
    df_beats = df_beats[df_beats['type'].isin(['N','V','S','Q'])].copy()

    # Compute RR intervals and RR type (NN, NV etc.)
    df_beats['interval'] = df_beats['sample'].diff()
    df_beats['type_previous'] = df_beats['type'].shift(1)
    df_beats['interval_type'] = df_beats['type_previous'] + df_beats['type']
    df_beats.drop('type_previous', axis=1, inplace=True)


    #------------
    # Compute NN avg over 1 minute intervals
    # Approximate by the average of all intervals of type NN, NV, VN
    # Only do computation if at least 10 beats in interval
    #----------

    df_beats['minute'] = (df_beats['sample']//(250*60)).astype(int)

    df_temp = df_beats[df_beats['interval_type'].isin(['NN','NV','VN'])].copy()
    # Remove rows that have interval > 2s -  these are due to missing
    # noise label in data.
    anomalies = df_temp[df_temp['interval']>2*250].index
    df_temp = df_temp.drop(anomalies)
    
    # Count number of beats in each minute
    beats_per_min = df_temp.groupby('minute')['interval'].count()
    beats_per_min.name = '#beats'
    beats_per_min = beats_per_min.reset_index()
    df_temp = df_temp.merge(beats_per_min, on='minute')
    
    # Take mean not median (median of all bigeminy would not give good average)
    # Only compute NN avg on miinute intervals with >10 beats
    df_temp = df_temp[df_temp['#beats'] > 10]
    nn_avg = df_temp.groupby('minute')['interval'].mean()
    nn_avg.name = 'nn_avg'
    nn_avg = nn_avg.reset_index()
    df_beats = df_beats.merge(nn_avg, on='minute')


    #-----------
    # Compute NIB values
    #-----------

    list_nib = get_nib_values(df_beats['type'], verbose=0)
    df_beats['nib'] = list_nib

    return df_beats




def get_vv_stats(record_id):

    # Mapping from Cornell beat classificaction type to UBC Holter type
    dict_beat_type = {
        0:'N',
        1:'S',
        2:'V',
        3:'V',
        4:'N',
        5:'V',
        6:'N',
        8:'Q'
        }
    
    filename = 'RR-pat{}.txt'.format(record_id)
    # Import beat data
    df = pd.read_csv('../data/'+filename, sep='\t', index_col=False)
    # Drop final row that has no data
    df = df.iloc[:-1]
    # Make int type
    df['sample'] = df['Sample'].astype('int')
    df['type'] = df['Type'].astype(int)
    df = df.drop(['Sample','Type','RR(ms)'],axis=1)
    # Apply label map
    df['type'] = df['type'].map(dict_beat_type)

    # Compute RR interval, NN avg, and NIB vals
    df_beats = compute_rr_nib_nnavg(df)

    # Get VV stats
    df_vv_stats = df_beats.query('type=="V"')[['sample','nib', 'nn_avg']]
    df_vv_stats['VV'] = df_vv_stats['sample'].diff()
    df_vv_stats['NV'] = df_beats.loc[df_vv_stats.index]['interval']
    
    df_vv_stats = df_vv_stats.query('nib!=-1').copy()
    idx_values = df_vv_stats.index - df_vv_stats['nib']
    df_vv_stats['VN'] = df_beats.loc[idx_values]['interval'].values    

    return df_vv_stats





# def get_vv_stats(record_id):
#     df_beats = pd.read_csv('../data/{}_df_beats.csv'.format(record_id))
    
#     # Compute RR, NIB and NNavg
#     df_beats = compute_rr_nib_nnavg(df_beats)
    
#     # Get VV stats
#     df_vv_stats = df_beats.query('type=="V"')[['sample','nib', 'nn_avg']]
#     df_vv_stats['VV'] = df_vv_stats['sample'].diff()
#     df_vv_stats['NV'] = df_beats.loc[df_vv_stats.index]['interval']
    
#     df_vv_stats = df_vv_stats.query('nib!=-1').copy()
#     idx_values = df_vv_stats.index - df_vv_stats['nib']
#     df_vv_stats['VN'] = df_beats.loc[idx_values]['interval'].values    

#     return df_vv_stats


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



def plot_regressions(df_reg):
    '''
    Make plot of all linear regressions.
    Color of regression for NN avg
    
    '''
    
    # Assign color based on nn_mean
    colors = px.colors.sample_colorscale('RdBu', 100, low=0.0, high=1.0, colortype='rgb')[::-1]
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
    fig.update_xaxes(title='VN', range=[0.4,2.5])
    fig.update_yaxes(title='VV', range=[0.5,3])
    
    return fig
    

def make_histograms(df_reg):
    
    df_plot = df_reg.melt(id_vars='segment', 
                          value_vars=['b','c'],
                          var_name='feature', 
                          value_name='val',
                          )
    
    fig = px.histogram(df_plot, x='val', facet_col='feature',
                       nbins=30,)
    
    fig.update_xaxes(matches=None)
    fig.for_each_xaxis(lambda xaxis: xaxis.update(showticklabels=True))
    
    fig.update_xaxes(range=[0.6,1.4], col=1)
    fig.update_xaxes(range=[0,1], col=2)

    return fig




#----------------
# Compute for patient
#---------------

record_id = '18'
sample_rate = 180

df_vv_stats = get_vv_stats(record_id)

df_vv_stats = df_vv_stats.query('nib==1').copy()

# Add col for segment number
seg_length = 5*60*sample_rate # (mins, seconds, Hz)
df_vv_stats['segment'] = df_vv_stats['sample'].apply(lambda x: int(x//seg_length))

# Get NN avg using VV/2 (since all bigeminy)
df_vv_stats['NN'] = df_vv_stats['VV']/2
df_vv_stats.to_csv('output/df_vv_stats.csv')


# Compute VV-VN regressions over each segment
df_reg = compute_regressions_vv_vn(df_vv_stats, pvc_thresh=40, sample_rate=sample_rate)
df_reg.to_csv('output/df_reg_vv_vn.csv', index=False)


# Compute NV-NN regressions over each segment
df_reg = compute_regressions_nv_nn(df_vv_stats, pvc_thresh=40, sample_rate=sample_rate)
df_reg.to_csv('output/df_reg_nv_nn.csv', index=False)






