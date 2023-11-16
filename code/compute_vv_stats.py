#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 07:54:52 2023

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
    anomalies = df_temp[df_temp['interval']>2*180].index
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

    return df_beats, df_vv_stats



#----------------
# Compute for patient
#---------------

record_id = '18'
sample_rate = 180

df_beats, df_vv_stats = get_vv_stats(record_id)
df_vv_stats = df_vv_stats.query('nib==1').copy()

# Add col for segment number
seg_length = 5*60*sample_rate # (mins, seconds, Hz)
df_vv_stats['segment'] = df_vv_stats['sample'].apply(lambda x: int(x//seg_length))

# Get NN approx using VV/2 (since all bigeminy)
df_vv_stats['NN'] = df_vv_stats['VV']/2

# Export
df_vv_stats.to_csv('output/df_vv_stats.csv', index=False)


