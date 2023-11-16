#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 10:49:51 2020

Stat test to compare patient data with 
    - model for fixed reentry
    - model for parasystole
    - model for EADs

@author: tbury
"""

import numpy as np
import pandas as pd
import os

import plotly.express as px
import plotly.graph_objects as go

import scipy.stats as stats

import funs as funs

import matplotlib.pyplot as plt

np.random.seed(0)

def get_vv_stats(record_id):
    fileroot = '/Users/tbury/research_storage/holter_data/cornell/RRdata/'

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
    df = pd.read_csv(fileroot+filename, sep='\t', index_col=False)
    # Drop final row that has no data
    df = df.iloc[:-1]
    # Make int type
    df['sample'] = df['Sample'].astype('int')
    df['type'] = df['Type'].astype(int)
    df = df.drop(['Sample','Type','RR(ms)'],axis=1)
    # Apply label map
    df['type'] = df['type'].map(dict_beat_type)

    # Compute RR interval, NN avg, and NIB vals
    df_beats = funs.compute_rr_nib_nnavg(df)

    # Get VV stats
    df_vv_stats = df_beats.query('type=="V"')[['sample','nib', 'nn_avg']]
    df_vv_stats['vv_interval'] = df_vv_stats['sample'].diff()
    df_vv_stats['nv_interval'] = df_beats.loc[df_vv_stats.index]['interval']
    
    df_vv_stats = df_vv_stats.query('nib!=-1').copy()
    idx_values = df_vv_stats.index - df_vv_stats['nib']
    df_vv_stats['vn_interval'] = df_beats.loc[idx_values]['interval'].values    

    return df_vv_stats


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



#-------------
# Import data and compute regressions
#-------------

sample_rate = 180
record_id = '18'
df = get_vv_stats(record_id)
# Only keep bigeminy
df = df.query('nib==1')
df['VV'] = df['vv_interval']/sample_rate
df['NV'] = df['nv_interval']/sample_rate
df['VN'] = df['vn_interval']/sample_rate

# Add col for segment number
seg_length = 5*60*sample_rate # (mins, seconds, Hz)
df['segment'] = df['sample'].apply(lambda x: int(x//seg_length))

# Get NN avg using VV/2 (since all bigeminy)
df['NN'] = df['VV']/2

df_reg = compute_regressions(df, pvc_thresh=40)

df_reg['unex_var'] = (1-df_reg['r']**2)*df_reg['total_variance']



#----------------
# Model for reentry - fixed CI with noise
# VV = VN + t_c + eps
#----------------

# Get an approximation for t_c (conduction time)
tc = (df['VV']-df['VN']).mean()

# Get approximation for noise term
sigma = np.sqrt(df_reg['unex_var'].mean())

list_b_model = []
list_b_patient = []
list_c_model = []
list_c_patient = []

list_p_slopes = []
list_p_intercept = []



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
    
    # Compute linear regression for null 
    out = stats.linregress(vn_vals, vv_vals)
    b_null, c_null, r_null, p_null, se_slope_null = out
    se_inter_null = out.intercept_stderr
    reg_text_null = "y={}x+{},  R^2={}".format(round(b_null, 4), round(c_null, 4), round(r_null**2, 4))
    # print(reg_text_null)
    
    list_b_model.append(b_null)
    list_c_model.append(c_null)

    list_b_patient.append(b_patient)
    list_c_patient.append(c_patient)
    
    # Determine p value for whether patient lin regression is different
    # https://real-statistics.com/regression/hypothesis-testing-significance-regression-line-slope/comparing-slopes-two-independent-samples/
    
    # Test for whether slopes are sign. different
    joint_std_error =  np.sqrt(se_slope_patient**2 + se_slope_null**2)
    test_statistic = (b_patient-b_null)/joint_std_error
    degrees_freedom = len(df_patient) + len(df_null) - 4
    
    t_cdf_val = stats.t.cdf(test_statistic, degrees_freedom)
    # account for 2-tailed t-test
    if t_cdf_val < 0.5:
        p = 2*t_cdf_val
    else:
        p = 2*(1-t_cdf_val)
    list_p_slopes.append(p)
    # print('p value for testing different slopes : {}'.format(p))
 
    
    # Test for whether intercepts are sign. different
    joint_std_error =  np.sqrt(se_inter_patient**2 + se_inter_null**2)
    test_statistic = (c_patient-c_null)/joint_std_error
    degrees_freedom = len(df_patient) + len(df_null) - 4
    t_cdf_val = stats.t.cdf(test_statistic, degrees_freedom)
    # account for 2-tailed t-test
    if t_cdf_val < 0.5:
        p = 2*t_cdf_val
    else:
        p = 2*(1-t_cdf_val)
    list_p_intercept.append(p)
    # print('p value for testing different intercepts : {}'.format(p))

df_pvals = pd.DataFrame()
df_pvals['slope'] = list_p_slopes
df_pvals['intercept'] = list_p_intercept

# lin regression is sig diff if either slope or intercept has p<thresh
p_thresh = 1e-3
df_sig = df_pvals.query('slope < @p_thresh or intercept < @p_thresh')
prop_diff = len(df_sig)/len(df_pvals)
print('% diff = {}'.format(prop_diff))
  

# ar_p_slopes = np.array(list_p_slopes)
# len(ar_p_slopes[ar_p_slopes<0.001])

# print('Proportion of p values that are less than 0.001: {}'.format(
#     len(ar_p_m1[ar_p_m1<0.001])/len(ar_p_m1))
#     )





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


list_b_model = []
list_b_patient = []
list_c_model = []
list_c_patient = []

list_p_slopes = []
list_p_intercept = []


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
    
   
    # Compute linear regression for null 
    out = stats.linregress(vn_vals, vv_vals)
    b_null, c_null, r_null, p_null, se_slope_null = out
    se_inter_null = out.intercept_stderr
    reg_text_null = "y={}x+{},  R^2={}".format(round(b_null, 4), round(c_null, 4), round(r_null**2, 4))
    # print(reg_text_null)
    
    list_b_model.append(b_null)
    list_c_model.append(c_null)

    list_b_patient.append(b_patient)
    list_c_patient.append(c_patient)
    
    # Determine p value for whether patient lin regression is different
    # https://real-statistics.com/regression/hypothesis-testing-significance-regression-line-slope/comparing-slopes-two-independent-samples/
    
    # Test for whether slopes are sign. different
    joint_std_error =  np.sqrt(se_slope_patient**2 + se_slope_null**2)
    test_statistic = (b_patient-b_null)/joint_std_error
    degrees_freedom = len(df_patient) + len(df_null) - 4
    
    t_cdf_val = stats.t.cdf(test_statistic, degrees_freedom)
    # account for 2-tailed t-test
    if t_cdf_val < 0.5:
        p = 2*t_cdf_val
    else:
        p = 2*(1-t_cdf_val)
    list_p_slopes.append(p)
    # print('p value for testing different slopes : {}'.format(p))
 
    
    # Test for whether intercepts are sign. different
    joint_std_error =  np.sqrt(se_inter_patient**2 + se_inter_null**2)
    test_statistic = (c_patient-c_null)/joint_std_error
    degrees_freedom = len(df_patient) + len(df_null) - 4
    t_cdf_val = stats.t.cdf(test_statistic, degrees_freedom)
    # account for 2-tailed t-test
    if t_cdf_val < 0.5:
        p = 2*t_cdf_val
    else:
        p = 2*(1-t_cdf_val)
    list_p_intercept.append(p)
    # print('p value for testing different intercepts : {}'.format(p))

df_pvals_m2 = pd.DataFrame()
df_pvals_m2['slope'] = list_p_slopes
df_pvals_m2['intercept'] = list_p_intercept

# lin regression is sig diff if either slope or intercept has p<thresh
p_thresh = 1e-3
df_sig = df_pvals_m2.query('slope < @p_thresh or intercept < @p_thresh')
prop_diff = len(df_sig)/len(df_pvals_m2)
print('% diff = {}'.format(prop_diff))


#----------------
# Model for EADs:
# VV = b*VN + c + gamma*ts + eps
#----------------


# Import EAD simulation
df_sim = pd.read_csv('output/record_18/df_ead_sim.csv')

list_b_model = []
list_b_patient = []
list_c_model = []
list_c_patient = []

list_p_slopes = []
list_p_intercept = []


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


    # Compute linear regression for EAD model
    vn_vals = df_sim.query('segment==@segment')['VN']
    vv_vals = df_sim.query('segment==@segment')['VV']

    out = stats.linregress(vn_vals, vv_vals)
    b_null, c_null, r_null, p_null, se_slope_null = out
    se_inter_null = out.intercept_stderr
    reg_text_null = "y={}x+{},  R^2={}".format(round(b_null, 4), round(c_null, 4), round(r_null**2, 4))
    # print(reg_text_null)
    
    list_b_model.append(b_null)
    list_c_model.append(c_null)

    list_b_patient.append(b_patient)
    list_c_patient.append(c_patient)
    
    # Determine p value for whether patient lin regression is different
    # https://real-statistics.com/regression/hypothesis-testing-significance-regression-line-slope/comparing-slopes-two-independent-samples/
    
    # Test for whether slopes are sign. different
    joint_std_error =  np.sqrt(se_slope_patient**2 + se_slope_null**2)
    test_statistic = (b_patient-b_null)/joint_std_error
    degrees_freedom = len(df_patient) + len(df_null) - 4
    
    t_cdf_val = stats.t.cdf(test_statistic, degrees_freedom)
    # account for 2-tailed t-test
    if t_cdf_val < 0.5:
        p = 2*t_cdf_val
    else:
        p = 2*(1-t_cdf_val)
    list_p_slopes.append(p)
    # print('p value for testing different slopes : {}'.format(p))
 
    
    # Test for whether intercepts are sign. different
    joint_std_error =  np.sqrt(se_inter_patient**2 + se_inter_null**2)
    test_statistic = (c_patient-c_null)/joint_std_error
    degrees_freedom = len(df_patient) + len(df_null) - 4
    t_cdf_val = stats.t.cdf(test_statistic, degrees_freedom)
    # account for 2-tailed t-test
    if t_cdf_val < 0.5:
        p = 2*t_cdf_val
    else:
        p = 2*(1-t_cdf_val)
    list_p_intercept.append(p)
    # print('p value for testing different intercepts : {}'.format(p))

df_pvals_m3 = pd.DataFrame()
df_pvals_m3['slope'] = list_p_slopes
df_pvals_m3['intercept'] = list_p_intercept


# lin regression is sig diff if either slope or intercept has p<thresh
p_thresh = 1e-3
df_sig = df_pvals_m3.query('slope < @p_thresh or intercept < @p_thresh')
prop_diff = len(df_sig)/len(df_pvals_m3)
print('% diff EAD = {}'.format(prop_diff))

len(df_pvals_m3.query('slope < @p_thresh'))/len(df_pvals_m3)

# ########### Plot to compare b values
# df_plot = pd.DataFrame(
#     {'reentry':list_b_m1,
#       'reentry+variable conduction':list_b_m2,
#       'patient data':list_b_patient,
#       'p_m1':list_p_m1,
#       'p_m2':list_p_m2}
#     )

# df_plot = df_plot.melt(id_vars=['p_m1','p_m2'], value_vars=['patient data','reentry', 'reentry+variable conduction'])

# fig = px.histogram(df_plot, x='value', color='variable', nbins=60)
# fig.update_layout(barmode='overlay')

# # Reduce opacity to see both histograms
# fig.update_traces(opacity=0.75)

# fig.update_layout(width=400,
#                   height=300,
#                   legend=dict(
#                       yanchor="top",
#                       y=0.99,
#                       xanchor="left",
#                       x=0.01,
#                       title='',
#                       ),
#                   margin=dict(l=0, r=10, t=10, b=10)
#                   )

# fig.update_xaxes(title='slope of linear regression')
# fig.update_yaxes(title='count')

# fig.write_html('temp.html')
# fig.write_image('figures/stats_b_comparison.png', scale=2)






# #------------
# # Khady's mod para simulations
# # model 2 - fixed params
# #------------

# list_p_slopes = []
# list_p_intercept = []

# # Import regressions
# filepath = '/Users/tbury/Google Drive/research/postdoc_23/parasystole_paradox/output/khady_output/data_v_models_053023.csv'
# df_reg = pd.read_csv(filepath, index_col=0)
# df_reg['segment'] = np.arange(len(df_reg))



# df_reg['diff2'] = (df_reg['model2 slope'] - df_reg['data slope'])**2
# df_reg['diff3'] = (df_reg['model3 slope'] - df_reg['data slope'])**2

# df_reg['diffint2'] = (df_reg['model2 intercept'] - df_reg['data intercept'])**2
# df_reg['diffint3'] = (df_reg['model3 intercept'] - df_reg['data intercept'])**2




# for segment in df_reg['segment'].unique():
#     # Determine p value for whether patient lin regression is different
#     # https://real-statistics.com/regression/hypothesis-testing-significance-regression-line-slope/comparing-slopes-two-independent-samples/
    
#     df = df_reg.query('segment==@segment')
    
#     b_patient = df['data slope'].iloc[0]
#     b_model2 = df['model2 slope'].iloc[0]
#     b_model3 = df['model3 slope'].iloc[0]
    
#     c_patient = df['data intercept'].iloc[0]
#     c_model2 = df['model2 intercept'].iloc[0]
#     c_model3 = df['model3 intercept'].iloc[0]
    
#     se_b_patient = df['data slope err'].iloc[0]
#     se_b_model2 = df['model2 slope err'].iloc[0]
#     se_b_model3 = df['model3 slope err'].iloc[0]
    
#     se_c_patient = df['data intercept err'].iloc[0]
#     se_c_model2 = df['model2 intercept err'].iloc[0]
#     se_c_model3 = df['model3 intercept err'].iloc[0]
    
#     count_patient = df['count data'].iloc[0]
#     count_model2 = df['count model2'].iloc[0]
#     count_model3 = df['count model3'].iloc[0]
    
    
#     # Test for whether slopes are sign. different
#     joint_std_error =  np.sqrt(se_b_patient**2 + se_b_model2**2)
#     test_statistic = (b_patient-b_model2)/joint_std_error
#     degrees_freedom = count_patient + count_model2 - 4
#     # print(degrees_freedom)
    
#     t_cdf_val = stats.t.cdf(test_statistic, degrees_freedom)
#     # account for 2-tailed t-test
#     if t_cdf_val < 0.5:
#         p = 2*t_cdf_val
#     else:
#         p = 2*(1-t_cdf_val)
#     list_p_slopes.append(p)
#     # print('p value for testing different slopes : {}'.format(p))
 
#     # Test for whether intercepts are sign. different
#     joint_std_error =  np.sqrt(se_c_patient**2 + se_c_model2**2)
#     test_statistic = (c_patient-c_model2)/joint_std_error
#     degrees_freedom = count_patient + count_model3 - 4

#     t_cdf_val = stats.t.cdf(test_statistic, degrees_freedom)
#     # account for 2-tailed t-test
#     if t_cdf_val < 0.5:
#         p = 2*t_cdf_val
#     else:
#         p = 2*(1-t_cdf_val)
#     list_p_intercept.append(p)
#     # print('p value for testing different intercepts : {}'.format(p))

# df_pvals_mp2 = pd.DataFrame()
# df_pvals_mp2['slope'] = list_p_slopes
# df_pvals_mp2['intercept'] = list_p_intercept


# # lin regression is sig diff if either slope or intercept has p<thresh
# p_thresh = 1e-3
# df_sig = df_pvals_mp2.query('slope < @p_thresh or intercept < @p_thresh')
# prop_diff = len(df_sig)/len(df_pvals_mp2)
# print('% diff = {}'.format(prop_diff))


#------------
# Khady's mod para simulations
# model 3 - tlag fit to each segment
#------------

list_p_slopes = []
list_p_intercept = []

# Import regressions
filepath = '/Users/tbury/Google Drive/research/postdoc_23/parasystole_paradox/output/khady_output/data_v_models_053023.csv'
df_reg = pd.read_csv(filepath, index_col=0)
df_reg['segment'] = np.arange(len(df_reg))

for segment in df_reg['segment'].unique():
    # Determine p value for whether patient lin regression is different
    # https://real-statistics.com/regression/hypothesis-testing-significance-regression-line-slope/comparing-slopes-two-independent-samples/
    
    df = df_reg.query('segment==@segment')
    
    b_patient = df['data slope']
    b_model2 = df['model2 slope']
    b_model3 = df['model3 slope']
    
    c_patient = df['data intercept']
    c_model2 = df['model2 intercept']
    c_model3 = df['model3 intercept']
    
    se_b_patient = df['data slope err']
    se_b_model2 = df['model2 slope err']
    se_b_model3 = df['model3 slope err']
    
    se_c_patient = df['data intercept err']
    se_c_model2 = df['model2 intercept err']
    se_c_model3 = df['model3 intercept err']
    
    count_patient = df['count data']
    count_model2 = df['count model2']
    count_model3 = df['count model3']
    
    # Test for whether slopes are sign. different
    joint_std_error =  np.sqrt(se_b_patient**2 + se_b_model3**2)
    test_statistic = (b_patient-b_model3)/joint_std_error
    degrees_freedom = count_patient + count_model3 - 4
    
    t_cdf_val = stats.t.cdf(test_statistic, degrees_freedom)
    # account for 2-tailed t-test
    if t_cdf_val < 0.5:
        p = 2*t_cdf_val
    else:
        p = 2*(1-t_cdf_val)
    list_p_slopes.append(p)
    # print('p value for testing different slopes : {}'.format(p))
 
    # Test for whether intercepts are sign. different
    joint_std_error =  np.sqrt(se_c_patient**2 + se_c_model3**2)
    test_statistic = (c_patient-c_model3)/joint_std_error
    degrees_freedom = count_patient + count_model3 - 4

    t_cdf_val = stats.t.cdf(test_statistic, degrees_freedom)
    # account for 2-tailed t-test
    if t_cdf_val < 0.5:
        p = 2*t_cdf_val
    else:
        p = 2*(1-t_cdf_val)
    list_p_intercept.append(p)
    # print('p value for testing different intercepts : {}'.format(p))

df_pvals_mp3 = pd.DataFrame()
df_pvals_mp3['slope'] = list_p_slopes
df_pvals_mp3['intercept'] = list_p_intercept


# lin regression is sig diff if either slope or intercept has p<thresh
p_thresh = 1e-3
df_sig = df_pvals_mp3.query('slope < @p_thresh or intercept < @p_thresh')
prop_diff = len(df_sig)/len(df_pvals_mp3)
print('% diff = {}'.format(prop_diff))

len(df_pvals_mp3.query('slope < @p_thresh'))/len(df_pvals_mp3)



# #------Random plots

# df_reg['data slope'].plot(kind='hist', xlim=[0.5,1.2])
# df_reg['model1 slope'].plot(kind='hist',xlim=[0.5,1.2])
# df_reg['model2 slope'].plot(kind='hist',xlim=[0.5,1.2])
# df_reg['model3 slope'].plot(kind='hist',xlim=[0.5,1.2])


# df_reg['data intercept'].plot(kind='hist', xlim=[50,200])
# df_reg['model1 intercept'].plot(kind='hist',xlim=[50,200])
# df_reg['model2 intercept'].plot(kind='hist',xlim=[50,200])
# df_reg['model3 intercept'].plot(kind='hist',xlim=[50,200])










