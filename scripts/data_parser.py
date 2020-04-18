import matplotlib
matplotlib.use("Agg") 

import numpy as np
#import torch
import pickle
import seaborn as sns
import pandas as pd
import xarray as xr
import scipy.interpolate
import os
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from sklearn.preprocessing import normalize

'''
This script compiles the dataset of eureka water vapour profiles from 2008 to 2017.
Radiosondes are launched twice (most) days at 12am and 12pm 
In particular it uses data from a (GCOS Reference Upper-Air Network) GRUAN reanalysis.

'''

PARAMS = {"LOAD": True}
def load_data(year):
    '''
    Loads datafiles for a particular year
    Returns a list of pandas dataframes
    '''
    #Data locaed on berg@atmosp.physics.utoronto.ca, not open source though. 
    prepended_dir = '/net/aurora/ground/eureka/radiosondes/GRUAN/' + str(year)

    dataframes = []
    for file in os.listdir(prepended_dir):
        timestamp = file[25:36]
        # Load and discard not required fields -- only keep altitude and water vapour mixing ratio
        dataframes.append(xr.open_dataset(prepended_dir + '/' + file).to_dataframe()[['alt','WVMR']])
        
    return dataframes

if PARAMS["LOAD"]: #Load and save dataframes 

    all_dataframes = []
    years = [2008,2009,2010,2011,2012,2013,2014,2015,2016,2017]
    for year in years:
        print(year)
        all_dataframes += load_data(year)

    if not os.path.exists('pickle'):
        os.makedirs('pickle')

    file_handler = open('./pickle/data.obj', 'wb')
    pickle.dump(all_dataframes, file_handler)

else:
    file_handler = open('./pickle/data.obj', 'rb')
    all_dataframes = pickle.load(file_handler)

#Interpolate onto this grid
interp_grid = np.linspace(0,15000,500)

data = [] 
for df in all_dataframes:
    extracted_alts = df['alt'].values.astype(float)
    extracted_wvmr = df['WVMR'].values.astype(float)
    try:
        interp_func = scipy.interpolate.interp1d(extracted_alts, extracted_wvmr,fill_value="extrapolate", kind="cubic")
        interp_wvmr = interp_func(interp_grid)
        data.append(interp_wvmr)

    except Exception: #sometimes heights are not in sorted order, just ignore as it doesn't occur often
        pass

# A bit of preprocessing to remove NaN values and normalize data
data = np.array(data)
data = np.nan_to_num(data,copy=False)
data = np.clip(data,a_min=0,a_max=1.5)
data = normalize(data)
nonzero = []
for i,v in enumerate(data):
    if not np.all((v == 0)):
        nonzero.append(v)
data = np.array(nonzero)

data_pts = len(data)

# Define train/test sets at a 90/10 split
training_data = data[:-int(np.ceil(data_pts*0.1))]
testing_data = data[-int(np.ceil(data_pts*0.1)):]


file_handler = open('./pickle/training.obj', 'wb')
pickle.dump(training_data,file_handler)
file_handler_2 = open('./pickle/testing.obj','wb')
pickle.dump(testing_data,file_handler_2)

print("Training data has shape",training_data.shape)
print("Testing data has shape",testing_data.shape)

#Sample plot to show two sample interpolated profiles in summer and winter
plt.title(str(np.shape(training_data)[1]) + " point interpolation")
plt.plot(training_data[1],interp_grid,label="Summer Interpolated")
plt.plot(testing_data[2],interp_grid,label="Winter Interpolated")
plt.xlabel("Water Vapour Mixing Ratio")
plt.ylabel("Altitude (m)")
plt.legend()
plt.savefig('test1')
