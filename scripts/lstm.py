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
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras import regularizers


# Load preprocessed data files
file_handler = open('./pickle/training.obj', 'rb')
training_data = pickle.load(file_handler)

file_handler_2 = open('./pickle/testing.obj','rb')
testing_data = pickle.load(file_handler_2)

print(np.shape(training_data))
print(type((training_data)))
print(training_data.shape)


def compile_pairs(sequences, input_steps, forecast_steps):
    '''Compiles dataset into (training_profiles, correct_forecast) sets. 
    
    training_profiles is a a list of (input_steps) number of elements
    correct_forecast is a list of (forecast_steps) number of elements
    '''
    X, y = [], []
    for i in range(len(sequences)):
        end_seq = i + input_steps
        end_seq_stp = end_seq + forecast_steps
        if end_seq_stp > len(sequences): 
            break
        inp,out = sequences[i:end_seq, :], sequences[end_seq:end_seq_stp, :]
        X.append(inp)
        y.append(out)
    return np.array(X), np.array(y)
 
input_steps, forecast_steps = 30,1

X, y = compile_pairs(training_data, input_steps, forecast_steps)


n_features = X.shape[2]

# define model
# If we want regularization, added the flag activity_regualrizer to the LSTM module) 
model = Sequential()
model.add(LSTM(1000, activation='relu', input_shape=(input_steps, n_features)))#, activity_regularizer=regularizers.l2(0.01)))
model.add(RepeatVector(forecast_steps))
model.add(LSTM(1000, activation='relu', return_sequences=True))#,activity_regularizer=regularizers.l2(0.01)))
model.add(TimeDistributed(Dense(n_features)))

#If loss is too large, change flag to 'msle' to convert computation to log space
model.compile(optimizer='adam', loss='mse')


# Fit model, this takes a while, especially without GPU
model.fit(X, y, epochs=700, verbose=1,batch_size=80)

# demonstrate prediction
x_input = training_data[-30:]

#Turn into expected format
x_input = x_input.reshape((1, input_steps, n_features)) 
predicted = model.predict(x_input, verbose=1)

#Save model and sample output
file_handler_m = open('./pickle/trained_model.obj','wb')
file_handler = open('./pickle/sample_predicted.obj','wb')

pickle.dump(predicted,file_handler)
pickle.dump(model,file_handler_m)

