import numpy as np
import pickle
import seaborn as sns
import pandas as pd
import xarray as xr
import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize

def compile_pairs(sequences, input_steps, forecast_steps):
    '''Compiles dataset into (input_profiles, correct_forecast) sets. 
    
    input_profiles is a a list of (input_steps) number of elements
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
 
in_dim = 500
in_profile,out_profile = 30,1

file_handler_1 = open('./pickle/testing.obj', 'rb')
testing_data = pickle.load(file_handler_1)

file_handler_4 = open('./pickle/training.obj', 'rb')
training_data = pickle.load(file_handler_4)

file_handler_2 = open('./pickle/600_150epoch.obj','rb')
model_600_150epoch = pickle.load(file_handler_2)

file_handler_3 = open('./pickle/1000_350epoch.obj','rb')
model_1000_350epoch = pickle.load(file_handler_3)

#Compile pairs for testing
test_x, test_y = compile_pairs(testing_data,in_profile,out_profile)
grid = np.linspace(0,15000,500)

pred_350, pred_150, = [], []


#Reshape testing pairs and run through model
for sett in test_x:
    pred_350.append(model_1000_350epoch.predict(sett.reshape((1,in_profile,in_dim))))
    pred_150.append(model_600_150epoch.predict(sett.reshape((1,in_profile,in_dim))))

#Plot comarisons between different training lengths
for i, inp in enumerate(pred_350):
    plt.title("Comparison between True Profile and Generated Profiles at time " + str(i))
    plt.plot(inp.reshape((500,)),grid, marker='1',label="350_epoch_trained")
    plt.xlabel("WVMR")
    plt.ylabel("Altitude (m)")
    plt.plot(pred_150[i].reshape((500,)),grid, marker='2',label="150_epoch_trained")
    plt.plot(test_y[i].reshape((500,)),grid,label="actual")
    plt.legend()
    plt.savefig("./plots/"+ str(i)+ "_predicted.png")
    plt.clf()


#Now use the generated images to forecast beyond a single point
predicted_set = training_data[:30]
window = predicted_set
for i in range(len(testing_data)):
    print(i)
    predicted = model_1000_350epoch.predict(window.reshape((1,in_profile,in_dim)))
    print("new", np.shape(window))
    print(np.shape(predicted))
    window = np.append(window[1:], predicted.reshape((1,in_dim)),axis=0)
    print(np.shape(window))
    plt.title("Comparison between True Profile and Forecasted Profiles at time " + str(i))
    plt.plot(predicted.reshape((500,)),grid, marker='1',label="Forecasted")
    plt.xlabel("WVMR")
    plt.ylabel("Altitude (m)")
    plt.plot(testing_data[i].reshape((500,)),grid,label="Actual")
    plt.legend()
    plt.savefig("./plots/forecasted/"+ str(i)+ "_forecasted.png")
    plt.clf()