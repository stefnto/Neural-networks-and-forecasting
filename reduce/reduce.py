import math, sys, getopt
import tensorflow
import datetime
import time
import requests as req
import json
import pandas as pd
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, LSTM, RepeatVector
from keras.models import Model
from keras.models import model_from_json
from keras import regularizers
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

argumentList = sys.argv[1:]

# run python reduce.py -d nasdaq2007_17.csv -o timeseries.csv
options = "d:q:"
long_options = ["od=", "oq="]
d_flag, q_flag, od_flag, oq_flag = 1, 1, 1, 1
dataset = ''
queryset = ''
output_dataset_file = ''
output_query_file = ''

# Parse the arguments
try:
    arguments, values = getopt.getopt(argumentList, options, long_options)
except getopt.error as err:
    print (str(err))

# Check each argument
for currentArgument, currentValue in arguments:
    if currentArgument == '-d':
        dataset = currentValue
        d_flag = 0
    elif currentArgument == '-q':
        queryset = currentValue
        q_flag = 0
    elif currentArgument == "--od":
        output_dataset_file = currentValue
        od_flag = 0
    elif currentArgument == "--oq":
        output_query_file = currentValue
        oq_flag = 0


if (d_flag == 1 or q_flag == 1 or od_flag == 1 or oq_flag == 1):
    sys.exit("Argument -d or -o or --od or --oq not passed, terminating...")


window_length = 10
encoding_dim = 3
epochs = 100
test_samples = 364 #364

# Read .csv files
df = pd.read_csv(dataset, header=None, sep = '\t')
df_q = pd.read_csv(queryset, header=None, sep = '\t')

dataset_ = df.iloc[:, 1:].to_numpy()
queryset_ = df_q.iloc[:, 1:].to_numpy()

# df_data is a list containing a data_frame for each stock with columns price, percentage_change, log_difference
df_data = []
for i in range(dataset_.shape[0]):
    tmp = dataset_[i].reshape(3650,1)
    df_data.append(pd.DataFrame(tmp, columns=['price']))

for i in range(queryset_.shape[0]):
    tmp = queryset_[i].reshape(3650,1)
    df_data.append(pd.DataFrame(tmp,columns=['price']))


# Holds 359 np.arrays of shape (3275, 10, 1)
x_train_scaled_data = []

# Holds 359 np.arrays of shape (364, 10, 1)
x_test_scaled_data = []

# Holds 359 scalers, each for each stock
scalers_data = []

if not os.path.isfile("x_train_scaled_data.pickle"):
    for i in range(len(df_data)):

        x_train_scaled = []
        scalers = []

        # Initialize scaler
        scaler = MinMaxScaler(feature_range = (0,1))

        stock = df_data[i]['price'].values

        # Scale the values of the stock shape (3650,1)
        scaled_stock = scaler.fit_transform(df_data[i]['price'].values.reshape(-1,1))

        # Save the scaler for the stock in the scalers_data list, scaler takes shape (3650,1)
        obj = scaler.fit(df_data[i]['price'].values.reshape(-1,1))
        scalers_data.append(obj)

        # Put values in windows of '10', 365 windows in total
        for k in tqdm(range(1, df_data[i].shape[0], window_length)):
            tmp = scaled_stock[k-1:k+window_length-1].reshape(-1,1)
            x_train_scaled.append(tmp)

        # Transform list to an numpy array, shape (365,10,1)
        x_train_scaled = np.array(x_train_scaled)

        # Get last 10% of x_train_scaled to use as test_set, shape (37,10,1)
        x_test_scaled = x_train_scaled[(math.floor(x_train_scaled.shape[0]*0.9)):]

        # Get first 90% of x_train_scaled to use as train_set, shape (328,10,1)
        x_train_scaled = x_train_scaled[:(math.floor(x_train_scaled.shape[0]*0.9))]

        # Cast values in array to float
        x_test_scaled = x_test_scaled.astype('float32')

        # Cast values in array to float
        x_train_scaled = x_train_scaled.astype('float32')

        # Add the np.array to the list x_train_scaled_data
        x_train_scaled_data.append(x_train_scaled)

        # Add the np.array to the list x_test_scaled_data
        x_test_scaled_data.append(x_test_scaled)

    # Save each scaled_data list for reuse
    pickle.dump(x_train_scaled_data, open("x_train_scaled_data.pickle", "wb"))
    pickle.dump(x_test_scaled_data, open("x_test_scaled_data.pickle", "wb"))
    pickle.dump(scalers_data, open("scalers_data.pickle", "wb"))

    print("Made pickle files")

else:
    x_train_scaled_data = pickle.load(open("x_train_scaled_data.pickle", "rb"))
    x_test_scaled_data = pickle.load(open("x_test_scaled_data.pickle", "rb"))
    scalers_data = pickle.load(open("scalers_data.pickle", "rb"))

    print("Loaded pickle files")

x_test_deep = x_test_scaled_data[0].reshape((len(x_test_scaled_data[0]), np.prod(x_test_scaled_data[0].shape[1:])))

# 1D Convolutional autoencoder
if not os.path.isdir("autoencoder"):
    input_window = Input(shape=(window_length,1))

    # Encoder --- 4 layers
    x = Conv1D(16, 3, activation="relu", padding="same")(input_window)

    x = MaxPooling1D(2, padding="same")(x)

    x = Conv1D(1,3,activation="relu", padding="same")(x)

    encoded = MaxPooling1D(2,padding="same")(x)

    encoder = Model(input_window, encoded)


    # Decoder --- 5 layers
    x = Conv1D(1,3,activation="relu", padding="same")(encoded) # 3 dims

    x = UpSampling1D(2)(x) # 6 dims

    x = Conv1D(16,2,activation="relu")(x) # 5 dims

    x = UpSampling1D(2)(x) # 10 dims

    decoded = Conv1D(1,3,activation="sigmoid", padding="same")(x) # 10 dims

    autoencoder = Model(input_window, decoded)

    encoder.compile(optimizer="adam", loss="binary_crossentropy")
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")


    for i in range(30):

        print("Training with ", i, "th stock ")

        history = autoencoder.fit(x_train_scaled_data[i], x_train_scaled_data[i],
            epochs=epochs,
            batch_size=1024,
            shuffle=True,
            validation_data=(x_test_scaled_data[i], x_test_scaled_data[i]))

    autoencoder.save("autoencoder")
    encoder.save("encoder")

    # Save history in .history file
    pickle.dump(history.history, open(history_str, "wb"))

    print("Training done")

else:
    autoencoder = keras.models.load_model("autoencoder")
    encoder = keras.models.load_model("encoder")
    print("Loaded trained models")

# Writing to output_dataset_file
for j in range(dataset_.shape[0]):
    scaler = scalers_data[j]

    stock_scaled = np.concatenate((x_train_scaled_data[j], x_test_scaled_data[j]), axis=0)
    stock_unscaled = scaler.inverse_transform(stock_scaled.reshape(365,10))

    # Make the prediction
    encoded_stock_scaled = encoder.predict(stock_scaled)

    # Reshape in order to unscale the prediction
    encoded_stock_scaled_reshaped = encoded_stock_scaled.reshape(365, 3)

    # Unscale to prepare to write to .csv file
    encoded_stock_unscaled = scaler.inverse_transform(encoded_stock_scaled_reshaped)

    # Changing numpy array to DataFrame in order to save to .csv file
    d_fr = pd.DataFrame(encoded_stock_unscaled.reshape(-1,1).reshape(1,-1))
    d_fr.loc[0,-1] = df.iloc[j,0]
    d_fr.index = d_fr.index + 1  # shifting index
    d_fr = d_fr.sort_index(axis=1)  # sorting by index

    # Append to .csv file
    d_fr.to_csv(output_dataset_file, index=False, sep = '\t', header=False, mode='a')


# Writing to output_query_file
for j in range(queryset_.shape[0]):
    scaler = scalers_data[349+j]

    stock_scaled = np.concatenate((x_train_scaled_data[349+j], x_test_scaled_data[349+j]), axis=0)
    stock_unscaled = scaler.inverse_transform(stock_scaled.reshape(365,10))

    # Make the prediction
    encoded_stock_scaled = encoder.predict(stock_scaled)

    # Reshape in order to unscale the prediction
    encoded_stock_scaled_reshaped = encoded_stock_scaled.reshape(365, 3)

    # Unscale to prepare to write to .csv file
    encoded_stock_unscaled = scaler.inverse_transform(encoded_stock_scaled_reshaped)

    # Changing numpy array to DataFrame in order to save to .csv file
    d_fr = pd.DataFrame(encoded_stock_unscaled.reshape(-1,1).reshape(1,-1))
    d_fr.loc[0,-1] = df_q.iloc[j,0]
    d_fr.index = d_fr.index + 1  # shifting index
    d_fr = d_fr.sort_index(axis=1)  # sorting by index

    # Append to .csv file
    d_fr.to_csv(output_query_file, index=False, sep = '\t', header=False, mode='a')

print("Writing to output files done")


# Code to print plot for 3rd query timeseries
j = 351
k = 3
scaler = scalers_data[j]

stock_scaled = np.concatenate((x_train_scaled_data[j], x_test_scaled_data[j]), axis=0)
stock_unscaled = scaler.inverse_transform(stock_scaled.reshape(365,10))
encoded_stock_scaled = encoder.predict(stock_scaled)
encoded_stock_scaled_reshaped = encoded_stock_scaled.reshape(365, 3)
encoded_stock_unscaled = scaler.inverse_transform(encoded_stock_scaled_reshaped)

time_spots = np.array([*range(0, 365*3, 1)])
time_spots1 = np.array([*range(0, 365*10, 1)])

# print(type(encoded_stock_unscaled.reshape(-1,1)))
# print(type(stock_unscaled.reshape(-1,1)))

plt.plot(time_spots, encoded_stock_unscaled.reshape(-1,1), color = "blue", label = """ '%s' Encoded""" % (df_q.iloc[k-1,0]))
plt.plot(time_spots1, stock_unscaled.reshape(-1,1), color = "red", label = """ '%s' Real""" % (df_q.iloc[k-1,0]))

plt.legend()
plt.show()
