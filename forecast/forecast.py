import getopt, sys, random
import math
import tensorflow
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping



argumentList = sys.argv[1:]

random.seed(datetime.now())
options = "d:n:"
dataset = ''
n_of_timeseries = ''
d_flag, n_flag = 1, 1
rand_nums = []


try:
    arguments, values = getopt.getopt(argumentList, options)
except getopt.error as err:
    print (str(err))

for currentArgument, currentValue in arguments:
    if currentArgument == '-d':
        dataset = currentValue
        d_flag = 0
    if currentArgument == '-n':
        n_of_timeseries = int(currentValue)
        n_flag = 0

if (d_flag == 1 or n_flag == 1):
    sys.exit("Argument -d or -n not passed, terminating...")

# Choose randomly the stocks to predict
for i in range(n_of_timeseries):
    rand_nums.append(random.randint(0, 359))


# Read .csv file
df = pd.read_csv("nasdaq2007_17.csv", header=None, sep = '\t')

# Get 80% of each timeseries to use as training_set
dataset = df.iloc[:, 1:].to_numpy()

training_set = dataset[:30, :(math.floor(dataset.shape[1]*0.8))]

# Get the remaining 20% of each timeseries for testing
test_set = dataset[:, (math.floor(dataset.shape[1]*0.8)):]

# Initialize scaler
scaler = MinMaxScaler(feature_range = (0,1))

# Scale training and test sets
training_set_scaled = scaler.fit_transform(training_set)

test_set_scaled = scaler.fit_transform(test_set)

X_train = np.reshape(training_set_scaled, (training_set_scaled.shape[0], training_set_scaled.shape[1], 1))

str = "training"

Model trained from the first 30 stocks of the dataset
# Create and initialize training model
model = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(units = 25, return_sequences = True, input_shape = (60, 1)))
model.add(Dropout(0.3))

# Adding a second LSTM layer and some Dropout regularisation
model.add(LSTM(units = 25, return_sequences = True))
model.add(Dropout(0.3))

# Adding a third LSTM layer and some Dropout regularisation
model.add(LSTM(units = 25, return_sequences = True))
model.add(Dropout(0.3))

# Adding a fourth LSTM layer and some Dropout regularisation
model.add(LSTM(units = 25))
model.add(Dropout(0.3))

# Adding the output layer
model.add(Dense(units = 1))

# Compiling the RNN
opt = keras.optimizers.Adam(learning_rate = 0.003)
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Training the model with each timeseries
count = 1
for timeseries in training_set_scaled:
    count = count+1

    X_train = []
    y_train = []

    for i in range(60, 2920):
        X_train.append(timeseries[i-60:i])
        y_train.append(timeseries[i])

    X_train, y_train = np.array(X_train), np.array(y_train)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model.fit(X_train, y_train, epochs = 10, batch_size = 100)

model.save(str)


model = keras.models.load_model(str)

predicted_stocks = []

# Predict the last 20% of each stock
for k in range(n_of_timeseries):
    n = rand_nums[k]

    # Get the first 80% and the remaining 20% of the stock respectivly
    stock_train = df.iloc[n-1, 1:(math.floor(dataset.shape[1]*0.8)+1)]
    stock_test = df.iloc[n-1, (math.floor(dataset.shape[1]*0.8)+1):]
    stock = df.iloc[n-1].to_numpy()

    # Reshape and scale input
    stock_values = pd.concat((stock_train, stock_test), axis = 0)
    inputs = stock_values[len(stock_values) - len(stock_test) - 60:].to_numpy()
    inputs = inputs.reshape(-1,1)
    inputs = scaler.fit_transform(inputs)

    X_test = []
    for i in range(60, 790):
        X_test.append(inputs[i-60:i, 0])

    # X_test is of shape (730, 60, 1)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predicted_stock_price = model.predict(X_test)

    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
    predicted_stocks.append(predicted_stock_price)


time_spots = np.array([*range(3650-730, 3650, 1)])

for k in range(n_of_timeseries):
    n = rand_nums[k]

    # Get the real values of the said timeseries
    stock_test = df.iloc[n-1, (math.floor(dataset.shape[1]*0.8)+1):]
    tmp_plot = plt.subplot(n_of_timeseries, 1, k+1)

    # Make the plots
    tmp_plot.plot(time_spots, stock_test.to_numpy(), color = "red", label = ("""Stock '%s' Real Values""" % (df.iloc[n-1][0])))
    tmp_plot.plot(time_spots, predicted_stocks[k], color = "blue", label = ("""Stock '%s' Predicted Values""" % (df.iloc[n-1][0])))
    tmp_plot.set_xticks(np.arange(3650-730, 3650, 50))
    tmp_plot.set_xlabel("Time Spots")
    tmp_plot.set_ylabel("Stock Price")
    tmp_plot.legend()

plt.show()
