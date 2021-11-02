# multivariate multi-step encoder-decoder lstm
from math import sqrt
import numpy as np
from numpy import split
from numpy import array

from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense

from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))


# split dataset into training and testing sets
def split_dataset(data):
    train, test = data[:216,: ], data[216:, :]

    #training and testing dataset are normalized
    train = scaler.fit_transform(train)
    test = scaler.fit_transform(test)
    train_scaled = array(split(train, 18))
    test_scaled = array(split(test, 3))
    return train_scaled, test_scaled


# evaluate one or more monthly forecasts against expected values
def evaluate_forecasts(actual, predicted,inv_yhat,inv_y):
    # # scores = list()
    # # calculate an RMSE score for each month
    # # for i in range(actual.shape[1]):
    # # calculate mse
    #     mse = mean_squared_error(actual[:, i], predicted[:, i])
    # # calculate rmse
    #     rmse = sqrt(mse)
    # # store
    #     scores.append(rmse)
    # # calculate overall RMSE
    #     s = 0
    # for row in range(actual.shape[0]):
    #     for col in range(actual.shape[1]):
    #         # s = (mean_squared_error(inv_y, inv_yhat))
    #         s += (actual[row, col] - predicted[row, col]) ** 2
    # score = sqrt(s / (actual.shape[0] * actual.shape[1]))
    #return score, scores
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    return rmse


# summarize scores
def summarize_scores(name, score, scores):
     s_scores = ', '.join(['%.1f' % s for s in scores])
     print('%s: [%.3f] %s' % (name, score, s_scores))


# convert history into inputs and outputs
def to_supervised(train_scaled, n_input, n_out=36):
    # flatten data
    data = train_scaled.reshape((train_scaled.shape[0] * train_scaled.shape[1], train_scaled.shape[2]))
    X, y = list(), list()
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end <= len(data):
            X.append(data[in_start:in_end, :])
            y.append(data[in_end:out_end, 0])
        # move along one time step
        in_start += 1
    return array(X), array(y)


# train the model # Model returns different values each time when run
def build_model(train_scaled, n_input):
    # prepare data
    (train_scaled_x), (train_scaled_y) = to_supervised(train_scaled, n_input)
    # define parameters
    verbose, epochs, batch_size = 0, 100, 12
    n_timesteps, n_features, n_outputs = train_scaled_x.shape[1], train_scaled_x.shape[2], train_scaled_y.shape[1]
    # reshape output into [samples, timesteps, features]
    train_scaled_y = train_scaled_y.reshape((train_scaled_y.shape[0], train_scaled_y.shape[1], 1))
    # define model
    model = Sequential()
    model.add(LSTM(4, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(RepeatVector(n_outputs))
    model.add(LSTM(4, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(12, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mse', optimizer='adam')
    # fit network
    model.fit(train_scaled_x, train_scaled_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model


# make a forecast
def forecast(model, history, n_input):
    # flatten data
    data = array(history)
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
    # retrieve last observations for input data
    input_x = data[-n_input:, :]
    # reshape into [1, n_input, n]
    input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
    # forecast the next month
    yhat = model.predict(input_x, verbose=0)
    # vector forecast value
    yhat = yhat[0]
    input_y = data[-n_input:, 0 ]

    inv_yhat = np.concatenate((yhat, input_x[0,:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
    # invert scaling for actual
    input_y = input_y.reshape((len(input_y), 1))
    inv_y = np.concatenate((input_y, input_x[0, :, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]

    return yhat, inv_yhat, inv_y


# evaluate the model
def evaluate_model(train_scaled, test_scaled, n_input):
    # fit model
    model = build_model(train_scaled, n_input)
    # history is a list of monthly data
    history = [x for x in train_scaled]

    # walk-forward validation over each month
    predictions = list()
    for i in range(len(test_scaled)):
        # predict the month
        yhat_sequence, inv_yhat, inv_y = forecast(model, history, n_input)
        # store the predictions
        predictions.append(yhat_sequence)
        # get real observation and add to history for predicting the next month
        history.append(test_scaled[i, :])
    # evaluate predictions for each month
    predictions = array(predictions)
    # score, scores = evaluate_forecasts(test_scaled[:, :, 0], predictions,inv_yhat,inv_y)
    rmse = evaluate_forecasts(test_scaled[:, :, 0], predictions, inv_yhat, inv_y)
    return rmse

# load the new file
dataset = read_csv(r'C:\Users\jaylu\Desktop\Projects\CP2\Dummy Data\DummyData.csv',
                   header=0, infer_datetime_format=True, parse_dates={'datetime': [0]}, index_col=['datetime'])

dataset = dataset.astype('float32')

# split into train and test
train_scaled, test_scaled = split_dataset(dataset.values)
# evaluate model and get scores
n_input = 36
# score, scores = evaluate_model(train_scaled, test_scaled, n_input)
rmse = evaluate_model(train_scaled, test_scaled, n_input)
print(rmse)
# summarize scores
# summarize_scores('lstm', rmse)
# plot scores
# year = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']

# pyplot.plot(year, scores, marker='o', label='lstm')
# pyplot.show()