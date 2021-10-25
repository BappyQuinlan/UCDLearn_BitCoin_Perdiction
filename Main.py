# Main Execution File
# Author : Barry Quinlan
# Date : 25th October 2021
# Email : bappyquinlan@gmail.com

# Import required library's for the program
# Install Pandas, numpy, matplotlib, sklearn, plotly, xgboost and tensorflow for keras

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
import sklearn.metrics as metrics
from sklearn.metrics import make_scorer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
import plotly.express as px
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense,LSTM
from sklearn.metrics import mean_absolute_error

def regression_results(y_true, y_pred):
    # Calculate regression metrics for the predictions on different models
    # Regression metrics
    explained_variance = metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    mean_squared_log_error = metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error = metrics.median_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    print('explained_variance: ', round(explained_variance, 4))
    print('mean_squared_log_error: ', round(mean_squared_log_error, 4))
    print('median_absolute_error: ', round(median_absolute_error, 4))
    print('r2: ', round(r2, 4))
    print('MAE: ', round(mean_absolute_error, 4))
    print('MSE: ', round(mse, 4))
    print('RMSE: ', round(np.sqrt(mse), 4))


def training_data_split(input_dataframe, split_point, column_name):
    #Create the require training split testing split based on timeseries data
    split_point1 = str(split_point)
    split_point2 = str(split_point + 1)
    X_train1 = input_dataframe.loc[:split_point1].drop([column_name], axis=1)
    y_train1 = input_dataframe.loc[:split_point1, column_name]
    X_test1 = input_dataframe.loc[split_point2].drop([column_name], axis=1)
    y_test1 = input_dataframe.loc[split_point2, column_name]
    return X_train1, y_train1, X_test1, y_test1

def working_dataframe(input_dataframe, new_column, column_to_copy):
    # creating new dataframe from closing price column
    df_closed = input_dataframe[[column_to_copy]].copy()
    # inserting new column with yesterday's closed values
    df_closed[new_column+'_Close'] = df_closed.loc[:, column_to_copy].shift()
    # inserting another column with difference between yesterday and day before yesterday's consumption values.
    df_closed.loc[:, new_column+'_Diff'] = df_closed.loc[:, new_column+'_Close'].diff()
    # dropping NAs
    df_closed = df_closed.dropna()
    return df_closed

def rmse(actual, predict):
    predict = np.array(predict)
    actual = np.array(actual)
    distance = predict - actual
    square_distance = distance ** 2
    mean_square_distance = square_distance.mean()
    score = np.sqrt(mean_square_distance)
    return score


rmse_score = make_scorer(rmse, greater_is_better=False)

df = pd.read_csv('Input/BTC-USD.csv')
df['Date'] = pd.to_datetime(df.Date, format='%Y-%m-%d')
df.index = df['Date']

df_closed = working_dataframe(df, 'Yesterday', 'Close')

X_train, y_train, X_test, y_test = training_data_split(df_closed, 2019, 'Close')

models = []
models.append(('LR', LinearRegression()))
models.append(('NN', MLPRegressor(solver='lbfgs', max_iter=1000)))  # neural network
models.append(('KNN', KNeighborsRegressor(n_neighbors=3)))
models.append(('RF', RandomForestRegressor(n_estimators=10)))  # Ensemble method - collection of many decision trees
models.append(('SVR', SVR(gamma='auto')))  # kernel = linear
# Evaluate each model in turn
results = []
names = []
for name, model in models:
    # TimeSeries Cross validation
    tscv = TimeSeriesSplit(n_splits=10)

    cv_results = cross_val_score(model, X_train.values, y_train.values, cv=tscv, scoring='r2')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Compare Algorithms
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()

model = KNeighborsRegressor()

# Define our candidate hyperparameters
params = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9]}
tscv = TimeSeriesSplit(n_splits=10)
gsearch = GridSearchCV(model, cv=tscv, param_grid=params, scoring=rmse_score)
gsearch.fit(X_train.values, y_train.values)
best_score = gsearch.best_score_
best_model = gsearch.best_estimator_

y_true = y_test.values
y_pred = best_model.predict(X_test.values)
print('KNeighbours Regression')
regression_results(y_true, y_pred)

valid = X_test
valid['Predictions'] = best_model.predict(X_test)

fig = px.line(valid, y=['Yesterday_Close' ,'Predictions'], title='Bitcoin Prediction Price Change KNeighbours Model ')
fig.show()


df_closed_2o = working_dataframe(df, 'Yesterday', 'Close')

X_train_2o, y_train_2o, X_test, y_test = training_data_split(df_closed_2o, 2019, 'Close')

model = RandomForestRegressor()
param_search = {
    'n_estimators': [20, 50, 100],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [i for i in range(5, 15)]
}
tscv = TimeSeriesSplit(n_splits=10)
gsearch = GridSearchCV(estimator=model, cv=tscv, param_grid=param_search, scoring=rmse_score)
gsearch.fit(X_train_2o, y_train_2o)
best_score = gsearch.best_score_
best_model = gsearch.best_estimator_
y_true = y_test.values
y_pred = best_model.predict(X_test)
print('Random Forest Regressor')
regression_results(y_true, y_pred)

valid = X_test
valid['Predictions'] = best_model.predict(X_test)

fig = px.line(valid, y=['Yesterday_Close' ,'Predictions'], title='Bitcoin Prediction Price Change Random Forest Model')
fig.show()

X_train_2, y_train_2, X_test_2, y_test_2 = training_data_split(df_closed_2o, 2019, 'Close')

model2 = XGBClassifier()
model2.fit(X_train_2, y_train_2)

y_pred2 = model2.predict(X_test_2.values)
predictions2 = [round(value) for value in y_pred2]

df_deep_learning = working_dataframe(df, 'Yesterday', 'Close')

# Create a new dataframe with only the 'Close column
scaler = MinMaxScaler(feature_range=(0, 1))

# Convert the dataframe to a numpy array
dataset_4 = df_deep_learning.filter(['Close']).copy().values

# Get the number of rows to train the model on
training_data_len = int(np.ceil( len(dataset_4) * .95 ))

scaled_data = scaler.fit_transform(dataset_4)

train_data = scaled_data[0:int(training_data_len), :]
# Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i - 60:i, 0])
    y_train.append(train_data[i, 0])

# Convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(32))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')


callbacks = [EarlyStopping(patience=4, monitor='loss', mode='min'),
             ReduceLROnPlateau(patience=2, verbose=1)]

history =model.fit(x_train, y_train,
                        epochs=20,
                        batch_size=10,
                        callbacks=callbacks,
                        validation_data=(x_train, y_train)
                        )

# Create the testing data set
# Create a new array containing scaled values from index 1543 to 2002
test_data = scaled_data[training_data_len - 60:, :]
# Create the data sets x_test and y_test
x_test = []
y_test = dataset_4[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i - 60:i, 0])

# Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE)
y_true = y_test
y_pred = predictions
print('Keras Model')
regression_results(y_true, y_pred)

train4 = df_deep_learning[:training_data_len]
valid4 = df_deep_learning[training_data_len:].copy()
valid4['Predictions'] = predictions

fig = px.line(valid4, y=['Close','Predictions'], title='Bitcoin Prediction Price Change Keras Model')
fig.show()

