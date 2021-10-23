# Main Execution File
# Author : Barry Quinlan
# Date : 22nd October 2021
# Email : bappyquinlan@gmail.com

# import Required Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from fastai.tabular.all import *
import plotly.offline as po
from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping , ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, LSTM

scaler = MinMaxScaler(feature_range=(0, 1))

# read the file
df = pd.read_csv('Input/BTC-USD.csv')

# print the head
print(df.head())

# setting index as date
df['Date'] = pd.to_datetime(df.Date, format='%Y-%m-%d')
df.index = df['Date']

print(df.isnull().sum())
df = df.dropna()
print(df.isnull().sum())

fig = px.line(df, x='Date', y='Close', title='Bitcoin Market Price Change')
fig.show()

# looking at the first five rows of the data
print(df.head())
print('\n Shape of the data:')
print(df.shape)


# Creating the required dataframe with date and our target output

def create_copy_dataframe_for_analysis(input_dataframe, column1, column2):
    data = input_dataframe.copy().sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0, len(df)), columns=[column1, column2]).copy()

    for i in range(0, len(data)):
        new_data[column1][i] = data[column1][i]
        new_data[column2][i] = data[column2][i]

    return new_data


new_df = create_copy_dataframe_for_analysis(df, 'Date', 'Close')

new_df['Close'] = pd.to_numeric(new_df.Close)


# Create Training and Test Data from the new data frame

# creating train and test sets
def training_and_test_datasets(input_data):
    dataset = input_data.values
    split = round(len(dataset) / 2)
    train = input_data.copy().loc[:split]
    valid = input_data.copy().loc[split - 1:]
    return train, valid, dataset


train, valid, dataset = training_and_test_datasets(new_df)

# shapes of training set
print('\n Shape of training set:')
print(train.shape)

# shapes of validation set
print('\n Shape of validation set:')
print(valid.shape)

# shape of dataset set
print('\n Shape of dataset set:')
print(dataset.shape)

# In the next step, we will create predictions for the validation sets and check the RMSE using the actual values.
# making predictions
preds = []

for i in range(0, valid.shape[0]):
    a = train['Close'][len(train) - 248 + i:].sum() + sum(preds)
    b = a / 248
    preds.append(b)

print(preds)

# checking the results (RMSE value)
rms = np.sqrt(np.mean(np.power((np.array(valid['Close']) - preds), 2)))
print('\n RMSE value on validation set:')
print(rms)

valid['Predictions'] = 0
valid['Predictions'] = preds

fig1 = px.line(valid, x='Date', y=['Close', 'Predictions'], title='Bitcoin Prediction Price Change')
#fig1.show()


# 2nd Model

new_df_2 = new_df.copy()
add_datepart(new_df_2, 'Date')
new_df_2.drop('Elapsed', axis=1, inplace=True)  # elapsed will be the time stamp

train_2, valid_2, dataset_2 = training_and_test_datasets(new_df_2)

x_train_2 = train_2.drop('Close', axis=1)
y_train_2 = train_2['Close']
x_valid_2 = valid_2.drop('Close', axis=1)
y_valid_2 = valid_2['Close']

model = LinearRegression()
model.fit(x_train_2, y_train_2)

# make predictions and find the rmse
preds_2 = model.predict(x_valid_2)
rms = np.sqrt(np.mean(np.power((np.array(y_valid_2) - np.array(preds_2)), 2)))
print(rms)

# scaling data
x_train_scaled = scaler.fit_transform(x_train_2)
x_train_3 = pd.DataFrame(x_train_scaled)
y_train_3 = y_train_2.copy()
x_valid_scaled = scaler.fit_transform(x_valid_2)
x_valid_3 = pd.DataFrame(x_valid_scaled)
y_valid_3 = y_valid_2.copy()

valid_2['Predictions'] = 0
valid_2['Predictions'] = preds_2
valid_2['Year'] = pd.to_datetime(valid_2.Year, format='%Y')

fig2 = px.line(valid_2, x='Year', y=['Close', 'Predictions'], title='Bitcoin Prediction Price Change')
#fig2.show()

# 3rd Model
# using gridsearch to find the best parameter
params = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9]}
knn = neighbors.KNeighborsRegressor()
model = GridSearchCV(knn, params, cv=5)

# fit the model and make predictions
model.fit(x_train_3, y_train_3)
preds_3 = model.predict(x_valid_3)

# rmse
rms = np.sqrt(np.mean(np.power((np.array(y_valid_3) - np.array(preds_3)), 2)))
print(rms)

x_valid_3['Predictions'] = 0
x_valid_3['Predictions'] = preds_3

fig3 = px.line(x_valid_3, y='Predictions', title='Bitcoin Prediction Price Change')
fig3.show()

# 4th Model
#creating train and test sets

# Create a new dataframe with only the 'Close column

#df2 = pd.read_csv('Input/BTC-USD.csv')
data2 = df.filter(['Close']).copy()
# Convert the dataframe to a numpy array
dataset_4 = data2.values
print(data2)
# Get the number of rows to train the model on
training_data_len = int(np.ceil( len(dataset_4) * .95 ))

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset_4)

train_data = scaled_data[0:int(training_data_len), :]
# Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(360, len(train_data)):
    x_train.append(train_data[i - 360:i, 0])
    y_train.append(train_data[i, 0])
    if i <= 361:
        print(x_train)
        print(y_train)
        print()

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
                        batch_size=5,
                        callbacks=callbacks,
                        validation_data=(x_train, y_train)
                        )

# Create the testing data set
# Create a new array containing scaled values from index 1543 to 2002
test_data = scaled_data[training_data_len - 360:, :]
# Create the data sets x_test and y_test
x_test = []
y_test = dataset_4[training_data_len:, :]
for i in range(360, len(test_data)):
    x_test.append(test_data[i - 360:i, 0])

# Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
print(rmse)

print(mean_absolute_error(y_test, predictions))

train4 = data2[:training_data_len]
valid4 = data2[training_data_len:].copy()
valid4['Predictions'] = predictions

fig4 = px.line(valid4, y=['Close','Predictions'], title='Bitcoin Prediction Price Change')
fig4.show()