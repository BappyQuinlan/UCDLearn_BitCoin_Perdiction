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
from sklearn.linear_model import LinearRegression

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

# In the next step, we will create predictions for the validation set and check the RMSE using the actual values.
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
fig1.show()

new_df_2 = new_df.copy()
add_datepart(new_df_2, 'Date')
new_df_2.drop('Elapsed', axis=1, inplace=True)  # elapsed will be the time stamp

print(new_df_2.head())

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

valid_2['Predictions'] = 0
valid_2['Predictions'] = preds_2
valid_2['Year'] = pd.to_datetime(valid_2.Year, format='%Y')

fig2 = px.line(valid_2, x='Year', y=['Close', 'Predictions'], title='Bitcoin Prediction Price Change')
fig2.show()

