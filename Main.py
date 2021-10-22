# Main Execution File
# Author : Barry Quinlan
# Date : 22nd October 2021
# Email : bappyquinlan@gmail.com

# import Required Packages
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.offline as po

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
    new_data = pd.DataFrame(index=range(0, len(df)), columns=[column1, column2])

    for i in range(0, len(data)):
        new_data[column1][i] = data[column1][i]
        new_data[column2][i] = data[column2][i]

    return new_data


new_data = create_copy_dataframe_for_analysis(df, 'Date', 'Close')

print(new_data.head())

# Create Training and Test Data from the new data frame


