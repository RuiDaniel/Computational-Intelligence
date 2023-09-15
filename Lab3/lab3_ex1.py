# Use the file EURUSD_Daily_Ask_2018.12.31_2019.10.05v2.csv as your data file.
# Read the file and plot it using pandas and matplotlib. Make sure the program is able to
# open the file correctly and plot it.

import pandas as pd

df = pd.read_csv('EURUSD_Daily_Ask_2018.12.31_2019.10.05v2.csv', decimal=',', sep=';')

# print(df)

# Use the function “to_datetime” from pandas to create a new column with the
# information about the time in the file that is read as a string to the internal format
# datetime. Also import the library “datetime” and use it to create a new element in the
# format datetime. You can use the following function to create a new datetime where you
# explicitly give the values startTime = datetime(year, month, day, 0, 0) or use the
# function now() to get the current time.

from datetime import datetime

df['Time'] = pd.to_datetime(df['Time (UTC)'])

# print(df)

startTime = datetime.now()

# print(startTime)

# Now make a function to detect and remove the outlier. Plot the data. Create two 
# different functions to detect the outlier:
# 1) Find the outlier value (just by looking at the figure).

import matplotlib.pyplot as plt
import numpy as np

def detect_outliers_plot(df):
    t1 = np.arange(0, 199, 1)
    plt.plot(t1, df['Volume '])

    #plt.legend(['volume', 'close'])
    #plt.xlabel('High - Low')
    #plt.ylabel('Volume')
    plt.grid()
    plt.show()
    
# detect_outliers_plot(df)

# 2) Detect the samples that are k*σ far from the average.

def detect_outliers(df, column, k):
    dp = df[column].std()
    mean = df[column].mean()
    outliers = df[abs(df[column] - mean) > k * dp]
    return outliers

outliers = detect_outliers(df, 'Volume ', 3.1)
print(outliers)

# Now make three different functions to remove the outlier.
# 3) Remove the line in the pandas dataset.

def remove_outlier_1(df, outliers):
    return pd.concat([df, outliers]).drop_duplicates(keep=False)

# print(df)
df_without_outliers = remove_outlier_1(df, outliers)
# print(df_without_outliers)

# 4) Change the value with the previous one.

def remove_outlier_2(df, outliers):
    indexes = []
    for i in range(outliers.shape[0]):
        indexes.append(outliers.loc[df['Volume '] > -1].index[i])

    row_old_value = df.iloc[-1]['Volume ']
    for index, row in df.iterrows():
        if index in indexes:
            df.at[index,'Volume '] = row_old_value
        row_old_value = row['Volume ']
    
    return df

# print(df)
# df_without_outliers = remove_outlier_2(df, outliers)
# print(df_without_outliers)

# 5) Change the value with the interpolation of the previous and the next one.

def remove_outlier_3(df, outliers):
    indexes = []
    for i in range(outliers.shape[0]):
        indexes.append(outliers.loc[df['Volume '] > -1].index[i])

    prev = -1
    next = 1
    
    for index, _ in df.iterrows():
        row_old_value = df.iloc[prev]['Volume ']
        row_next_value = df.iloc[next]['Volume ']
        if index in indexes:
            print(index, row_old_value, row_next_value)
            df.at[index,'Volume '] = (row_old_value + row_next_value) / 2
        prev += 1
        next += 1
        
        if next >= df.shape[0]:
            next = 0
    
    return df

# print(df)
# df_without_outliers = remove_outlier_3(df, outliers)
# print(df_without_outliers)

# Finally plot the results without the outliers (and save them).

# t1 = np.arange(0, 196, 1)
# plt.plot(t1, df_without_outliers['Volume '])
# plt.grid()
# plt.show()