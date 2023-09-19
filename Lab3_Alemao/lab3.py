import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

file1 = pd.read_csv('EURUSD_Daily_Ask_2018.12.31_2019.10.05v2.csv', decimal=',', sep=';')


file1.rename(columns={"Volume  ": "Volume"}, inplace = True)
# print(file1.head())
# print(file1.columns)


file1['time1'] = pd.to_datetime(file1['Time (UTC)'], format='%Y.%m.%d %H:%M:%S')
# print(file1['time1'])
# print(file1.columns)

startTime = datetime.now()

plt.figure('Open_outlier')
plt.scatter(range(len(file1['Open'])),file1['Open'])
plt.xlabel('Day')
plt.ylabel('Open')
plt.show()

plt.figure('High_outlier')
plt.scatter(range(len(file1['High'])), file1['High'])
plt.xlabel('Day')
plt.ylabel('High')
plt.show()

plt.figure('low_outlier')
plt.scatter(range(len(file1['Low'])), file1['Low'])
plt.xlabel('Day')
plt.ylabel('Low')
plt.show()


plt.figure('close_outlier')
plt.scatter(range(len(file1['Close'])), file1['Close'])
plt.xlabel('Day')
plt.ylabel('Close')
plt.show()

average_Close = file1['Close'].mean()
average_High = file1['High'].mean()
average_Low = file1['Low'].mean()
average_Open = file1['Open'].mean()

std_Close = file1['Close'].std()
std_High = file1['High'].std()
std_Low = file1['Low'].std()
std_Open = file1['Open'].std()


outlier = file1[(file1['Close'] < (average_Close - 3 * std_Close)) | (file1['Close'] > (average_Close + 3 * std_Close))]
print(outlier)
out_index = outlier.index

no_outlier1 = file1[(file1['Close'] > (average_Close - 3 * std_Close)) & (file1['Close'] < (average_Close + 3 * std_Close))].reset_index()
print(no_outlier1['Close'])

plt.figure('no_outlier1')
plt.scatter(range(len(no_outlier1['Close'])), no_outlier1['Close'])
plt.xlabel('Day')
plt.ylabel('Close_no_outlier_1')
plt.show()

#no_outlier2
file1.iloc[out_index] = file1.iloc[out_index - 1]
plt.figure('no_outlier2')
plt.scatter(range(len(file1['Close'])), file1['Close'])
plt.xlabel('Day')
plt.ylabel('Close_no_outlier_1')
plt.show()

#no_outlier3
file1.loc[out_index, 'Close'] = (file1.loc[out_index - 1, 'Close'] + file1.loc[out_index + 1, 'Close']) / 2
plt.figure('no_outlier3')
plt.scatter(range(len(file1['Close'])), file1['Close'])
plt.xlabel('Day')
plt.ylabel('Close_no_outlier_1')
plt.show()

####################################################################################
###   Ex 2   ###

file2 = pd.read_csv('DCOILBRENTEUv2.csv')

date_2 = file2['DATE']
DCOILBRENTEU = file2['DCOILBRENTEU']

plt.figure(6)
plt.plot(range(len(date_2)), DCOILBRENTEU)
plt.xlabel('day')
plt.ylabel('DCOILBRENTEU')
plt.show()

prev = DCOILBRENTEU[0]

variation = []

for element in DCOILBRENTEU:
    # print(element)
    variation.append(element - prev)
    prev = element
    
# print(variation)
plt.figure(7)
plt.hist(variation, bins=30)
plt.ylabel('Variation')
plt.xlabel('day')
plt.show()

# plt.figure(8)
# file2['Variation'] = file2['DCOILBRENTEU'].diff()

# plt.hist(file2['Variation'].dropna(), bins=30, edgecolor='k', alpha=0.7)
# plt.xlabel('Daily Variation')
# plt.ylabel('Frequency')
# plt.title('Histogram of Daily Variation in DCOILBRENTEU')
# plt.grid(True)
# plt.show()  

file1['variation'] = file1['High'] - file1['Low']

plt.figure('High - Low')
plt.hist(file1['variation'], bins=30)
plt.ylabel('Variation')
plt.xlabel('day')
plt.show()



prev = file1['Close'][0]

variation = []

for element in file1['Close']:
    # print(element)
    variation.append(element - prev)
    prev = element

plt.figure('Close prev')
plt.hist(variation, bins=30)
plt.ylabel('Variation')
plt.xlabel('day')
plt.show()


####################################################################################
###   Ex 3   ###


file3 = pd.read_csv('DCOILWTICOv2.csv')

plt.figure('Oil price')
plt.scatter(range(len(file3['DCOILWTICO'])), file3['DCOILWTICO'])
plt.scatter(range(len(file2['DCOILBRENTEU'])), file2['DCOILBRENTEU'])
plt.legend(['US', 'UK'])
plt.xlabel('Day')
plt.ylabel('Oil price')
plt.show()