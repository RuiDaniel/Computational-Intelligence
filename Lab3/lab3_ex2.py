# Use the file DCOILBRENTEUv2.csv as your data file. Read the file and plot it using
# pandas and matplotlib. Make sure the program can open the file correctly and plot it
# correctly.
# Plot a histogram where in the x axis you plot the variation from the previous day (Close
# from the previous day to Close from the present day).
# Plot another histogram with the variation in the day (the difference from High to Low in
# each day).
# Now use your creativity to plot some interesting statistic for this data file.

import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('DCOILBRENTEUv2.csv')

t1 = np.arange(0, df.shape[0], 1)
plt.plot(t1, df['DCOILBRENTEU'])
plt.ylabel('DCOILBRENTEU')
plt.grid()
plt.show()

df['diff'] = df['DCOILBRENTEU']

prev = -1

for index, row in df.iterrows():
    if prev == -1:
        df.at[index,'diff'] = 0
    else:
        df.at[index,'diff'] = row['DCOILBRENTEU'] - df.iloc[prev]['DCOILBRENTEU']
    prev += 1
    
t1 = np.arange(0, df.shape[0], 1)
plt.hist(df['diff'])
plt.ylabel('diff')
plt.grid()
plt.show()

