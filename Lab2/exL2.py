# -*- coding: utf-8 -*-
"""
Created on  September 27 2019

@author: Rui Neves

Lab 2 Computacional Inteligence

The file DCOILBRENTEUv2.csv shows the evolution of the oil price through time. 
"""

import numpy as np
import pandas as pd

df = pd.read_csv('DCOILBRENTEUv2.csv')

# a) Normalize (min-max) and standardize (Z-score) the price and visualize it.
# Min-Max normalization: newx = x-min/(max-min)
# z-score standardization z = (x-µ)/σ=(x-mean)/(standard deviation)


df['minmax'] = (df['DCOILBRENTEU'] - df['DCOILBRENTEU'].min()) / (df['DCOILBRENTEU'].max() - df['DCOILBRENTEU'].min())

df['zscore'] = (df['DCOILBRENTEU'] - df['DCOILBRENTEU'].mean()) / df['DCOILBRENTEU'].std()

print(df)

# b) Smooth the normalized price by computing a moving average with a 50-day period. Visualize the result.

sma = df['minmax'].rolling(50).mean()

import matplotlib.pyplot as plt

t1 = np.arange(0, 1273, 1)
plt.plot(t1, sma, 'r', t1, df['minmax'])

plt.legend(['moving average', 'minmax'])
plt.grid()
plt.show()


