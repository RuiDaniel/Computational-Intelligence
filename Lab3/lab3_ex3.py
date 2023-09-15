import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

brent = pd.read_csv('DCOILBRENTEUv2.csv')
wti = pd.read_csv('DCOILWTICOv2.csv')

# t1 = np.arange(0, len(brent), 1)
# t2 = np.arange(0, len(wti), 1)
# plt.plot(t1, brent['DCOILBRENTEU'], 'r', t2, wti['DCOILWTICO'])

# plt.legend(['brent', 'wti'])
# plt.grid()
# plt.show()

# Now make a Scatter plot with the value oil of the different regions and see how they behave

t1 = np.arange(0, len(brent), 1)
t2 = np.arange(0, len(wti), 1)
plt.scatter(t1, brent['DCOILBRENTEU'], c='r', label='brent')
plt.scatter(t2, wti['DCOILWTICO'], c='b', label='wti')

plt.legend()
plt.grid()
plt.show()