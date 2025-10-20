#Kalman_Filter & Pairs_Trading

from pykalman import KalmanFilter
import numpy as np
import pandas as pd
from numpy import poly1d
from datetime import datetime

import matplotlib.pyplot as plt

plt.style.use('seaborn-darkgrid')
plt.rcParams['figure.figsize'] = (10,7)

plt.show()

# Define path where data file is saved in your system
#path = '../data/'
data = pd.read_csv('./trial_data.csv', index_col ='Date')
data['ratio'] = data['AAPL']/ data['FB']
stock_1 = data['AAPL']
stock_2 = data['FB']

# Calculate the hedge ratio for pairs trading
ratio =stock_1/stock_2
data.head()

kf = KalmanFilter(transition_matrices = [1], observation_matrices = [1],
              initial_state_mean = 0,
              initial_state_covariance = 1,
              observation_covariance=1,
              transition_covariance=.0001)

mean, cov = kf.filter(ratio.values)
mean, std = mean.squeeze(), np.std(cov.squeeze())

plt.figure(figsize=(15,7))
plt.plot(ratio.values - mean, 'm', lw=1)
plt.plot(np.sqrt(cov.squeeze()), 'y', lw=1)
plt.plot(-np.sqrt(cov.squeeze()), 'c', lw=1)
plt.title('Kalman filter estimate')
plt.legend(['Error: real_value - mean', 'std', '-std'])
plt.xlabel('Day')
plt.ylabel('Value')

data['mean'] = mean.squeeze()
data['cov'] = cov.squeeze()
data['std'] = np.sqrt(data['cov'])
data = data.dropna()

data['ma'] = data['ratio'].rolling(5).mean()
data['z_score'] = (data['ma'] - data['mean']) / data['std']

# Initialise positions as zero
data['position_1'] = np.nan
data['position_2'] = np.nan

# Generate buy, sell and square off signals as: z<-1 buy, z>1 sell and -1<z<1 liquidate the position
for i in range(data.shape[0]):
    if data['z_score'].iloc[i] < -1:
        data['position_1'].iloc[i] = 1
        data['position_2'].iloc[i] = -round(data['ratio'].iloc[i], 0)
    if data['z_score'].iloc[i] > 1:
        data['position_1'].iloc[i] = -1
        data['position_2'].iloc[i] = round(data['ratio'].iloc[i], 0)
    if (abs(data['z_score'].iloc[i]) < 1) & (abs(data['z_score'].iloc[i]) > 0):
        data['position_1'].iloc[i] = 0
        data['position_2'].iloc[i] = 0

# Calculate returns
data['returns'] = ((data['AAPL'] - data['AAPL'].shift(1)) / data['AAPL'].shift(1)) * data['position_1'].shift(1) + (
            (data['FB'] - data['FB'].shift(1)) / data['FB'].shift(1)) * data['position_2'].shift(1)
data['returns'].sum()
plt.show()

