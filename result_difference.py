# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 15:18:31 2025

@author: umroot
"""

#####plot data
# import pandas as pd

# df_etth1 = pd.read_csv('Ettm2.csv')

# # print(df_etth1.info())
# # print(df_etth1.isna().sum())

# df_etth1['date'] = pd.to_datetime(df_etth1['date'])

# import matplotlib.pyplot as plt

# cols = ['HUFL',	'HULL',	'MUFL',	'MULL',	'LUFL',	'LULL',	'OT']

# plt.figure(figsize=(15, 10))

# for i, col in enumerate(cols):
#     plt.subplot(len(cols), 1, i+1)
#     plt.plot(df_etth1['date'], df_etth1[col], label=col,color='orange')
#     plt.title(f'{col}')
#     plt.xlabel('Date')
#     plt.ylabel('Load')
#     #plt.legend()
#     plt.tight_layout()

# plt.show()



##### acf plot
# import pandas as pd
# import matplotlib.pyplot as plt
# from statsmodels.graphics.tsaplots import plot_acf

# # Load and prepare data
# df_etth1 = pd.read_csv('Ettm2.csv')
# df_etth1['date'] = pd.to_datetime(df_etth1['date'])

# cols = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']

# plt.figure(figsize=(15, 20))

# for i, col in enumerate(cols):
#     ax = plt.subplot(len(cols), 1, i + 1)
    
#     # Plot ACF and capture the returned Line2D objects
#     plot_acf(df_etth1[col].dropna(), ax=ax, lags=50, title=f'ACF of {col}')

#     # Change color of bars (ACF values) to orange
#     # Typically, first two lines are the confidence intervals
#     for line in ax.lines[2:]:
#         line.set_color('orange')

#     # Change color of markers (dots) to orange
#     for line in ax.lines[2:]:
#         line.set_markerfacecolor('orange')
#         line.set_markeredgecolor('orange')

# plt.tight_layout()
# plt.show()


###power spectra
# import pandas as pd
# import matplotlib.pyplot as plt

# # Load datasets
# df_etth1 = pd.read_csv('ETTm1.csv')
# df_etth2 = pd.read_csv('ETTm2.csv')

# # Convert 'date' column to datetime
# df_etth1['date'] = pd.to_datetime(df_etth1['date'])
# df_etth2['date'] = pd.to_datetime(df_etth2['date'])

# # Shared columns to compare
# cols = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']

# # Plot settings
# plt.figure(figsize=(15, 25))

# for i, col in enumerate(cols):
#     ax = plt.subplot(len(cols), 1, i + 1)

#     # Plot PSD for ETTm1
#     plt.psd(df_etth1[col].dropna(), NFFT=256, Fs=1, color='blue', label='ETTm1')

#     # Plot PSD for ETTm2
#     plt.psd(df_etth2[col].dropna(), NFFT=256, Fs=1, color='orange', label='ETTm2')

#     plt.title(f'Power Spectrum of {col} (ETTm1 vs ETTm2)')
#     plt.xlabel('Frequency')
#     plt.ylabel('Power/Frequency (dB/Hz)')
#     plt.legend()

# plt.tight_layout()
# plt.show()

#####variance and snr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load datasets
df_etth1 = pd.read_csv('ETTm1.csv')
df_etth2 = pd.read_csv('ETTm2.csv')
#seasonality=24*4 #daily
#seasonality=24*4*7 #weekly
seasonality=24*4*365 #yearly
# Convert date to datetime
df_etth1['date'] = pd.to_datetime(df_etth1['date'])
df_etth2['date'] = pd.to_datetime(df_etth2['date'])

# Columns to analyze
cols = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']

# Dictionaries to store results
var1, var2 = {}, {}
snr1, snr2 = {}, {}

# Simple moving average smoothing for noise estimation
def moving_average(x, window=seasonality):
    return x.rolling(window=window, min_periods=1, center=True).mean()

# Compute variance and SNR for each variable
for col in cols:
    x1 = df_etth1[col].dropna()
    x2 = df_etth2[col].dropna()

    # Variance
    var1[col] = np.var(x1)
    var2[col] = np.var(x2)

    # Smoothed signal (moving average)
    smooth_x1 = moving_average(x1)
    smooth_x2 = moving_average(x2)

    # Noise estimation (original - smooth)
    noise1 = x1 - smooth_x1
    noise2 = x2 - smooth_x2

    # SNR estimate
    snr1[col] = np.var(x1) / np.var(noise1)
    snr2[col] = np.var(x2) / np.var(noise2)

# Plotting Variance Comparison
plt.figure(figsize=(12, 6))
x = np.arange(len(cols))
width = 0.35 #width of bar

# plt.bar(x - width/2, [var1[c] for c in cols], width, label='ETTm1', color='blue')
# plt.bar(x + width/2, [var2[c] for c in cols], width, label='ETTm2', color='orange')
# plt.xticks(x, cols)
# plt.ylabel('Variance')
# plt.title('Variance Comparison Between ETTm1 and ETTm2')
# plt.legend()
# plt.tight_layout()
# plt.show()

# Plotting SNR Comparison
plt.figure(figsize=(12, 6))
plt.bar(x - width/2, [snr1[c] for c in cols], width, label='ETTm1', color='blue')
plt.bar(x + width/2, [snr2[c] for c in cols], width, label='ETTm2', color='orange')
plt.xticks(x, cols)
plt.ylabel('SNR (approximate)')
#plt.title('SNR Comparison Between ETTm1 and ETTm2')
plt.legend()
plt.tight_layout()
plt.show()
