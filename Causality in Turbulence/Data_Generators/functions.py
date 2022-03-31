"""
This script contains functions which produce data for testing transfer entropy on
"""

# %%
# Test run of importations

import numpy as np
import os
import matplotlib.pyplot as plt
from numpy import savetxt
import cmath
import warnings

from matplotlib import rc

plt.rcParams['font.family'] = "serif"
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

from pandas import read_csv
from pandas.plotting import lag_plot
import pandas as pd
from pandas import DataFrame
from pandas import concat
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from math import sqrt

# Import Statsmodels
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def adfuller_test(series, signif=0.05, name='', verbose=False):
    """Perform ADFuller to test for Stationarity of given series and print report"""
    r = adfuller(series, autolag='AIC')
    if verbose:
      output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
      p_value = output['pvalue'] 
      def adjust(val, length= 6): return str(val).ljust(length)

      # Print Summary
      print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
      print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
      print(f' Significance Level    = {signif}')
      print(f' Test Statistic        = {output["test_statistic"]}')
      print(f' No. Lags Chosen       = {output["n_lags"]}')

      for key,val in r[4].items():
          print(f' Critical value {adjust(key)} = {round(val, 3)}')

      if p_value <= signif:
          print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
          print(f" => Series is Stationary.")
          return True
      else:
          print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
          print(f" => Series is Non-Stationary.")
          return False 

    else:
      p_value = round(r[1], 4)
      if p_value <=signif:
        return True
      else:
        return False

def baccala5_linear(N, scale=1.0):
    """
    Inputs:
        N (int): Integer describing the data length 
            requested for each variable.
        scale (float): Standard deviation for noise

    Returns:
        Data (pandas.DataFrame): Dataframe
            with all data
    """
    from numpy.random import normal
    from math import sqrt
    x1 = np.zeros(N+1)
    x2 = np.zeros(N+1)
    x3 = np.zeros(N+1)
    x4 = np.zeros(N+1)
    x5 = np.zeros(N+1)

    for i in range(1,N+1):
        x1[i] = 0.95*sqrt(2) * x1[i-1] - 0.9025*x1[i-1] + normal(0,scale,1)
        x2[i] = 0.5*x1[i-1] + normal(0,scale,1)
        x3[i] = -0.4*x1[i-1] + normal(0,scale,1)
        x4[i] = -0.5*x1[i-1] + 0.25*sqrt(2)*x4[i-1] + 0.25*sqrt(2)*x5[i-1] + normal(0,scale,1)
        x5[i] = -0.25*sqrt(2)*x4[i-1] + 0.25*sqrt(2)*x5[i-1] + normal(0,scale,1)
    
    x1 = x1[1:]
    x2 = x2[1:]
    x3 = x3[1:]
    x4 = x4[1:]
    x5 = x5[1:]
    data = dict(
        x1 = x1,
        x2 = x2,
        x3 = x3,
        x4 = x4,
        x5 = x5
    )
    data = pd.DataFrame(data)
    return data

def baccala5_nonlinear(N, scale=1.0):
    from numpy.random import normal
    from math import sqrt
    x1 = np.zeros(N+1)
    x2 = np.zeros(N+1)
    x3 = np.zeros(N+1)
    x4 = np.zeros(N+1)
    x5 = np.zeros(N+1)

    for i in range(1,N+1):
        x1[i] = 0.95*sqrt(2) * x1[i-1] - 0.9025*x1[i-1] + normal(0,scale,1)
        x2[i] = 0.5*x1[i-1]**2 + normal(0,scale,1)
        x3[i] = -0.4*x1[i-1] + normal(0,scale,1)
        x4[i] = -0.5*x1[i-1]**2 + 0.25*sqrt(2)*x4[i-1] + 0.25*sqrt(2)*x5[i-1] + normal(0,scale,1)
        x5[i] = -0.25*sqrt(2)*x4[i-1] + 0.25*sqrt(2)*x5[i-1] + normal(0,scale,1)
    
    x1 = x1[1:]
    x2 = x2[1:]
    x3 = x3[1:]
    x4 = x4[1:]
    x5 = x5[1:]
    data = dict(
        x1 = x1,
        x2 = x2,
        x3 = x3,
        x4 = x4,
        x5 = x5
    )
    data = pd.DataFrame(data)
    return data

def baccala5(N, scale=1.0):
    """
    Used in the paper Granger Causality in the Frequency Domain
    https://www.scielo.br/j/rbef/a/m4LwwHLvk7YwPNMhngNJQwp/?lang=en

    Originally comes from Bacccala and Sameshima 2001

    It is similar to the above set, but has varying time delays
    """
    from numpy.random import normal
    from math import sqrt
    x1 = np.zeros(N+3)
    x2 = np.zeros(N+3)
    x3 = np.zeros(N+3)
    x4 = np.zeros(N+3)
    x5 = np.zeros(N+3)

    for i in range(3,N+3):
        x1[i] = 0.95*sqrt(2) * x1[i-1] - 0.9025*x1[i-2] + normal(0,scale,1)
        x2[i] = 0.5*x1[i-2] + normal(0,scale,1)
        x3[i] = -0.4*x1[i-3] + normal(0,scale,1)
        x4[i] = -0.5*x1[i-2] + 0.25*sqrt(2)*x4[i-1] + 0.25*sqrt(2)*x5[i-1] + normal(0,scale,1)
        x5[i] = -0.25*sqrt(2)*x4[i-1] + 0.25*sqrt(2)*x5[i-1] + normal(0,scale,1)
        
    # Removing the first 2 values, as they are 0s
    x1 = x1[3:]
    x2 = x2[3:]
    x3 = x3[3:]
    x4 = x4[3:]
    x5 = x5[3:]

    data = dict(
        x1=x1,
        x2=x2,
        x3=x3,
        x4=x4,
        x5=x5
    )
    data = pd.DataFrame(data)
    return data

def chen11_linear(N, c):
    """
    Equation 11 from Chen 2004
    https://www.sciencedirect.com/science/article/pii/S0375960104002403
    
    The coupling between variables is linear.

    Inputs:
        N (int): Length of data
        c (float): Coupling parameter between x and y.
            0 means no coupling
            1 means total coupling
    """
    from numpy import exp
    from numpy.random import normal
    x = np.zeros(N+2)
    y = np.zeros(N+2)

    # Set initial values
    x[:2] = normal(size=2)
    y[:2] = normal(size=2)

    for i in range(2, N+2):
        x[i] = 3.4*x[i-1] * (1-x[i-1]**2) * exp(-x[i-1]**2) + 0.8*x[i-2]
        y[i] = 3.4*y[i-1] * (1-y[i-1]**2) * exp(-y[i-1]**2) + 0.5*y[i-2] + c*x[i-2]

    # Removing the first two values
    x, y = x[2:], y[2:]

    data = dict(
        x = x,
        y = y
    )
    data = pd.DataFrame(data)
    return data


def chen12_nonlinear(N, c):
    """
    Equation 12 from Chen 2004
    https://www.sciencedirect.com/science/article/pii/S0375960104002403

    The coupling between variables is nonlinear

    Inputs:
        N (int): Length of data
        c (float): Coupling parameter between x and y.
            0 means no coupling
            1 means total coupling
    """
    from numpy import exp
    from numpy.random import normal
    x = np.zeros(N+2)
    y = np.zeros(N+2)

    # Set initial values
    x[:2] = normal(size=2)
    y[:2] = normal(size=2)

    for i in range(2, N+2):
        x[i] = 3.4*x[i-1] * (1-x[i-1]**2) * exp(-x[i-1]**2) + 0.8*x[i-2]
        y[i] = 3.4*y[i-1] * (1-y[i-1]**2) * exp(-y[i-1]**2) + 0.5*y[i-2] + c*x[i-2]**2

    # Removing the first two values 
    x, y = x[2:], y[2:]

    data = dict(
        x = x,
        y = y
    )
    data = pd.DataFrame(data)
    return data

def pd_to_txt(data, title, save_loc):
    """
    Given a pandas.DataFrame, file title, and save location,
    will save the pandas dataframe in the file location.
    """
    np.savetxt(save_loc + title +'.txt', data.values, fmt='%d')


def heatmap2d(arr: np.ndarray):
    plt.imshow(arr, cmap='Reds')
    plt.colorbar()