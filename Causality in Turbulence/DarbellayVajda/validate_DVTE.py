"""
This script will create data and validate the Granger Causality metric
"""
# %%
# Execution

# Import modules
from datetime import datetime
from shutil import copyfile
import numpy as np
import os
try:
    dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir)
except:
    print("Unable to set directory to this file's location")
import sys
sys.path.insert(1, '../Data_Generators')
from functions import tds_linear, tds_nonlinear, plot_results
from DarbellayAdaptive import TE_Matrix as DVTE

# Create a save folder
now = str(datetime.now())
now = now.replace(":","")
now = now[:-7]
saveLoc = f"Verification {now}/"
os.makedirs(saveLoc, exist_ok=True)

# Save this file and the data generation file for logging
import functions, DarbellayAdaptive
copyfile(functions.__file__,saveLoc + 'functions.txt')
copyfile(DarbellayAdaptive.__file__,saveLoc+'DarbellayAdaptive.txt')
copyfile(__file__,saveLoc+'Verification.txt')

# Create and save the data to be checked
linear_data = tds_linear(10_000)
nonlinear_data = tds_nonlinear(10_000)
np.save(saveLoc+'LinearData.npy',linear_data)
np.save(saveLoc+'NonlinearData.npy',nonlinear_data)

# Calculating and plotting GC Matrices
te_matrix_linear = DVTE(linear_data, lag=1, conditional=True, maxNumPartitions=np.inf)
plot_results(te_matrix_linear,"Linear Model", saveLoc, normalize=True)
te_matrix_nonlinear = DVTE(nonlinear_data, lag=1, conditional=True, maxNumPartitions=np.inf)
plot_results(te_matrix_nonlinear,"Nonlinear Model", saveLoc, normalize=True)
