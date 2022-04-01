# Import modules
from datetime import datetime
from shutil import copyfile
import numpy as np
import os

from Data_Generators.functions import baccala5_linear, baccala5_nonlinear
from Master_Plotter.PlottingTools import *
from DarbellayAdaptive import TE_Matrix as DVTE

# Create a save folder
now = str(datetime.now())
now = now.replace(":","")
now = now[:-7]
saveLoc = f"Verification {now}/"
os.makedirs(saveLoc, exist_ok=True)

# Save this file and the data generation file for logging
import Data_Generators.functions as functions
import DarbellayAdaptive
copyfile(functions.__file__,saveLoc + 'functions.txt')
copyfile(DarbellayAdaptive.__file__,saveLoc+'DarbellayAdaptive.txt')
copyfile(__file__,saveLoc+'Verification.txt')

# Create and save the data to be checked
linear_data = baccala5_linear(10_000)
nonlinear_data = baccala5_nonlinear(10_000)
np.save(saveLoc+'LinearData.npy',linear_data)
np.save(saveLoc+'NonlinearData.npy',nonlinear_data)

# Calculating and plotting GC Matrices
te_matrix_linear = DVTE(linear_data, lag=1, conditional=True, maxNumPartitions=np.inf)
plot_results(te_matrix_linear,"Linear Model", saveLoc, normalize=True)
te_matrix_nonlinear = DVTE(nonlinear_data, lag=1, conditional=True, maxNumPartitions=np.inf)
plot_results(te_matrix_nonlinear,"Nonlinear Model", saveLoc, normalize=True)
