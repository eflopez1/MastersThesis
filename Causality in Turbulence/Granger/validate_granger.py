# Import modules
from datetime import datetime
from shutil import copyfile
import numpy as np
import os

from Data_Generators.functions import baccala5_linear, baccala5_nonlinear
from Master_Plotter.PlottingTools import *
from GrangerCausality import GrangerCausality

# Create a save folder
now = str(datetime.now())
now = now.replace(":","")
now = now[:-7]
saveLoc = f"Verification {now}/"
os.makedirs(saveLoc, exist_ok=True)

# Save this file and the data generation file for logging
import Data_Generators.functions as functions
import GrangerCausality
copyfile(functions.__file__,saveLoc + 'functions.txt')
copyfile(GrangerCausality.__file__,saveLoc+'GrangerCausality.txt')
copyfile(__file__,saveLoc+'Verification.txt')

# Create and save the data to be checked
linear_data = baccala5_linear(10_000)
nonlinear_data = baccala5_nonlinear(10_000)
np.save(saveLoc+'LinearData.npy',linear_data)
np.save(saveLoc+'NonlinearData.npy',nonlinear_data)

# Calculating and plotting GC Matrices
gc = GrangerCausality(verbose=True)
gc_matrix_linear = gc.GC_Matrix(linear_data, lag=1, conditional=True)
plot_results(gc_matrix_linear,"Linear Model",saveLoc, normalize=True)
gc_matrix_nonlinear = gc.GC_Matrix(nonlinear_data, lag=1, conditional=True)
plot_results(gc_matrix_nonlinear,"Nonlinear Model",saveLoc, normalize=True)
