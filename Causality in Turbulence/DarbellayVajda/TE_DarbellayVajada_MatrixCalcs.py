# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 20:49:33 2021

@author: elope

It has been observed that on the workstation in the lab, calculating the transfer entropy on 8 parallel processes 
approximately doubles the computation time for a single matrix. Thus, while it may take longer for an individual matrix
to be made, the number of matrices made concurrently outweighs this computational cost
"""

import timeit
import os
import pandas as pd
from DarbellayAdaptive import TE_Matrix, plot_results
from scipy.io import loadmat
import numpy as np
from multiprocessing import Pool, freeze_support
from itertools import product
from concurrent.futures.thread import ThreadPoolExecutor
from numpy import savetxt
import pdb

# Cluster
# data_folder="/home/elopez8/Turbulent_Causality/Ricardo's Data/"
# results_folder='/home/elopez8/Turbulent_Causality/CorrectedConditionalCausality/'

# Desktop
# data_folder="C:/Users/elope/Documents/Turbulent Causality/Ricardo's Data/"
# results_folder = "C:/Users/elope/Documents/Turbulent Causality/DVTE/"

# Laptop
# data_folder='C:/Users/17088/Documents/Soft Robotics Research/Turbulent Causality/Data/'
# results_folder='C:/Users/17088/Documents/Soft Robotics Research/Turbulent Causality/Conditional Causality Results/'

# Big Gucci
data_folder = 'C:/Causality Results/Data/'
results_folder = 'C:/Causality Results/DarbellayVajadaTE_Stacked (16Mar2021)/'
os.makedirs(results_folder,exist_ok=True)

sequences=list(range(0,600,50))
lags=[1,6]
maxNumPartition = 15
numProc = 8
conditional=True
stacking=True
numStacks = 5
numCombos=5
effec = False
effecOnly = False

if stacking:
    #For random stacking
    combos = []
    comboDict = {}
    for i in range(numCombos):
        combo = np.random.randint(0,600,numStacks)
        combos.append(combo)
        comboDict[i+1] = combo
        
    txtFile2 = results_folder+'ComboDict.txt'
    combosSaved = []
    for key in comboDict:
       combosSaved.append([str(key)+':',comboDict[key]])
    with open(txtFile2,'w') as file:
        for line in combosSaved:
            file.write("%s\n" % line)

def worker(lag,sequence):
    global results_folder
    global data_folder
    global maxNumPartition
    maxPartition = maxNumPartition
    global conditional
    global stacking
    global numStacks
    global effec
    global effecOnly
    global comoDict
    
    # Importing data
    if stacking:
        print('Stacking Data')
        combo = comboDict[sequence]
        dataFrames = []
        for seq in combo:
            # Importing data
            file='seq_{}.mat'.format(str(seq))
            loc=data_folder+file
            data=loadmat(loc)
            data=data['testSeq']
            
            data_frame=pd.DataFrame(data=data)
            data_frame=data_frame.T
            dataFrames.append(data_frame)
            
        data = pd.concat(dataFrames)
        data = data.to_numpy()
        
    else:
        file='seq_{}.mat'.format(str(sequence))
        loc=data_folder+file
        data=loadmat(loc)
        data=data['testSeq'].T
    
    # For labels on plot
    columns_int = np.arange(1,data.shape[1]+1)
    columns = []
    for c in columns_int:
        columns.append(str(c))
        
    if conditional: title = 'Conditional Darbellay TE Seq {}, Lag {}, MaxPart {}'.format(sequence, lag, maxPartition)
    else: title = 'Darbellay TE Seq {}, Lag {}, MaxPart {}'.format(sequence, lag,maxPartition)
    
    if stacking: title = 'Conditional Darbellay TE Combo {}, Lag {}, MaxPart {}, {} Stacks'.format(sequence, lag, maxPartition, numStacks)
    if effec: title = 'Effective '+title
    
    TEresults = TE_Matrix(data,lag,maxNumPartitions=maxPartition,conditional=conditional,effec=effec,effecOnly=effecOnly)
    plot_results(TEresults,title,columns,results_folder)
    
    np.nan_to_num(TEresults, copy=False)
    
    normalized=[] #Saving the normalized TE for plotting
    for row in TEresults:
        row_tot=sum(row)
        normalized.append(row/row_tot)
                
    normalized=np.asarray(normalized)
    
    for i in range(normalized.shape[0]):
        for j in range(normalized.shape[1]):
            if i==j:
                normalized[i][j]=np.nan
                
    if conditional: title = 'Normalized Conditional DV-TE on Seq {}, Lag {}, MaxPart {}'.format(sequence,lag,maxPartition)
    else: title = 'Normalized DV-TE on Seq {}, Lag {}, MaxPart {}'.format(sequence, lag, maxPartition)
        
    if stacking: title = 'Normalized Conditional DV-TE Combo {}, Lag {}, MaxPart {}, {} Stacks'.format(sequence, lag, maxPartition, numStacks)
    if effec: title = 'Effective '+title
    plot_results(normalized,title,columns,results_folder)
    
    

if __name__=='__main__':
    # freeze_support()
    # allCombo = list(product(lags,sequences))
    allCombo = list(product(lags,list(range(1,numCombos+1))))
    p= Pool(numProc)
    p.starmap(worker,allCombo)
    
    # No multiprocessing
    # for combo in allCombo:
    #     worker(*combo)
    # for lag in lags:
    #     for seq in sequences:
            # worker(seq,lag)
