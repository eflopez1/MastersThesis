"""
This script contains plotting tools to plot causal heatmaps
"""

import numpy as np
from numpy import savetxt
import matplotlib.pyplot as plt
import os

plt.rcParams['font.family'] = "serif"
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

def heatmap2d(arr: np.ndarray):
    plt.imshow(arr, cmap='Reds')
    plt.colorbar()
    
def plot_results(Matrix, title='', save_loc=None, fig_kwargs={}, normalize=False):

    columns_int = np.arange(1,max(Matrix.shape)+1)
    labels = []

    for col in columns_int:
        labels.append(str(col))
    
    for i in range(Matrix.shape[0]):
        for j in range(Matrix.shape[1]):
            if i==j:
                Matrix[i][j]=np.nan
    
    if normalize:
        Matrix /= np.nanmax(Matrix)

    fig, ax = plt.subplots(**fig_kwargs)
    ax.set_xticks(np.arange(Matrix.shape[1]), minor=False)
    ax.set_xticklabels(labels, minor=False)
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(Matrix.shape[1]), minor=False)
    ax.set_yticklabels(labels, minor=False)
    
    # Trying new labels
    ax.set_xlabel('Effect', fontsize='xx-large')
    ax.set_ylabel('Cause', fontsize='xx-large')
    
    plt.title(title, fontsize='xx-large')
    heatmap2d(Matrix)
    
    if save_loc != None:
        os.makedirs(save_loc, exist_ok=True)
        
        plt.savefig(save_loc + title +  '.png', bbox_inches='tight')
        savetxt(save_loc + title +  '.csv', Matrix, delimiter=',')
        plt.close('all')