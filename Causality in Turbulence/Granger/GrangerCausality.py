"""
@author: elopez8

A class to calculate Granger Causality
"""
from sklearn.linear_model import LinearRegression
import numpy as np
from tqdm import tqdm
import pandas as pd
import timeit

class GrangerCausality:
    def __init__(self, verbose=False):
        """
        If verbose=True, information will be shared while calculations are performed.
        """
        self.verbose=verbose
        
    def linear_GC(self, X, Y, conditions=None, lag=1):
        """
        Calculates Granger Causality from X to Y
        
        Notes: 
            -Vectors X and Y must be of the same length.
            -If conditions are present, they must be passed as a Pandas Dataframe with the same length as X and Y
        """
        if self.verbose: start_GC=timeit.default_timer()
        
        # Step 0) Prepare data by discretizing.
        n = len(X)
        
        # Make vectors of appropriate length and position for the given # of lag.
        X_past=[]
        Y_past=[]
        if conditions is not None: conditions_past=[]
            
        for l in range(lag):
            X_past.append(X[0+l:n-lag+l])
            Y_past.append(Y[0+l:n-lag+l])
            if conditions is not None:
                for ind,con in conditions.iteritems():
                    conditions_past.append(con[0+l:n-lag+l])
        Y_fut = Y[lag:]
        
        # Reset the indecies
        for i in range(lag):
            X_past[i].reset_index(drop=True,inplace=True)
            Y_past[i].reset_index(drop=True,inplace=True)
        if conditions is not None:
            for ind in range(len(conditions_past)):
                conditions_past[ind].reset_index(drop=True,inplace=True)
                
        # Create empty pd.DataFrame()
        multi_Y=pd.DataFrame()
        multi_data=pd.DataFrame()
        for i in range(lag):
            multi_data[i]=X_past[i]
            multi_Y[i]=Y_past[i]
                
        if conditions is not None:
            for i in range(lag,len(conditions_past)+lag):
                multi_Y[i]=conditions_past[i-lag]
        
        for i in range(lag,2*lag):
            multi_data[i]=Y_past[i-lag]
        
        if conditions is not None:
            for i in range(2*lag,len(conditions_past)+2*lag):
                multi_data[i]=conditions_past[i-2*lag]
                
        #Make your models
        uni_regress_model = LinearRegression().fit(multi_Y, Y_fut)
        multi_regress_model = LinearRegression().fit(multi_data, Y_fut)
        
        #Make some predicitions
        Y_uni_fut_pred = uni_regress_model.predict(multi_Y)
        Y_multi_fut_pred = multi_regress_model.predict(multi_data)
        
        #Calculate the errors
        uni_error = Y_fut - Y_uni_fut_pred
        multi_error = Y_fut - Y_multi_fut_pred
        
        #Calculate the variances
        e_uni_hat = np.var(uni_error)
        e_multi_hat = np.var(multi_error)
        
        #Calculate Granger Causality
        GC = np.log(e_uni_hat/e_multi_hat)
        
        if self.verbose:
            end_GC=timeit.default_timer()
            runtime=end_GC-start_GC
            print('Time to Calculate GC:',runtime)
        
        return(GC)
    

    
    def GC_Matrix(self, data, lag, conditional=True):

        #Takes a Pandas dataframe and applies a function D_func to all columns in the dataframe
        col = data.columns
        n_col = len(col)
        
        results_matrix = np.zeros(shape=(n_col, n_col))
        
        #Iterate through the columns of your matrix (dataframe)
        i=0
        for column_i in tqdm(col, desc='Column i'):
            
            j=0
            for column_j in tqdm(col, desc='Column j', leave=False):
                # Make sure you aren't performing an operation on the same column
                if column_i != column_j:
                    if conditional:
                        result = self.linear_GC(data[column_i], data[column_j], lag=lag, conditions=data.drop(columns=[column_i,column_j]))
                    else: 
                        result = self.linear_GC(data[column_i], data[column_j], lag=lag)
                    results_matrix[i,j] = result
                j+=1
            i+=1
            
        return(results_matrix)
        