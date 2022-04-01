import pandas as pd
import numpy as np
from numpy.random import shuffle
# import seaborn as sb
from math import log10, log2, log, floor # Switching between which log is used changes the unit of your transfer entropy
from collections import Counter
import matplotlib.pyplot as plt #For plotting the heatmap
from numpy import savetxt, mean, std
import os
import timeit
from tqdm import tqdm

class TransferEntropy:
    def __init__(self, discrete_type = "quantiles", bins = 10, quantiles = [25,75]):
        """
        Options for discrete_type:
            -bins (equal-width bins)
            -quantiles 
            -fd (Freedman-Diaconis)
        """
        
        potential_discretes=['bins','quantiles','fd']
        assert discrete_type in potential_discretes, "Must pick a correct discrete type!"
        
        self.discrete_type = discrete_type
        self.nbins = bins
        self.quantiles = quantiles

    def calc_te(self, Y, X, lag, conditions=None, perm=False):
        """
        Calculates Transfer Entropy from Y to X (with conditions if wanted)
        
        Notes:
            -Conditions must be a pandas dataframe with shape (N,#Condtions) where N is len(X)
            -Perm is still experimental, DO NOT USE
            -Portion calculates the transfer entropy with only a portion of the given data. Value should be between [0,1], representing the portion of data to be used.
        """
        start_time=timeit.default_timer()

        con_pres=False        
        if type(conditions)==pd.DataFrame: con_pres=True
        
        N=len(X)
        if con_pres: N_conditions = min(conditions.shape)
        
        # Step 1) Convert all data to its discrete form
        if self.discrete_type=='fd':
            Y=self.discrete(Y)
            X=self.discrete(X)
        else:
            Y=self.discrete(Y).to_numpy()
            X=self.discrete(X).to_numpy()
        if con_pres:
            dis_con=np.zeros((conditions.shape))
            ind=0
            for i,j in conditions.iteritems():
                dis_con[:,ind]=self.discrete(j)
                ind+=1
                
        # Reformatting the conditions for later use
        if con_pres: dis_con = dis_con.T 
        
        # Step 2) Gather Proper Matrices to calculate probabilities
                # Remember, we need ALL the vectors up back in time up to the lag time
        X_past = np.zeros((lag,N-lag))
        Y_past = np.zeros((lag,N-lag))
        X_fut= X[lag:]
        if con_pres: conditions_past = np.zeros((lag*N_conditions,N-lag))
        
        for l in range(lag):
            X_past[l,:]=X[l:N-lag+l]
            Y_past[l,:]=Y[l:N-lag+l]
            
            if con_pres:
                for ind in range(N_conditions):
                    conditions_past[N_conditions*l+ind,:]=dis_con[ind][l:N-lag+l]
                    
        # Shuffling the past of Y in-place!
                
        if perm:
            # A new term will be included requiring shuffling to take place until at least 40% of the data has been shuffled
            print('Permutating Y Now!!!')
            np.random.shuffle(Y_past)
            np.random.shuffle(Y_past)
            np.random.shuffle(Y_past)
        
        # Step 3) The Big One. Make those probability tables
        """
        TE_{Y->X|W} where W is a matrix of conditions is calculated by:
            H(Xt|Xt-1,Wt-1)-H(Xt|Xt-1,Yt-1,Wt-1)
            
        Where H(Xt|Xt-1,Wt-1) is calculated with:
            H(Xt,Xt-1,Wt-1) - X(Xt-1,Wt-1)
            
        Multivariate entropy terms such as H(A,B,C) are calculated by:
            -sum[p(a,b,c)*log(p(a,b,c))]
            where we sum over all possible events.
            
        Thus, for a single lag, we need 4 tables:
            P1 = P(Xt,X1-1,Wt-1)
            P2 = P(Xt-1,Wt-1)
            P3 = P(Xt,Xt-1,Yt-1,Wt-1)
            P4 = P(Xt-1,Yt-1,Wt-1)
        """
        
        if con_pres:
            P1 = np.vstack((X_fut,X_past,conditions_past))
            P2 = np.vstack((X_past,conditions_past))
            P3 = np.vstack((X_fut,X_past,Y_past,conditions_past))
            P4 = np.vstack((X_past,Y_past,conditions_past))
            
        else:
            P1 = np.vstack((X_fut,X_past))
            P2 = np.vstack((X_past))
            P3 = np.vstack((X_fut,X_past,Y_past))
            P4 = np.vstack((X_past,Y_past))
        
        P1, C1 = np.unique(P1,axis=1,return_counts=True)
        C1=C1/sum(C1)
        P2, C2 = np.unique(P2,axis=1,return_counts=True)
        C2=C2/sum(C2)
        P3, C3 = np.unique(P3,axis=1,return_counts=True)
        C3=C3/sum(C3)
        P4, C4 = np.unique(P4,axis=1,return_counts=True)
        C4=C4/sum(C4)
        
        # Step 4) Calculate individual entropy values:
        H1=0
        H2=0
        H3=0
        H4=0
        for c1 in C1:
            H1 += c1*log2(c1)
        for c2 in C2:
            H2 += c2*log2(c2)
        for c3 in C3:
            H3 += c3*log2(c3)
        for c4 in C4:
            H4 += c4*log2(c4)
        
        # Don't forget to take the negative!
        H1 = -H1
        H2 = -H2
        H3 = -H3
        H4 = -H4
        
        Transfer_Entropy = (H1 - H2) - (H3 - H4) # Using the difference of entropies 
        
        end=timeit.default_timer()
        # print('TE Calculation Time:',end-start_time) # Can uncomment if you wish to see the runtimes actively.
        return Transfer_Entropy
        
    def discrete(self, X1):

        if self.discrete_type == "bins":
            discrete_X = pd.cut(X1, bins=self.nbins, labels=False)
        
        if self.discrete_type == 'quantiles':
            q=[]
            self.quantiles.sort()
            for num in self.quantiles:
                q.append(num/100)
            q.append(1.0)
            q.insert(0, 0.0)
            
            discrete_X = pd.qcut(X1, q=q, labels=False)
        
        if self.discrete_type == 'fd':
            X=X1.to_numpy()
            dis_binX = np.histogram_bin_edges(X, bins='fd')
            discrete_X = np.digitize(X,dis_binX)
        
        return discrete_X
    
    def TE_Matrix(self, data, lag, conditional=False, effec=False, numPerm=10, section = 1, effecOnly=False):
        
        if effecOnly: 
            assert effec==effecOnly, "If calculating only effective transfer entropy, variable 'effec' must also be True!"
            print('\n','--'*20)
            print('The resultant matrix will only have negative results. This is to be expected! Since you are calculating effective only, we will be subtracting from 0. In order to obtain the proper transfer entropy, you will need to manually sum this result with whatever non-effective transfer entropy has already been calculated')
            print('--'*20,'\n')
        
        #Takes a Pandas dataframe and applies a function D_func to all columns in the dataframe
        data = pd.DataFrame(data)
        col = data.columns
        n_col = len(col)
        
        results_matrix = np.zeros(shape=(n_col, n_col))
        
        #Iterate through the columns of your matrix (dataframe)
        i=0
        for column_i in tqdm(col,desc='Outer Columns'):
            
            #DO IT AGAIN
            j=0
            for column_j in tqdm(col, desc='Inner Columns', leave=False):
                
                # Make sure you aren't performing an operation on the same column
                if column_i != column_j:
                    
                    #Calculating the original transfer entropy on its own
                    if not effecOnly:
                        if conditional: result = self.calc_te(data[column_i], data[column_j], lag, conditions=data.drop(columns=[column_i,column_j]))
                        else: result=self.calc_te(data[column_i], data[column_j], lag)
                    else: result = 0
                    result_perm=0
                    
                    if effec==True:
                        permutations = np.zeros(numPerm)
                        for ind in range(numPerm):
                            permutations[ind] =  self.calc_te(data[column_i], data[column_j], lag, conditions=data.drop(columns=[column_i,column_j]), perm=True)
                        result_perm = mean(permutations)
                    results_matrix[i,j] = result - result_perm
                j+=1
            i+=1
            
        return(results_matrix)