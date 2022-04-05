"""
Calculates bin-width's using Darbellay's adaptive partition.

Steps:
    1. Calculate the equi-width point of the data. I.e, the point where the data is split to 0s and 1s and has an equi-probability of being either 0 or 1. 
    2. Calculate the Chi-Square significane for a 5% significance level with 1 degree of freedom (refer to equation 19 in https://www.mdpi.com/1099-4300/17/1/438)
    3. If the significance level is above 3.841, the cells must be partitioned further (repeat steps 1 and 2). If not, partitioning stops for this cell. (Significance vaue taken from https://www.mun.ca/biology/scarr/IntroPopGen-Table-D-01-smc.jpg)
    (In the case of a 511 DOF Chi-Square test, the significance value is 564.6961 from https://www.di-mgt.com.au/chisquare-table.html)
    
NOTE: This method of implenting transfer entropy only uses the data point of a current measure and the one of a measure prior. 

TODO: Implement multiprocessing similar to https://chriskiehl.com/article/parallelism-in-one-line
TODO: Currently the means are being found each time. Because the mean ensures that the data is split equiprobably into two sides, there is no need to worry about the size of certain columns not being the same as others in the function continuedPartitions. In the future, this may not be the case. It'll be necessary transpose the means to avoid any issues.
"""
import numpy as np
from warnings import warn
from itertools import product
from math import log2
from tqdm import tqdm
from scipy.stats import chi2

def darbellayTE(Y, X, lag=1, conditions = None, maxNumPartitions = 10):
    """
    Parameters
    ----------
    Y : numpy.array
        Source variable.
    X : numpy.array
        Target variable.
    lag : int, optional
        Lag-embedding. The default is 1.
    conditions : numpy.array, optional
        Array of conditions when calculating transfer entropy. Each conditions should be a column. The default is None.

    Returns
    -------
    TE : float64
        The transfer entropy of Y->X with lag (with conditions, if present.)
    

    Steps:
        (NOTE: The paper https://www.mdpi.com/1099-4300/17/1/438 will be henced referred to as 'the source paper')
        1) Create arrays that have the appropriate data representing the past of each variable (up to lag)
        2) Create two groups of mutual arrays for mutual information per the source paper
            (a) (X_{i+1},[X_i,Y_i])
            (b) (X_{i+1},X_i)
        3) Split each grouping using the childPartition function above. NOTE: Do not have more than 10 bins in any dimension
        4) Calcuate the mutual information per equation 21 in the source paper.
    """
    
    warn("Note that this method of calculating TE only considers the data at the lagged time, NOT all of the data up until that time.")
    con_pres = False
    if type(conditions) == np.ndarray: con_pres = True
    
    # Get sizes of arrays
    N = len(X)
    
    # Step 1) Organize the arrays
    Y_past = Y[:N-lag]
    X_past = X[:N-lag]
    X_fut  = X[lag:]
    if con_pres:
        conditions_past=conditions[:N-lag, :]
        conditions_past = conditions_past.T
    
    # Step 2) Create groupings for darbellay partioning
    group1 = np.vstack((X_fut,X_past,Y_past))
    
    if con_pres:
        group1 = np.vstack((group1, conditions_past))
        
    group1 = group1.T
    
    # Step 3) Conducting continued partitions until one of the terms is met
    cells1, _ = continuedPartitions(group1, maxNumPartitions)
    
    DigitData1 = np.zeros(N-lag, dtype=np.int)
    for index, cell in enumerate(cells1):
        if cell.Master:
            DigitData1[cell.subDummyKeys] = cell.cellNumber      
            
    # Step 4) Calculate mutual information
    N -= lag
    TE = 0 # Initiating TE here
    
    P1, C1 = np.unique(DigitData1, return_counts=True)
    
    # This part is update using equation 11 from https://link.springer.com/article/10.1186/1475-925X-11-19
    for index, cellNum in enumerate(P1):
        boundary = cells1[cellNum-1].boundaries
        a=C1[index]/N
        locB=np.where((np.delete(group1,[0,2],axis=1)>=np.delete(boundary,[0,2],axis=1)[0,:]).all(axis=1)&(np.delete(group1,[0,2],axis=1)<=np.delete(boundary,[0,2],axis=1)[1,:]).all(axis=1))[0]
        b=len(locB)/N
        locC = np.where((np.delete(group1,0,axis=1)>=np.delete(boundary,0,axis=1)[0,:]).all(axis=1)&(np.delete(group1,0,axis=1)<=np.delete(boundary,0,axis=1)[1,:]).all(axis=1))[0]
        c = len(locC)/N
        locD = np.where((np.delete(group1,2,axis=1)>=np.delete(boundary,2,axis=1)[0,:]).all(axis=1)&(np.delete(group1,2,axis=1)<=np.delete(boundary,2,axis=1)[1,:]).all(axis=1))[0]
        d = len(locD)/N
        TE += (a)*log2((a*b)/(c*d))
        
    return TE





def continuedPartitions(group, max_num_part=10):
    """
    Parameters
    ----------
    group : np.array
        The matrix of data that will be sub partitioned until all.
    max_num_part : int, optional
       Maximum number of partitions in any spatial direction. The default is 10.

    Returns
    -------
    digitized_data : np.array (1D, len=group.shape[0])
        .
    edges : np.array (2D, group.shape[1] x NumEdges)
        The edges for each column in the given group.
    """
    n = group.shape[1]
    keys = np.array(list(product([0,1],repeat=n)))
    sortedGroup = np.sort(group, axis=0)
    
    top_edge = np.max(group,axis=0)
    bottom_edge = np.min(group,axis=0)
    edges = np.sort(np.vstack((bottom_edge,top_edge)), axis=0)   
    
    cells = []
    partitions = []
    dummyKeys = np.array(list(range(group.shape[0])),dtype=np.int) # Used to keep track of which rows are falling into which cells
    
    # First cell append
    cellNum = 1
    firstMeans = np.mean(group,axis=0)
    kwargs = {'subgroup':group,
              'keys':keys,
              'boundaries':edges,
              'cellNumber':cellNum,
              'means':firstMeans, # This will only apply to the first cell
              'subDummyKeys':dummyKeys} #Used to keep track of which data falls into which cell
    cells.append(Cell(**kwargs))
    cellNum += 1

    for cell in cells:
        """
        This for loop will continue to add a cell into the cells list and will end for one of two reasons:
            (1) After iterating through each cell, it is decided that it should NOT be partitioned
            (2) The number of partitions in each spatial dimension is greater than the given value 'max_num_part'
        """
        if cell.partition==None:# If the cell has not been partitioneed yet
            Ddata, partition = cell.childPartition()
            partitions.append(partition)
            
            if partition:
                # Include these means into the edges, as they have been accepted.
                edges = np.sort((np.vstack((edges, cell.means))), axis=0) # But we must check if there are repeated values here
                newCells, _ = np.unique(Ddata, return_counts = True)
                
                for newCell in newCells:
                    key = keys[newCell] # Recall that the newCell is simply which combination of 0s and 1s from the keys matched the appropriate location.

                    # Now we must go through the main cells means and figure out what the upper and lower bounds for this group will be
                    subEdges = np.zeros((2,n))
                    for kIndex, k in enumerate(key):
                        if k: # 1
                            subEdges[:,kIndex] = cell.means[kIndex], cell.boundaries[1, kIndex]
                        else: # 0
                            subEdges[:, kIndex] = cell.boundaries[0,kIndex], cell.means[kIndex]
                    
                    # Indentifying cell means
                    # TODO: The <= was placed in this location arbitrarily. That should not be the case! Decide if the = is needed.
                    newMeans = np.zeros(n)
                    for column in range(n):
                        inRange = np.where((sortedGroup[:,column]>subEdges[0,column]) & (sortedGroup[:,column]<=subEdges[1,column]))[0]
                        newMeans[column] = np.mean(sortedGroup[inRange,column])
                    
                    newSubGroupLocations = np.where(Ddata == newCell)[0]
                    subDummyKeys = cell.subDummyKeys[newSubGroupLocations]
                    newSubGroup = cell.data[newSubGroupLocations,:]
                    kwargs = {'subgroup':newSubGroup,
                              'keys':keys,
                              'boundaries':subEdges,
                              'cellNumber':cellNum,
                              'means':newMeans,
                              'subDummyKeys':subDummyKeys}
                    cells.append(Cell(**kwargs))
                    cellNum+=1
           
            # Check if there are too many partitions in each spatial dimension
            # TODO: Make this portion a single command with np.unique that counts through each column, instead of iterating through each column.
            checks = np.zeros(n)
            for column in range(n):
                uniques = np.unique(edges[:,column])
                checks[column]=len(uniques)
            
            # the bottom will only be true if all dimensions have been partitioned at leas
            numPartitions = np.where(checks>(max_num_part),1,0)
            if 0 not in numPartitions:
                break
            
    return cells, edges




    

class Cell:
    def __init__(self, subgroup, keys, boundaries, cellNumber, means, subDummyKeys):
        self.data = subgroup
        self.means = means 
        self.partition = None
        self.keys = keys
        self.cellNumber = cellNumber
        self.Master = True
        self.boundaries = boundaries
        self.subDummyKeys = subDummyKeys
        
    def childPartition(self):
        """
        Parameters
        ----------
        dataframe : np.array
            2D array with each column representing another DOF of the system.
    
        Returns
        -------
        partition : bool
            TRUE if the partition is statistically independent (accept partition), FALSE otherwise
        means : np.array
            The location each partition took place for each respective column
        digitize_data : np.array
            An array that represents the original data in its partitioned form
        
        For computational efficiency, operations will be performed in numpy to the degree possible. Assume that the dataframe is a numpy 2D array
        """
        self.partition = False
        n = self.data.shape[1]
        num_cells = 2**n
        DOF = num_cells -1 # The degrees of freedom in this case
        
        total_cells = np.zeros(num_cells) # Captures how many datapoints are going into each cell. Used to calculate the Chi-Square significance, telling us if this partitioning of the data produces independent data.
        digitized_data = np.zeros(self.data.shape[0], dtype=np.int) # Organizes each combination of events into its respective digitized location
    
        for index, row in enumerate(self.data):
            # If the value in that cell is greater than the mean, then return 1, otherwise return 0.
            greater_than_mean = np.where(row >= self.means, 1, 0)
            
            # Find what index/cell this is using: https://stackoverflow.com/questions/40382384/finding-a-matching-row-in-a-numpy-matrix
            location = np.where((self.keys==greater_than_mean).all(axis=1))[0][0]
            total_cells[location] += 1
            digitized_data[index] = location
            
        # Now let's calculate if this captures the data in a unique way. Equation from https://www.mdpi.com/1099-4300/17/1/438
        # TODO: Reverify that this equation is being implemented properly.
        T = np.sum((total_cells - np.mean(total_cells))**2)
        
        # The threshold that the independent data must pass for the partition to be accepted
        sig_threshold = chi2.ppf(q=0.95, df = DOF)
        
        if T > sig_threshold: 
            self.partition = True
            self.Master = False
        else:
            pass # print('Cell partitioned DENIED')
            
        return(digitized_data, self.partition)

    



def TE_Matrix(data, lag, conditional=False, maxNumPartitions=10):
    """
    Calculates the causal heatmap using the Darbellay-Vajda adaptive partioning method

    Parameters
    ----------
    data : np.array
        2D array with each column representing another DOF of the system.
    lag: scalar
        Time-delay for transfer entropy calculation
    conditional: bool
        Whether to calculate the conditional transfer entropy.
    maxNumPartitions: scalar
        Maximum number of partitions for each variable

    Returns
    -------
    results_matrix : np.array
        Causal matrix with index [i,j] refering to the transfer entropy from i to j.
    """
    
    # Convert to numpy array in case it is not
    data = np.asarray(data)

    n_col = data.shape[1]
    
    results_matrix = np.zeros(shape=(n_col, n_col))
    
    #Iterate through the columns of your matrix (dataframe)
    i=0
    for column_i in tqdm(range(n_col), desc = 'Column i'):
        
        #DO IT AGAIN
        j=0
        for column_j in tqdm(range(n_col), desc='Column j', leave=False):
            
            # Make sure you aren't performing an operation on the same column
            if column_i != column_j:
                if conditional: 
                    result = darbellayTE(data[:,column_i], data[:,column_j], lag, conditions=np.delete(data,[column_i,column_j],axis=1),maxNumPartitions=maxNumPartitions)
                else: 
                    result=darbellayTE(data[:,column_i], data[:,column_j], lag, conditions=None, maxNumPartitions=maxNumPartitions)
                results_matrix[i,j] = result
            j+=1
        i+=1
        
    return(results_matrix)