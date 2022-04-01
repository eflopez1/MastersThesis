clc; clear all

fullScriptTime = tic;
% JIDT Loc:
jarLoc = "path/to/JAR";


% End data file name
endStr = '.mat';

% Loops are used to iterate through several parameters
IDs = [1,2,3]; % It is assumed that several files with unique IDs are used
lags = [1000];
embedding = 1; % Tau value, or embedding delay to consider
len = 20; % Length of past to consider. Can also set to 'lag'.
numNeighbors =  [4];
numPermutations = 0;
name='ID%d_Lag%d_Embed%d_Length%d_K%d_%dPermutations';

% Loading JIDT Library
javaaddpath(jarLoc)

% Creating the TE Calculator
teCalc = javaObject('infodynamics.measures.continuous.kraskov.ConditionalTransferEntropyCalculatorKraskov');

% Make sure the save folder is present
saveLoc = 'SaveFolder/';
mkdir(saveLoc);
contents = dir(saveLoc);
names_completed = {contents.name};

% Calculate and store results
for lagIndex = 1:length(lags)
    for neighborIndex=1:length(numNeighbors)
        for IDIndex=1:length(IDs)
            
            numNeighbor = numNeighbors(neighborIndex);
            lag = lags(lagIndex);
            singleMatrixRun = tic;
            ID = seqNums(IDIndex);
            if strcmp(len,"lag")
                len=lag;
                len_is_lag=true;
            else
                len_is_lag=false;
            end
            thisName = compose(name,ID,lag,embedding,len,numNeighbor,numPermutations);
            
            % Check if this file has been completed already
            if any(contains(names_completed,thisName))
                msg = ["Skipping ID number: ", num2str(ID)];
                disp(msg)
                continue
            end
            
            
            
            % Loading data
            file = append(beginStr,string(seqNum),endStr);
            data = load(file);
            data = data.testSeq;
            [numRow, numCol] = size(data);

            resultMatrix = zeros(numRow);
            significanceMatrix = zeros(numRow);
            effecMatrix = zeros(numRow);
            condEmbedding = ones(1,numRow-2);
            condDelay = ones(1,numRow-2)*lag;
            for i = 1:numRow %Source Array
                for j = 1:numRow %Destination Array
                    if i~=j %Ignoring self-causal investigation
                        singleTE = tic;
                        sourceArray = data(i,:);
                        destArray = data(j,:);
                        rowsRemove = [i j];
                        condData = data;
                        condData(rowsRemove,:) = [];
                        teCalc.initialise(len,... % k - Length of destination past history to consider
                            embedding,...         % k_tau - Embedding dimension for the destination variable
                            len,...               % l - length of source past history to consider
                            embedding,...         % l_tau - embedding delay for the source variable
                            lag,...               % delay - time lag between last element of source and destination next value
                            ones(1,numRow-2)*len,...
                            ones(1,numRow-2)*embedding,...
                            ones(1,numRow-2)*lag);
                        teCalc.setProperty('k',string(numNeighbor));
                        teCalc.setObservations(sourceArray', destArray', condData');
                        condResult = teCalc.computeAverageLocalOfObservations();
                        nullDistrib = teCalc.computeSignificance(numPermutations);
                        pVal = nullDistrib.pValue;
                        distrib = nullDistrib.distribution;
                        effecResult = mean(distrib);
                        
                        % Appending results to proper matrix
                        resultMatrix(i,j) = condResult;
                        significanceMatrix(i,j) = pVal;
                        effecMatrix(i,j) = effecResult;
                        
                        endTE = toc(singleTE);
                        fprintf('Single TE Calc time: %4.2f seconds\n',endTE);
                    end
                end
            end %This end designates the end of the matrix
            
            if len_is_lag
                len="lag";
            end
            
            %Save that matrix!
            fileLoc = append(saveLoc,thisName);
            fileLoc=char(fileLoc);
            save(fileLoc,'resultMatrix','significanceMatrix','effecMatrix');

            endMatrixRun = toc(singleMatrixRun);
            message = 'Runtime for most recent causal matrix: %4.2f seconds\n';
            fprintf(message,endMatrixRun);
        end
    end
end

endScriptTime = toc(fullScriptTime);
message = 'Runtime for entire script: %4.2f seconds\n';
fprintf(message,endScriptTime);
