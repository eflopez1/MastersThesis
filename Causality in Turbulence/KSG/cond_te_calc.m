clc; clear all

fullScriptTime = tic;

% JIDT Loc:
jarLoc = "path/to/JAR";

% Defining a location to save results
saveLoc = 'SaveFolder/';
mkdir(saveLoc);
contents = dir(saveLoc);
names_completed = {contents.name};

% Saving this execution file
filename = [mfilename];
t = datetime('now');
t_str = datestr(t);
t_str = strrep(t_str, ':', '_');
backup = char(compose('%s/%s %s.txt', saveLoc, filename, t_str));
currentfile = strcat(filename, '.m');
copyfile(currentfile, backup);

% Parameters for transfer entropy calculation
lags = [5];
embedding = 1; % Tau value, or embedding delay to consider
len = 'lag'; % Length of past to consider. Can also set to 'lag'.
%{
For now, when setting len, change the parameter in the function below
DIRECTLY to the lag if you want to actually use lag as your length.
%}
numNeighbors =  [10];
numPermutations = 30;
name='Lag%d_Embed%d_Length%d_K%d_%dPermutations';

% Creating Data
N = 10000;
scale = 1.0;
data = papana2(N,scale); % In this example, the data generator papana is used
data_func = 'papana2';
[numRow, numCol] = size(data);
numCond = numCol - 2;
% scales = [100,90,50,10,1];
% for i = 1:numCol
%    data(:,i) = data(:,i)*scales(i);
% end
%-------------------------------------------------
% Everything below this comment does not need to be touched!
%-------------------------------------------------

% Save the data used in for later analysis if needed
data_file = 'Data %s';
data_file = compose(data_file, t_str);
data_loc = string(append(saveLoc, data_file));
save(data_loc, 'data');

% Save the function used to create the data
currentfile = append(data_func,'.m');
backup_name = append(data_func,'.txt');
newbackup = append(saveLoc, backup_name);
copyfile(currentfile, newbackup);


% Loading JIDT Library
javaaddpath(jarLoc)

% Creating the TE Calculator
teCalc = javaObject('infodynamics.measures.continuous.kraskov.ConditionalTransferEntropyCalculatorKraskov');

for lag_index = 1:length(lags)
   for k_index = 1:length(numNeighbors)

       k = numNeighbors(k_index);
       lag = lags(lag_index);
       if strcmp(len,"lag")
           len=lag;
           len_is_lag = true;
       else
           len_is_lag = false;
       end

       thisName = compose(name,lag,embedding,len,k,numPermutations);

       % Check if this file has been calculated for already
       if any(contains(names_completed,thisName))
           msg = ["Skipping sequence filename: ", thisName];
           disp(msg)
           continue
       end

       resultMatrix = zeros(numCol);
       significanceMatrix = zeros(numCol);
       effecMatrix = zeros(numCol);

       % Timing of this matrix
       singleMatrixRun = tic;

       % Iterate through each column
        for i = 1:numCol % Source Array
            for j = 1:numCol % Destination Array
                if i~=j % Ignoring self-causal interactions
                    singleTE = tic;
                    sourceArray = data(:,i);
                    destArray = data(:,j);
                    colsRemove = [i j];
                    condData = data;
                    condData(:,colsRemove) = [];
                    teCalc.initialise(len,... % k - Length of destination past history to consider
                        embedding,...         % k_tau - Embedding dimension for the destination variable
                        len,...               % l - length of source past history to consider
                        embedding,...         % l_tau - embedding delay for the source variable
                        lag,...               % delay - time lag between last element of source and destination next value
                        ones(1,numCond)*len,...
                        ones(1,numCond)*embedding,...
                        ones(1,numCond)*lag);
                    teCalc.setProperty('k',string(k));
                    teCalc.setObservations(sourceArray, destArray, condData);
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
        end % End of the matrix

        if len_is_lag
            len='lag';
        end

        % Store the results
        fileLoc = append(saveLoc, thisName);
        fileLoc = char(fileLoc);
        save(fileLoc, 'resultMatrix', 'significanceMatrix', 'effecMatrix');

        endMatrixRun = toc(singleMatrixRun);
        message = 'Runtime for most recent causal matrix: %4.2f seconds\n';
        fprintf(message, endMatrixRun);

   end
end


endScriptTime = toc(fullScriptTime);
message = 'Runtime for entire script: %4.2f seconds\n';
fprintf(message,endScriptTime);

