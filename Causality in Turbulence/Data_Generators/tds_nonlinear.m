function data = tds_nonlinear(N,scale)
% Data created using the functions from the Towards Data Science article
%   Refer to python script functions.py (found in the causality folder) for
%   more details.

    data = zeros(N, 5);
    % Setting initial variables
    for i = 1:5
        data(1,i) = normrnd(0,scale);
    end
    % Now setting iterating through all points 
    for i = 2:N
       data(i,1) = 0.95*sqrt(2)*data(i-1,1) -0.9025*data(i-1,1) + normrnd(0,scale);
       data(i,2) = 0.5*data(i-1,1)^2 + normrnd(0,scale);
       data(i,3) = -0.4*data(i-1,1) + normrnd(0,scale);
       data(i,4) = -0.5*data(i-1,1)^2 + 0.25*sqrt(2)*data(i-1,4) + 0.25*sqrt(2)*data(i-1,5) + normrnd(0,scale);
       data(i,5) = -0.25*sqrt(2)*data(i-1,4) + 0.25*sqrt(2)*data(i-1,5) + normrnd(0,scale);
    end
end