function data = baccala5(N,scale)
%Data from baccala paper with 5 variables created
%   Refer to python script functions.py for more details

    data = zeros(N, 5);
    % Setting initial variables
    for i = 1:5
        data(1,i) = normrnd(0,scale);
    end
    for i = 1:5
        data(2,i) = normrnd(0,scale);
    end
    for i = 1:5
        data(3,i) = normrnd(0,scale);
    end
    
    % Now setting iterating through all points 
    for i = 4:N
       data(i,1) = 0.95*sqrt(2)*data(i-1,1) -0.9025*data(i-2,1) + normrnd(0,scale);
       data(i,2) = 0.5*data(i-2,1) + normrnd(0,scale);
       data(i,3) = -0.4*data(i-3,1) + normrnd(0,scale);
       data(i,4) = -0.5*data(i-2,1) + 0.25*sqrt(2)*data(i-1,4) + 0.25*sqrt(2)*data(i-1,5) + normrnd(0,scale);
       data(i,5) = -0.25*sqrt(2)*data(i-1,4) + 0.25*sqrt(2)*data(i-1,5) + normrnd(0,scale);
    end
end

