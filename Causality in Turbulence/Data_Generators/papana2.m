function data = papana2(N,scale)
%Data from Papana paper, system 2.

    data = zeros(N+5, 4);
    % Setting initial variables
    for i = 1:4
        data(1,i) = normrnd(0,scale);
    end
    for i = 1:4
        data(2,i) = normrnd(0,scale);
    end
    for i = 1:4
        data(3,i) = normrnd(0,scale);
    end
    for i = 1:4
        data(4,i) = normrnd(0,scale);
    end
    for i = 1:4
        data(5,i) = normrnd(0,scale);
    end
    
    % Now setting iterating through all points 
    for i = 6:N+5
       data(i,1) = 0.8*data(i-1,1) + 0.65*data(i-4,2) + normrnd(0,scale);
       data(i,2) = 0.6*data(i-1,2) + 0.6*data(i-5,4) + normrnd(0,scale);
       data(i,3) = 0.5*data(i-3,3) - 0.6*data(i-1,1) + 0.4*data(i-4,2) + normrnd(0,scale);
       data(i,4) = 1.2*data(i-1,4) - 0.7*data(i-2,4) + normrnd(0,scale);
    end
    data = data(6:end,:);
end

