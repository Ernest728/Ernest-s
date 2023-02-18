% Orthogonal Matching Pursuit for LEO Channel Estimation
% Assume the received signal is stored in a matrix called 'Y'
% The dictionary matrix is stored in a matrix called 'D'

% Initialize the channel estimate
h = zeros(size(D,2),1);

% Set the maximum number of iterations
max_iter = 10;

% Set the desired level of sparsity
s = 4;

for iter = 1:max_iter
    % Compute the residual
    r = Y - D*h;
    
    % Find the index of the largest magnitude correlation
    [~,idx] = max(abs(D'*r));
    
    % Add the corresponding column of the dictionary to the support set
    S(iter) = idx;
    
    % Solve the least squares problem using the support set
    h(S(1:iter)) = pinv(D(:,S(1:iter)))*Y;
    
    % Check for convergence
    if nnz(h) >= s
        break;
    end
end