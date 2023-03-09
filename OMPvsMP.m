% Generate a random sparse signal
n = 1000; % signal length
k = 50; % sparsity
x = zeros(n,1);
idx = randperm(n,k);
x(idx) = randn(k,1);

% Generate a random sensing matrix
m = 500; % number of measurements
A = randn(m,n);

% Compress the signal
y = A*x;

% Set MP and OMP parameters
T = 1000; % maximum number of iterations
tol = 1e-6; % tolerance for residual

% Run MP algorithm
tic;
[coeff_mp, idx_mp] = MatchingPursuit(A, y, T, tol);
t_mp = toc;

% Run OMP algorithm
tic;
[coeff_omp, idx_omp] = OrthogonalMatchingPursuit(A, y, T, tol);
t_omp = toc;

% Compare results
x_mp = zeros(n,1);
for q = 1 : size(idx_mp)
    x_mp(idx_mp(q)) = coeff_mp(idx_mp(q));
end
x_omp = zeros(n,1);
for q = 1 : size(idx_omp)
    x_omp(idx_omp(q)) = coeff_omp(idx_omp(q));
end

fprintf('Matching Pursuit took %.4f seconds\n', t_mp);
fprintf('Orthogonal Matching Pursuit took %.4f seconds\n', t_omp);
fprintf('Matching Pursuit took %d iterations\n', size(idx_mp,1));
fprintf('Orthogonal Matching Pursuit took %d iterations\n', size(idx_omp,1));
fprintf('Matching Pursuit residual: %.4f\n', norm(y - A*x_mp));
fprintf('Orthogonal Matching Pursuit residual: %.4f\n', norm(y - A*x_omp));

function [coeff, idx] = MatchingPursuit(A, y, T, tol)
% Matching Pursuit algorithm
% Inputs:
%   A: sensing matrix (m x n)
%   y: compressed signal (m x 1)
%   T: maximum number of iterations
%   tol: tolerance for residual
% Outputs:
%   coeff: coefficient vector (n x 1)
%   idx: index vector of selected atoms (T x 1)

[~,n] = size(A);
r = y; % residual
idx = zeros(T,1); % index vector
coeff = zeros(n,1); % coefficient vector

for t = 1:T
    % Find the index of the atom with the largest correlation
    [~, i] = max(abs(A'*r));
    idx(t) = i;
    
    % Compute the coefficient of the selected atom
    % a == gr0
    % c == 
    g = A(:,i);
    c = g'*r/(g'*g);
    coeff(i) = coeff(i) + c;
    
    % Update the residual
    r = r - c*g;
    
    % Check if residual is small enough
    if norm(r) < tol
        break;
    end
end

idx = idx(1:t);

end

function [coeff, idx] = OrthogonalMatchingPursuit(A, y, T, tol)
% Orthogonal Matching Pursuit algorithm
% Inputs:
%   A: sensing matrix (m x n)
%   y: compressed signal (m x 1)
%   T: maximum number of iterations
%   tol: tolerance for residual
% Outputs:
%   coeff: coefficient vector (n x 1)
%   idx: index vector of selected atoms (T x 1)
[m,n] = size(A);
r = y; % residual
idx = zeros(T,1); % index vector
coeff = zeros(n,1); % coefficient
D0 = [];
% Iteratively select the atom that is most correlated with the residual
    for t = 1:T
        [~, i] = max(abs(A'*r));
        idx(t) = i;
        D0 = [D0, A(:,idx(t))];
        A(:,idx(t)) = zeros(m,1);
        % Compute the projection of the residual onto the selected atom
        % Update the coefficient and residual
        % coeff is == an , idx == r0, g = gr0
        coeff(idx(1:t)) = pinv(D0)*y;
        r = y - D0*coeff(idx(1:t));
        % Check if residual is small enough
        if norm(r) < tol
            break;
        end
    end
    idx = idx(1:t);
end
