% Generate a random sparse signal
N = 2000; % signal length
k = 50; % sparsity
mp_iterations = zeros(N, 1);
omp_iterations = zeros(N, 1);
t_mps = zeros(N, 1);
t_omps = zeros(N, 1);

for n = 1000: N
t_temp_mp = 0;
temp_mp_iterations = 0;
temp_omp_iterations = 0;
t_temp_omp = 0;

    for it = 1:100
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
[coeff_mp, idx_mp, num_iter_mp] = MatchingPursuit(A, y, T, tol);
t_mp = toc;

% Run OMP algorithm
tic;
[coeff_omp, idx_omp, num_iter_omp] = OrthogonalMatchingPursuit(A, y, T, tol);
t_omp = toc;

% Compare results
t_temp_mp = t_temp_mp + t_mp;
temp_mp_iterations = temp_mp_iterations + num_iter_mp;
temp_omp_iterations = temp_omp_iterations +num_iter_omp;
t_temp_omp = t_temp_omp + t_omp;
    end
mp_iterations(n) = temp_mp_iterations/100;
omp_iterations(n) = temp_omp_iterations/100;
t_mps(n) = t_temp_mp/100;
t_omps(n) = t_temp_omp/100;
end

% Plot the results
plot(1:N, mp_iterations, 'b', 1:N, omp_iterations, 'r');
xlabel('Signal length');
ylabel('Iterations');
legend('Matching Pursuit', 'Orthogonal Matching Pursuit');
figure;
plot(1:N, t_mps, 'b', 1:N, t_omps, 'r');
xlabel('Signal length');
ylabel('Time');
legend('Matching Pursuit', 'Orthogonal Matching Pursuit');

function [coeff, idx, num_iter] = MatchingPursuit(A, y, T, tol)
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
    a = A(:,i);
    c = a'*r/(a'*a);
    coeff(i) = coeff(i) + c;
    
    % Update the residual
    r = r - c*a;
    
    % Check if residual is small enough
    if norm(r) < tol
        break;
    end
end

idx = idx(1:t);
num_iter = t;
end

function [coeff, idx, num_iter] = OrthogonalMatchingPursuit(A, y, T, tol)
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
    num_iter = t;
end
