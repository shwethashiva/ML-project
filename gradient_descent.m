function [ theta, J_history ] = gradient_descent( X, y, theta, alpha, iters  )
%GRADIENT_DESCENT
%
% Inputs:
%   X          m x n data matrix
%   y          m x 1 vector of training outputs
%   theta      n x 1 vector of parameters (initial values)
%   alpha      step size
%   iters      number of iterations
%
% Outputs:
%   theta      final learned parameter vector
%   J_history  the cost function history

J_history = [];

a = alpha;
lambda = 0.001;
% m = the number of training examples
m = length(y);

param = theta;

for i=1:iters
    %updating theta vector
    xparam_y = X*param-y;
    param = param - a*(X'*(logistic(X*param)-y));
    
end

theta = param;

%compute cost using log loss
J_history = sum(cost_function( X , y, theta));

end