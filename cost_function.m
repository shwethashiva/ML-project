function [ cost ] = cost_function( X, y, theta )
%Cost Function for Logistic regression

% Inputs: X      m x n data matrix
%         y      m x 1 vector of training outputs
%         theta  n x 1 vector of weights

z = theta' * X';
p = logistic(z);

y_0 = -y*log(p);
y_1 = (1-y)*log(p);

cost = y_0 - y_1;

end