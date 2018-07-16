function [ p ] = logistic( z )
%The logistic function
p = [];
% e
e= exp(1);
% logistic function
p = 1./(1+e.^(-z));

end