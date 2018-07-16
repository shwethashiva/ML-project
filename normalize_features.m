function [ X, mu, sigma ] = normalize_features( X, mu, sigma )
%NORMALIZE_FEATURES Normalize the features
%
%
%  [X_train, mu, sigma] = normalize_features(X_train);
%  X_test = normalize_features(X_test, mu, sigma)
%

if nargin < 2
    mu = mean(X);
    sigma = std(X);
end

% If sigma = 0 set it to 1
sigma(sigma==0)=1;

for j=1:size(X,2)
    X(:,j) = (X(:,j) - mu(j))/sigma(j);
end

end
