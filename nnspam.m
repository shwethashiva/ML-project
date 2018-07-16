% Name: Shwetha Shivaraju

%% Initialization
clear ; close all; clc

%% nn parameters
input_layer_size  = 2001;  % 2001 Input dicts of texts
hidden_layer_size = 25;   % 25 hidden units
num_labels = 1;          % 12 labels, from 1 to 12   
                          
fprintf('Loading preprocessed Data ...\n')

%load('ex4data1.mat');
load('sms','X_test','X_train','y_test','y_train'); 
X_train = normalize_features(X_train);

fprintf('\nInitializing nn Parameters ...\n')

initial_Theta1 = initializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = initializeWeights(hidden_layer_size, num_labels);


initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

fprintf('\nTraining Neural Network... \n')

options = optimset('MaxIter', 50);


lambda = 3.5;

%cost function
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X_train, y_train, lambda);


[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Theta1 and Theta2 from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

%displayData(Theta1(:, 2:end));

pred = predict(Theta1, Theta2, X_train);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y_train)) * 100);

%Test the model 
costFunctionTest = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X_test, y_test, lambda);
[nn_paramsTest, costTest] = fmincg(costFunctionTest, initial_nn_params, options);
% Theta1 and Theta2 from nn_params
Theta1Test = reshape(nn_paramsTest(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2Test = reshape(nn_paramsTest((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

testResult = predict(Theta1Test, Theta2Test, X_test);
fprintf('\nTesting Set Accuracy: %f\n', mean(double(testResult ==y_test)) * 100);
confusionmatrix = confusionmat(testResult,y_test);
