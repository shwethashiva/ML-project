function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
m = size(X, 1);
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J.

% Add ones to the X data matrix
X = [ones(m, 1) X];
% Convert y from (1-10) class into num_labels vector
yd = eye(num_labels);
 
%%% Map from Layer 1 to Layer 2
a1=X;
z2=X*Theta1';
% Sigmoid function converts to p between 0 to 1
a2=sigmoid(z2);

%%% Map from Layer 2 to Layer 3
% Add ones to the h1 data matrix
a2=[ones(m, 1) a2]; 
z3=a2*Theta2';
% Sigmoid function converts to p between 0 to 1
a3=sigmoid(z3);

% Compute cost
%logisf=(-y)'*log(a3)-(1-y)'*log(1-a3);
%y=[zeros(1000,11) y];
logisf=(-y).*log(a3)-(1-y).*log(1-a3); 

%% Regularized cost
Theta1s=Theta1(:,2:end);
Theta2s=Theta2(:,2:end);
J=((1/m).*sum(sum(logisf)))+(lambda/(2*m)).*(sum(sum(Theta1s.^2))+sum(sum(Theta2s.^2)));

%backpropagation
tridelta_1=0;
tridelta_2=0;

% Compute delta, tridelta and big D
	delta_3=a3-y;
    z2=[ones(m,1) z2];
	delta_2=delta_3*Theta2.*sigmoidGradient(z2);
    delta_2=delta_2(:,2:end);
	tridelta_1=tridelta_1+delta_2'*a1; % Same size as Theta1_grad (25x401)
    tridelta_2=tridelta_2+delta_3'*a2; % Same size as Theta2_grad (10x26)
	Theta1_grad=(1/m).*tridelta_1;
    Theta2_grad=(1/m).*tridelta_2;
    %Theta1_grad=0;
	%Theta2_grad=0;
%end
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end