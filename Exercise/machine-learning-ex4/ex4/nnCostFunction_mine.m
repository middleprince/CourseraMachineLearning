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

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


%part1 the cost withour regularization

%compute ai and y
a_1 = [ones(m,1),X];
z_2 = a_1 * Theta1';
a_2 = [ones(m,1),sigmoid(z_2)];
z_3 = a_2 * Theta2';
a_3 = sigmoid(z_3);   

log_a3 = log(a_3);
log_a3_minus = log(1 - a_3);

%recode y
yeye = eye(num_labels);
y_recode = yeye(y,:);

% the for-loop is less effecient,try not using it

%using for-loop to compute the cost J
for i=1:m
    for j=1:num_labels
        J += - y_recode(i,j) * log_a3(i,j) - ((1 - y_recode(i,j)) * log_a3_minus(i,j));
    endfor
endfor

J /= m;          %unregulization cost J

%the regularization term for any layers


Theta1_temp = [zeros(size(Theta1,1), 1) Theta1(:,2:end)];    %don't regularize the bais terms
Theta2_temp = [zeros(size(Theta2,1), 1) Theta2(:,2:end)];    %don't regularize the bais terms

%using for loop to compute the regularization terms.
regular_term = 0;

for i=1:size(Theta1,1)
    for j=1:size(Theta1,2)
        regular_term += Theta1_temp(i, j )^2;
    endfor
endfor

for i=1:size(Theta2,1)
    for j=1:size(Theta2,2)
        regular_term += Theta2_temp(i, j)^2;
    endfor
endfor
     
regular_term *= lambda/(2*m);

J += regular_term;


%implement the gradient term
deltaf_2 = zeros(size(Theta2_grad));  %initial the accumalater
deltaf_1 = zeros(size(Theta1_grad));

delta_3 = a_3 - y_recode;

tmp  = [ones(m,1) z_2]; 
delta_2 = delta_3*Theta2 .* sigmoidGradient(tmp);   %add the bias term to g'(z) 

deltaf_1 = (delta_2(:,2:end))' * a_1;          %remove the bias term to compute delta.

deltaf_2 = delta_3' * a_2;

Delta_1 = 1/m * deltaf_1;
Delta_2 = 1/m * deltaf_2;

%implement regularization term
theta1_tmp = [zeros(size(Theta1,1),1) Theta1(:,2:end)];
theta2_tmp = [zeros(size(Theta2,1),1) Theta2(:,2:end)];

Theta1_grad = Delta_1 + lambda*theta1_tmp/m;
Theta2_grad = Delta_2 + lambda*theta2_tmp/m;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
