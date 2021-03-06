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

k_classes = (unique(y));
K = size(k_classes, 1);
y_relabel = zeros(m,size(k_classes,1));
for i= 1:m
    y_relabel(i, find(k_classes==y(i))) = 1;
endfor

[a2 a3 z2 z3] = h(Theta1, Theta2, X);
for i = 1:m
for k = 1:K
    J = J + -y_relabel(i, k) * log(a3(k,i)) - (1 - y_relabel(i,k)) * log(1 - a3(k,i));
endfor
endfor

J = 1/m*J;
[size1 size2] = size(Theta1);
reg_factor = 0;
for j=1:size1
    for k=1:size2-1
        reg_factor = reg_factor + Theta1(j,k+1)*Theta1(j,k+1);
    endfor
endfor

[size1 size2] = size(Theta2);
for j=1:size1
    for k=1:size2-1
        reg_factor = reg_factor + Theta2(j,k+1)*Theta2(j,k+1);
    endfor
endfor
reg_factor = lambda/(2*m) * reg_factor;
J = J + reg_factor;



tri_1 = zeros(size(Theta1));
tri_2 = zeros(size(Theta2));
for t= 1:m
    a_1 = X(t,:);
    [a_2 a_3 z_2 z_3] = h(Theta1, Theta2, a_1);
    a_1 = a_1';
    a_1 = [1; a_1];
    delta_3 = (a_3 - y_relabel(t,:)');
    delta_2 = Theta2'*delta_3;
    delta_2 = delta_2(2:end) .* sigmoidGradient(z_2);
    tri_1 = tri_1 + delta_2 * a_1';
    tri_2 = tri_2 + delta_3 * a_2';
endfor

reg_term = lambda/m .* Theta1;
reg_term(:,1) = 0;
Theta1_grad = 1/m*tri_1 + reg_term;
reg_term2 = lambda/m .* Theta2;
reg_term2(:,1) = 0;
Theta2_grad = 1/m*tri_2 + reg_term2;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end


function [a2 a3 z2 z3] = h(Theta1, Theta2, X)
m = size(X, 1);
num_labels = size(Theta2, 1);
p = zeros(size(X, 1), 1);
layers = 3;
for i=1:layers
    if i == 1
        a1 = [ones(m, 1) X];
        a1 = a1';
    elseif i == layers
        z3 = Theta2*a2;
        a3 = sigmoid(z3);
    else
        z2 = Theta1*a1;
        z2 = z2';
        a2 = [ones(m,1) sigmoid(z2)]; 
        z2 = z2';
        a2 = a2';
    endif
endfor

% =========================================================================


end
