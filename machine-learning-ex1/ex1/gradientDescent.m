function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    temp_1 = theta(1) - alpha * (1/m)*sum_h(X,y,1,m,theta);
    temp_2 = theta(2) - alpha * (1/m)*sum_h(X,y,2,m,theta);

    theta = [temp_1; temp_2];


    % ============================================================

    % Save the cost J in every iteration    
    disp(computeCost(X, y, theta));
    J_history(iter) = computeCost(X, y, theta);

end

function sum_h = sum_h(X, y, j, m, theta)
    loop_count = 1;
    sum_h = 0;
    while(loop_count <= m)
       sum_h = sum_h + (theta(1) + theta(2)*X(loop_count, 2) - y(loop_count))*X(loop_count, j);
       loop_count = loop_count + 1; 
    end
end

end
