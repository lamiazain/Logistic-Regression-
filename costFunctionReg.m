function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%my_code
% z=X * theta;
% h=1./(1+exp(-z));
% first_term=y.*log(h);
% second_term=(1-y).*log(1-h);
% reg_term=(lambda./(2*m)).*sum(theta.^2)
% J=((-1/m)*sum(first_term+second_term))+reg_term
% 
% grad(1)=(1/m)*sum((h-y).*X(:,1));
% for i=2:1:28
%     grad(i)=((1/m)*sum((h-y).*X(:,i)))+((lambda/m).*theta(i));
% end
% 


  z = X * theta;      % m x 1
  h_x = sigmoid(z);  % m x 1 
  
  reg_term = (lambda/(2*m)) * sum(theta(2:end).^2);
  
  J = (1/m)*sum((-y.*log(h_x))-((1-y).*log(1-h_x))) + reg_term; % scalar
  
  grad(1) = (1/m)* (X(:,1)'*(h_x-y));                                  % 1 x 1
  grad(2:end) = (1/m)* (X(:,2:end)'*(h_x-y))+(lambda/m)*theta(2:end);  % n x 1
  

% =============================================================

end
