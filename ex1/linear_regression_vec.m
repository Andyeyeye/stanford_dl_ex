function [f,g] = linear_regression_vec(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The target value for each example.  y(j) is the target for example j.
  %
  m=size(X,2);
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));

  %
  % TODO:  Compute the linear regression objective function and gradient 
  %        using vectorized code.  (It will be just a few lines of code!)
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %
%%% YOUR CODE HERE %%%
  J=0;
  fprintf("start call\n");
  for j = 1:m
    h_x=0;
    for i = 1:n
      h_x=h_x+theta(i)*X(i,j);
    end
    
    J=J+(h_x-y(j))*(h_x-y(j));
    
  end
  J=J/2;
  nabla_J=zeros(size(theta));
  for j = 1:m
    h_x=0;
    for i = 1:n
      h_x=h_x+theta(i)*X(i,j);
    end
    for i = 1:n
      nabla_J(i)=nabla_J(i)+X(i,j)*(h_x-y(j));
    end
  end
  fprintf("finish call\n");  
  f=J;
  g=nabla_J;
