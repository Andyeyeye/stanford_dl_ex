function [f,g] = linear_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The target value for each example.  y(j) is the target for example j.
  %
  
  m=size(X,2);
  n=size(X,1);

  f=0;
  g=zeros(size(theta));

  %
  % TODO:  Compute the linear regression objective by looping over the examples in X.
  %        Store the objective function value in 'f'.
  %
  % TODO:  Compute the gradient of the objective with respect to theta by looping over
  %        the examples in X and adding up the gradient for each example.  Store the
  %        computed gradient in 'g'.
  
%%% YOUR CODE HERE %%%
  fflush(1);
  %h_x=zeros(size(theta));
  J=0;
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
  
  f=J;
  g=nabla_J;
