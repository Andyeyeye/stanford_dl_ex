function [f,g] = logistic_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A column vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %

  m=size(X,2);
  n=size(X,1);
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));
  

  %
  % TODO:  Compute the objective function by looping over the dataset and summing
  %        up the objective values for each example.  Store the result in 'f'.
  %
  % TODO:  Compute the gradient of the objective by looping over the dataset and summing
  %        up the gradients (df/dtheta) for each example. Store the result in 'g'.
  %
%%% YOUR CODE HERE %%%
  J=0;
  fflush(1);
  %fprintf("start call(m=%d,n=%d)\n",m,n);
  %
  h_x=zeros(m,1);
  % h(x)
  for j = 1 : m
    h_x(j)=theta'*X(:,j);
    temp = 1 / (1 + exp(- h_x(j)));
    h_x(j)=temp;
  end
  
  %fprintf("h_x finish(%d x %d)\n",size(h_x,1),size(h_x,2));
  %fflush(1);
  %J(x)
  for j = 1 : m
    J = J + y(j) * log(h_x(j))+(1 - y(j))*log(1 - h_x(j));
  end
  J = -J;
  %fprintf("J finish(m=%d,n=%d)\n",m,n);
  %fflush(1);
  %nabla_J(X)
  
  for j = 1 : m
    h_x(j) = h_x(j) - y(j);
  end
  %fprintf("(h_x - y) finish(m=%d,n=%d)\n",m,n);
  %fflush(1);
  

  nabla_J = zeros(size(theta),1);
  for i = 1 : n
    nabla_J(i) = X(i,:) * h_x;
  end
  %fprintf("nabla_J finish(m=%d,n=%d)\n",m,n);
  %fflush(1);
  %output
  f=J;
  g=nabla_J;
