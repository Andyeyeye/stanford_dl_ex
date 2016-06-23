function [f,g] = softmax_regression_vec(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  num_classes=size(theta,2)+1;
  
  %full extra col to let running better
  %theta_full=zeros(n,num_classes);
  %theta_full=[theta';zeros(1,n)]';

  
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));

  %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
%%% YOUR CODE HERE %%%
  
  J=0;
  fflush(1);
  %fprintf("start call(m=%d,n=%d)\n",m,n);
  %fflush(1);
  %h_x=zeros(m,num_classes); 
  %****************h(x)****************
  
  h_x_mix = X' * theta;
  %fprintf("h_x_mix finish(%d x %d x %d)\n",size(h_x_mix,1),size(h_x_mix,2),size(h_x_mix,3));
  %fflush(1);

  h_x_exp = exp(h_x_mix);

  %fprintf("h_x_exp finish(%d x %d x %d)\n",size(h_x_exp,1),size(h_x_exp,2),size(h_x_exp,3));
  %fflush(1);
  h_x_exp_rev=h_x_exp';

  %%%
  %The reason that here add [1...] to h_x_exp_rev:
  %In theta is that k[1..9] and all k[10]=0
  %then in "h_x_mix", is (m*n) * (n * (10-1)) => m*(10-1),
  %so here h_x_mix[:,10] = X'* theta[:,10] = X' * zeros(n,1)= zeros(m,1);
  %because e^0=1
  %so h_x_exp[:,10]=exp(zeros(m,1))=ones(m,1);
  %so h_x_exp_rev[10,:]=ones(1,m)
  %%%

  h_x_exp_rev=[h_x_exp_rev;ones(1,m)];

  h_x = bsxfun(@rdivide,h_x_exp_rev,sum(h_x_exp_rev)); 
  
  
  
  %fprintf("h_x finish(%d x %d x %d)\n",size(h_x,1),size(h_x,2),size(h_x,3));
  %fflush(1);
  %****************J(x)****************
  h_x_log=log(h_x);
  temp1=sub2ind(size(h_x_log), y,1:size(h_x_log,2));
  J = -sum(h_x_log(temp1));

  %fprintf("J finish(m=%d,n=%d)\n",m,n);
  %fflush(1);
  %****************nabla_J(X)********************
  y_k=zeros(m,num_classes);
  for j = 1 : m
    y_k(j,y(j))=1;
  end
  %fprintf("j_k finish(m=%d,n=%d)\n",m,n);
  %fflush(1);
  nabla_J = zeros(size(theta));


  %nabla_J(i,k) = -sum(j = 1 .. m){x(i,j) * ((y(j) == k ? 1 : 0) - h_x(j,k)}
  %nabla_J(i,k) = -sum(j = 1 .. m){x(i,j) * (y_k(j,k) - h_x(j,k)}
  %              = ( x(i,:) * ( h_x(:,k)-y_k(:,k) )   )


  h_x = h_x';
  h_x = h_x(:,1:num_classes-1);
  y_k = y_k(:,1:num_classes-1);
  h_x = h_x - y_k;
  nabla_J=X*h_x;

  %fprintf("nabla_J finish(%d x %d x %d)\n",size(nabla_J,1),size(nabla_J,2),size(nabla_J,3));
  %fflush(1);
  %****************output****************
  f=J;
  g=nabla_J(:); % make gradient a vector for minFunc
