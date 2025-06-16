function [x_minus,y_minus,Pxx,Pyy,Pxy] = UKF_DS_Dubin(x,P,f,Q,R)

%{
%%%%%% UKF %%%%%%%%%%
 Dynamic Prediction Step for a UKF
   Inputs:
       x0: Initial State
       P0: Initial State Covariance
       f: Dynamics Model
       Q: State Noise Model
       R: Measurment Noise Model
       

   Outputs:
       X: 6x1 state estimation
       P: 6x6 state covaraince
%}


n = 5; % # of states
alpha = 1; % Large a -> SP more spread out
kappa = 1000; %Typical Values
beta = 0; %Typical Values

%Tuning Parameter
Lambda = alpha^2*(n+kappa)-n;

% Generate weights for each sigma point
w0m = Lambda/(n+Lambda); % Weight the original mean the most
w0c = Lambda/(n+Lambda) + 1 - alpha^2+beta;

wim = 1/(2*(n+Lambda));
wic = wim;

meanWeights = ones(1,2*n+1);
covWeights = ones(1,2*n+1);

meanWeights(1) = w0m*meanWeights(1);
covWeights(1) = w0c*covWeights(1);

meanWeights(2:end) = wim*meanWeights(2:end);
covWeights(2:end) = wic*covWeights(2:end);

S = chol(P);
% Generate 2n+1 new sigma points
SP = zeros(n,2*n+1); %Preallocate vectors of sigma points
% Define our mean SP
SP(:,1) = x;

 for j =1:n 
     SP(:,j+1) = x + sqrt(n+Lambda)*S(j,:)';
     SP(:,n+j+1) = x - sqrt(n+Lambda)*S(j,:)';
 end

 ChiBarX = zeros(height(x),2*n+1);
 ChiBarX = zeros(height(x),2*n+1);
 gamma = zeros(2,2*n+1);
 y_minus = [0;0]; % Predicted Measurment

 Pxx = zeros(height(x),height(x));
 Pyy = zeros(height(y_minus),height(y_minus));
 Pxy = zeros(height(x),height(y_minus));

  for j = 1:2*n+1
         
     % Push both process noise and state through dynamics mode
     %Chi(3,j) = wrapToPi(Chi(3,j));
     ChiBarX(:,j) = f(SP(:,j)) +  mvnrnd([0;0;0;0;0],Q)';
     
     ChiBarX(3,j) = wrapToPi(ChiBarX(3,j));

     % Measurments
     gamma(:,j) = [ChiBarX(1,j); ChiBarX(2,j)] + mvnrnd([0;0],R)';


  end

   % Sum to get predicted state/measurment
   x_minus = sum(meanWeights.*ChiBarX,2);
   y_minus = sum(meanWeights.*gamma,2);

    for j = 1:2*n+1

    Pxx = covWeights(j)*(ChiBarX(:,j)-x_minus)*(ChiBarX(:,j)-x_minus)' + Pxx;
    Pyy = covWeights(j)*(gamma(:,j)-y_minus)*(gamma(:,j)-y_minus)' + Pyy; %Measurment covariance
    Pxy = covWeights(j)*(ChiBarX(:,j)-x_minus)*(gamma(:,j)-y_minus)' + Pxy; %Cross covariance

    end

    Pyy = Pyy + R;
    Pxx = Pxx + Q;

end

