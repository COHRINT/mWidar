function [x_minus,y_minus,Pxx,Pyy,Pxy] = UKF_DynamicsStepLinear(x,P,f,Q,R)

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

n = 14; %# of states

alpha = 1e-2; % Large a -> SP more spread out
kappa = 0; %Typical Values
beta = 2; %Typical Values

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

% Augmented State Vector
X_aug = [x;[0;0;0;0;0;0];[0;0]];

% Augmented Covariance Matrix
P_aug = blkdiag(P,Q,R);
S = chol(P_aug);

% Generate 2n+1 new sigma points
SP = zeros(n,2*n+1); %Preallocate vectors of sigma points

% Mean SP
SP(:,1) = X_aug;

 % Generate 29 SP
 for j =1:n 
     SP(:,j+1) = X_aug + sqrt(n+Lambda)*S(j,:)';
     SP(:,n+j+1) = X_aug - sqrt(n+Lambda)*S(j,:)';
 end

 Chi = SP(1:6,:); % State SP
 Chiq = SP(7:12,:); % Process Noise SP
 Chir = SP(13:14,:); % Measurment Noise SP
    
 gamma = zeros(2,2*n+1);
 ChiBarX = zeros(6,2*n+1);

 y_minus = [0;0]; % Predicted Measurment

 x_minus = [0;0;0;0;0;0];

 Pxx = zeros(6,6);
 Pyy = zeros(2,2);
 Pxy = zeros(6,2);

 for j = 1:2*n+1
         
     % Push both process noise and state through dynamics mode
     ChiBarX(:,j) = f(Chi(:,j)) + Chiq(:,j);

     % Measurments
     gamma(:,j) = [ChiBarX(1,j); ChiBarX(4,j)] + Chir(:,j);

     % Sum to get predicted state/measurment
     x_minus = x_minus + meanWeights(j)*ChiBarX(:,j);
     y_minus = y_minus + meanWeights(j)*gamma(:,j);

 end
 y_minus
 for j = 1:2*n+1

    Pxx = covWeights(j)*(ChiBarX(:,j)-x_minus)*(ChiBarX(:,j)-x_minus)' + Pxx;
    Pyy = covWeights(j)*(gamma(:,j)-y_minus)*(gamma(:,j)-y_minus)' + Pyy; %Measurment covariance
    Pxy = covWeights(j)*(ChiBarX(:,j)-x_minus)*(gamma(:,j)-y_minus)' + Pxy; %Cross covariance

 end

 Pyy = Pyy + R;
 Pxx = Pxx + Q;

end

