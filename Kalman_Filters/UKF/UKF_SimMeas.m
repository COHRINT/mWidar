function [X,P] = UKF_SimMeas(x0,P0,y,F,Q,R)
%{
%%%%%% UKF %%%%%%%%%%
 Unscented Kalman Filter
   Inputs:
       x0: Initial State
       P0: Initial State Covariance
       y: Measurments
       Q: State Noise Model
       R: Measurment Noise Model
       tMax: Simulation time

   Outputs:
       X: 6x1 state estimation
       P: 6x6 state covaraince
%}

M = load("sampling.mat").M;
G = load("recovery.mat").G;

c = physconst('LightSpeed'); %speed of light in m/s
dtsamp = 0.5*c*667e-12; %image frame subsampling step size for each Tx

R_TRUE = [0.1 0;0 0.1];

X = zeros(6,length(y));
X(:,1) = x0;

P = cell(1,length(y));
P{1} = P0;

% # of states
n = 6;
alpha = 1e-5; % Large a -> SP more spread out
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


for k = 2:length(y)
    
    % Sample measurment/process noise

    q = mvnrnd([0;0;0;0;0;0],Q)';
    r = mvnrnd([0;0],R_TRUE)';

    %% Prediction Step from k -> k+1
    
%     X_aug = [X(:,k-1);q;r];
%     n = length(X_aug);
% 
%     P_aug = zeros(n);
% 
%     P_aug(1:6,1:6) = P{k-1};
%     P_aug(7:12,7:12) = Q;
%     P_aug(13:14,13:14) = R_TRUE; % Or use R_True?
    
    S = chol(P{k-1});

    % Generate 2n+1 new sigma points
    SP = zeros(n,2*n+1); %Preallocate vectors of sigma points

    SP(:,1) = X(:,k-1);
    
    % Not sure if this is the best approach with the SP, getting them first
    % to get a better noise estimate
    for j =1:n 
        SP(:,j+1) =  X(:,k-1) + sqrt(n+Lambda)*S(j,:)';
        SP(:,n+j+1) =  X(:,k-1) - sqrt(n+Lambda)*S(j,:)';
    end
    
    % Simulate Each Sigma Point a Unique mWidar Image
    gamma = zeros(2,2*n+1);

    for j = 1:2*n+1
        [gamma(:,j),~] = SimulateImages(SP(:,j),M,G);
    end

    y_minus = [0;0]; % Predicted Measurment

     for j = 1:2*n+1
         y_minus = y_minus + meanWeights(j)*gamma(:,j);
     end
    
%      y_minus(1) = mean(gamma(1,:));
%    y_minus(2) = mean(gamma(2,:));
   
    x_minus = F*X(:,k-1) + q;
    P_minus = F*P{k-1}*F' + Q;
    
    %% Measurment Step
    
    y(:,k) = y(:,k) + r;
    

    Pyy = zeros(2,2);
    Pxy = zeros(6,2);

    for i=1:2*n+1
    Pyy = covWeights(i)*(gamma(:,i)-y_minus)*(gamma(:,i)-y_minus)' + Pyy; %Measurment covariance
    Pxy = covWeights(i)*(SP(1:6,i)-x_minus)*(gamma(:,i)-y_minus)' + Pxy; %Cross covariance
    end

    Pyy = Pyy + R;

    % Kalman gain
    Kk = Pxy/Pyy;

    %Perform Kalman state and Covariance update w/ observation

    X(:,k) = x_minus + Kk*(y(:,k) - y_minus);
    P{k} = P_minus - Kk*Pyy*Kk';


end

end
