function [X,P] = UKF_general(x0,P0,y,F,Q,R)
%{
%%%%%% UKF %%%%%%%%%%
 Unscented Kalman Filter
   Inputs:
       x0: Initial State
       P0: Initial State Covariance
       y: Measurments
       Q: State Noise Model
       R: Measurment Noise Model
       f: Non-Linear Dynamics Model
       h: Non-Linear Measurment Model
       tMax: Simulation time

   Outputs:
       X: 6x1 state estimation
       P: 6x6 state covaraince
%}

c = physconst('LightSpeed'); %speed of light in m/s
dtsamp = 0.5*c*667e-12; %image frame subsampling step size for each Tx

X = zeros(6,length(y));
X(:,1) = x0;

P = cell(1,length(y));
P{1} = P0;

n = 6; %# of states

% Hyperparameters
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

for k = 2:length(y)

%% Prediction Step from k -> k+1

x_minus = F*X(:,k-1);
Pminus = F*P{k-1}*F' + Q;

%% Measurment step from k -> k+1

% Generate 2n+1 new sigma points
SP = zeros(n,2*n+1); %Preallocate vectors of sigma points
gamma = zeros(2,2*n+1); %Preallocate vectors of transformed sigma points

Sbar = chol(Pminus); %Matrix sq root

SP(:,1) = x_minus;

for j =1:n
        SP(:,j+1) = x_minus + sqrt(n+Lambda)*Sbar(j,:)';
        SP(:,n+j+1) = x_minus - sqrt(n+Lambda)*Sbar(j,:)';
end

% Propogate new sigma points through non-linear measurment function to
% produce predicted measurments, gamma

for i = 1:2*n+1
        gamma(:,i) = [1 0 0 0 0 0;
                      0 0 0 1 0 0]*SP(:,i); %Only pass pos SP
end

y_minus = zeros(2,1);
Pyy = zeros(2,2);
Pxy = zeros(6,2);

% Get predicted measurment mean and covariance
for i=1:2*n+1
    y_minus = meanWeights(i)*gamma(:,i) + y_minus;
end

for i=1:2*n+1
    Pyy = covWeights(i)*(gamma(:,i)-y_minus)*(gamma(:,i)-y_minus)' + Pyy; %Measurment covariance
    Pxy = covWeights(i)*(SP(:,i)-x_minus)*(gamma(:,i)-y_minus)' + Pxy; %Cross covariance
end

Pyy = Pyy + R;

% Kalman gain
Kk = Pxy/Pyy;

%Perform Kalman state and Covariance update w/ observation

X(:,k) = x_minus + Kk*(y(:,k) - y_minus);
P{k} = Pminus - Kk*Pyy*Kk';

end

end

