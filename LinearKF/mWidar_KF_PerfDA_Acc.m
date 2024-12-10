function [xhat_plus,P,inov,inovCov] = mWidar_KF_PerfDA_Acc(x0,y,P0,F,Q,R,step)
%% Description

%%This function uses the simulated mWidar data, and performs a KF to
%%estimate the track for a single object. 

%%INPUTS: x0 ~ Initial conditions for our state 
%%y ~ Measurments, for this function we assume these are all associated
% with the object we are trying to track
% P0 ~ Initial state covariance matrix

%%OUTPUTS: xhat_plus ~ KF estimate of our state; P ~ State covariance
%%matrix; inov ~ inovation errors, mainly for debugging, these should be
%%converging on 0; inovCov ~ inovation covariances

%mvnpdf ~ gaussian pdf

%% KF Matrices

xhat_plus = cell(1,step);
xhat_plus{1} = x0;

P = cell(1,step);
P{1} = P0;
inovCov = zeros(2,2,step-1);
inov = zeros(2,step-1);
y_filter = zeros(2,step-1);

H = [1 0 0 0 0 0;0 0 0 1 0 0];

R_TRUE = 0.1*eye(2); %Measurment noise uncertainty
%Q_TRUE = 0.01*Q; %Process noise uncertainty
O = zeros(6,1);

%% Filter
for k=1:length(y)-1
  
    %Non-deterministic
    y_filter(:,k+1) = mvnrnd(y(:,k),R_TRUE)';
    

    %%Time Update
    x_hat_minus = F*xhat_plus{k} + mvnrnd(O,Q)';
    P_minus = F*P{k}*F' + Q;
    inovCov(:,:,k) = H*P_minus*H'+ R;
    Kk = P_minus*H'/(inovCov(:,:,k));
    

    %%Measurment update
    inov(:,k) = (y_filter(:,k+1)) - H*x_hat_minus;
    xhat_plus{k+1} = x_hat_minus + Kk*(inov(:,k));
    P{k+1} = (eye(6) - Kk*H)*P_minus;

end

end

