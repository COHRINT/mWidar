function [X,P] = UKF_MeasurmentStep(x_minus,y_minus,Pxx,Pyy,Pxy,y)

%{
%%%%%% UKF %%%%%%%%%%
Measurment Correction Step for a UKF

Inputs:
    x_minus: Prior dynamics state estimate
    y_minus: Predicted Measurment
    Pxx: State Covaraince
    Pyy: Innovation Covaraince
    Pxy: Cross Covaraince
    y: Noisy Measurments

   Outputs:
       X: 6x1 state estimation
       P: 6x6 state covaraince
%}

% Kalman gain
Kk = Pxy/Pyy;

%Perform Kalman state and Covariance update w/ observation
X = x_minus + Kk*(y - y_minus);
P = Pxx - Kk*Pyy*Kk';

end

