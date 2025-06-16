function [KF] = mWidar_PDAF(KF)
% Standard PDAF for mWidar Applications

for i = 1:size(KF,2)-1
    % Filter
    i
    beta = 0;
    beta0 = 0;

    [KF{i},x_minus, z_hat, P_minus] = Prediction_Step(KF{i});

    [KF{i},meas] = Validation(KF{i},z_hat);
    
    if meas
        [KF{i},beta,beta0] = data_Association(KF{i},z_hat);
    end
    [X,P,innov] = state_Estimation(KF{i},beta,beta0, P_minus, x_minus, z_hat, meas);

    KF{i+1}.x = X;
    KF{i+1}.P = P;
    KF{i+1}.innov = innov;

end

end

function [KF,x_minus, z_hat, P_minus] = Prediction_Step(KF)
    % Standard KF prediction step
    F = KF.F;
    H = KF.H;
    Q = KF.Q;
    R = KF.R;
    x = KF.x;
    P = KF.P;

    x_minus = F*x; % state prediction
    z_hat = H*x_minus; % measurment prediction
    P_minus = F*P*F' + Q; % state covariance prediction
    KF.S = H*P_minus*H' + R; % measurment innovation covariance

end

function [KF,meas] = Validation(KF,z_hat)

    % Pass measurment through validation ellipse
    S = KF.S;
    meas = true;
    gamma = chi2inv(0.05/2,2);
    %gamma = 0.5;
    for j = 1:size(KF.z,2)
        z = KF.z(:,j);
        Nu = (z-z_hat)'/S*(z-z_hat);
        if Nu < gamma % Validation gate
            KF.valid_z = [KF.valid_z z]; % Append new measurment onto validated list
        end
    end

    if isempty(KF.valid_z)
        fprintf('No Valid Measurments \n')
        meas = false;
    end


end

function [KF,beta,beta0] = data_Association(KF,z_hat)

% Association step

% Tuning parameters
lambda = 2.5;
PD = 0.95;
PG = 0.95;

% Pre allocate space
likelihood = zeros(1,size(KF.valid_z,2));
beta = zeros(1,size(KF.valid_z,2));

% Compute likelihood of each validated measurment
for j = 1:size(KF.valid_z,2)
    likelihood(j) = (mvnpdf(KF.valid_z(:,j),z_hat,KF.S)*PD)/lambda;
end

sum_likelihood = sum(likelihood,2);

for j = 1:size(KF.valid_z,2)
    beta(j) = likelihood(j)/(1-PD*PG+sum_likelihood); % Compute beta values
end

beta0 = (1-PD*PG)/(1-PD*PG+sum_likelihood); % beta0 -> probability of no detections being true

end

function [X,P,innov] = state_Estimation(KF,beta,beta0, P_minus, x_minus, z_hat,meas)

H = KF.H;
S = KF.S;
R = KF.R;
z = KF.valid_z;

KK = 0;
innov = 0;
Pc = 0;
P_tilde = 0;

if meas == false
    z = z_hat;
    innov = zeros(size(z,1),1);

    %For Probability Data Association
    beta0 =1; %switching value
            
    %Calc Kalman gain
    KK =(P_minus*H')/(H*P_minus*H'+R);
    Pc=P_minus-KK*S*KK';
    P_tilde=0;

else
    nu = zeros(2,size(z,2));
    innov = [0;0];
    for j = 1:size(z,2)
        nu(:,j) = z(:,j) - z_hat; 
        innov = innov + beta(j)*nu(:,j); % Combined innovations -> weighted sum of measurments
    end

    % Standard kalman state update
    KK = P_minus*H'*S;

    Pc = P_minus - KK*S*KK';

    temp = 0;
    for j = 1:size(z,2)
        temp = temp + (beta(j)*(nu(j)*nu(j)'));
    end
    P_tilde =  KK*(temp-innov*innov')*KK';

    % Updated state/Covariance
end

X = x_minus + KK*innov;
P = beta0*P_minus + (1-beta0)*Pc + P_tilde;
end

