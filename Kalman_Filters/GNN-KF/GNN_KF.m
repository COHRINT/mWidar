function [KF] = GNN_KF(KF)
%GNN_KF

for i = 1:size(KF,2)-1
    [KF{i},x_minus,z_hat,P_minus] = Prediction_Step(KF{i});
    [KF{i},meas] = Validation(KF{i},z_hat);
    if meas
        [KF{i},z] = GNN(KF{i},z_hat);
    end
    [X, P, innov] = Measurment_Step(KF{i},z,P_minus,x_minus,z_hat);

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

function [KF,z] = GNN(KF,z_hat)
    S = KF.S;
    z = KF.valid_z;
    d2 = zeros(size(z,2),1);
    for i = 1:size(z,2)
        innov = z(:,i) - z_hat;
        d2(i) = mahalanobis(innov,S);
    end

    [~,idx] = min(d2);
    z = z(:,idx);
end

function [d2] = mahalanobis(innov,S)
    d2 = sqrt(innov\S*innov);
end

function [X,P,innov] = Measurment_Step(KF,z,P_minus, x_minus,z_hat)
    S = KF.S;
    H = KF.H;


    KK = P_minus*H'/S;
    innov = z - z_hat;
    X = x_minus + KK*innov;
    P = (eye(6) - KK*H)*P_minus;

end