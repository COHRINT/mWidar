function [x_minus,y_minus,P_minus,S] = KF_DS(KF)
    %{
    Performs a dynamics update on state
    Inputs:
    Outputs:
    %}
    
    F = KF.F;
    Q  =KF.Q;
    R = KF.R;
    H = KF.H;
    P = KF.P;
    x = KF.x;

    x_minus = F*x; %+ mvnrnd([0;0;0;0;0;0],Q)';
    P_minus = F*P*F' + Q;
    S = H*P_minus*H' + R;
    y_minus = H*x_minus;

    
end

