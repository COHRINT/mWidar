function [X,P,inov] = KF_MS(KF,x_minus,y_minus,P_minus)
S = KF.S;
H = KF.H;
y = KF.y;

KK = P_minus*H'/S;
inov = y - y_minus;
X = x_minus + KK*inov;
P = (eye(6) - KK*H)*P_minus;



end

