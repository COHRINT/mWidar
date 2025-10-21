function [xplus] = DubinsModel(dt,x)
%%%%%%%%%%%%%%% Dubins Dynamic Model %%%%%%%%%%%%%%%%%%%%
%{
x - state vector: [zeta; eta; theta; v; omega]

zeta - x pos
eta - y pos
theta - bearing
v - velocity (magnitude of vel vector)
omega - angular vel

dt - deltaT time step

%}

theta = x(3);
v = x(4);
omega = x(5);

zeta_dot = v*cos(theta);
eta_dot = v*sin(theta);
theta_dot = omega;

vdot = 0;
omegadot = 0;

xdot = [zeta_dot;eta_dot;theta_dot;vdot;omegadot];

xplus = x + xdot*dt;


end



