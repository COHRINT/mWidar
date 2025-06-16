clear; clc; close all

load DubinsCircle.mat xplus

dt = 0.1;

n = 6; % Number of timesteps we want to estimate forward;

GT = xplus(:,1:n);
%% Generate 5 timestep GT

% Linear Measurment Matrix
H = [1 0 0 0 0;0 1 0 0 0];

t = 0:dt:n*dt;

v = 0.5*ones(1,length(t));
omega = 0.05*ones(1,length(t));

x = GT(:,1);

% O will define our observibility matrix O = [ H; H*F; H*F*F; ...] 
O = H; 
% O will have as many matrix blocks as time steps we define

% Set initial F_tilde_prev to be the identity, F_tilde_prev will be the 
% Matrix multiplication of all the previous F_tilde matrices;
F_tilde_prev = eye(5);

% For each timestep k, push previous timestep through DT STM 
for k = 2:length(t)
    
    theta = x(3);
    v = x(4);

    % x = [x; y; theta; v; theta_dot]
    % F is the Jacobian of our DT Dubins model evaluated at nominal values
    F = [0 0 -v*sin(theta) cos(theta) 0;
         0 0 v*cos(theta) sin(theta) 0;
         0 0 0 0 1;
         0 0 0 0 0
         0 0 0 0 0];

    F_tilde = (eye(5) + F*dt) * F_tilde_prev;

    O = [O; H*F_tilde];

    x = F_tilde*x; % Propogate x through our linearized dynamics model for new nominal state.
    
    F_tilde_prev = F_tilde;

end

rank(O)
