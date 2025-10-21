clear; clc; close all

Q = 0.001*diag([0.01 0.01 0.1 0.01 0.01 0.1]);
%% Generate a new constant acceleration and constant velocity track for KF

% x = [x,vx,ax, y, vy, ay]
% Declare constants
dt = 0.1; % [s]
t_vec = 0:dt:20;

A = [0 1 0 0 0 0;
    0 0 1 0 0 0;
    0 0 0 0 0 0;
    0 0 0 0 1 0;
    0 0 0 0 0 1;
    0 0 0 0 0 0];
%% Track 1: Constant velocity track

% Initial state, x0 -> starts from rest
xplus = zeros(6,length(t_vec));
xplus(:,1) = [0;0.5;0;0;0.5;0];

% Generate GT a constant velocity trajectory
for k = 2:length(t_vec)
    xplus(:,k) = expm(A*dt)*xplus(:,k-1) + mvnrnd([0;0;0;0;0;0],Q)';
end

% Plot Track
figure(); hold on; grid on
title('Constant Velocty - Ground Truth')
tiledlayout(6,1)
nexttile
plot(t_vec,xplus(1,:),'k',LineWidth=1)
nexttile
plot(t_vec,xplus(2,:),'k',LineWidth=1)
nexttile
plot(t_vec,xplus(3,:),'k',LineWidth=1)
nexttile
plot(t_vec,xplus(4,:),'k',LineWidth=1)
nexttile
plot(t_vec,xplus(5,:),'k',LineWidth=1)
nexttile
plot(t_vec,xplus(6,:),'k',LineWidth=1)

figure(); hold on; grid on
plot(xplus(1,:),xplus(4,:),'k',LineWidth=1)
save linear_const_v.mat xplus -mat

%% Track 2: Constant Acceleration

% Initial state, x0 -> starts from rest
xplus = zeros(6,length(t_vec));
xplus(:,1) = [0;0;0.075;0;0;0.065];

% Generate GT a constant velocity trajectory
for k = 2:length(t_vec)
    xplus(:,k) = expm(A*dt)*xplus(:,k-1)+ mvnrnd([0;0;0;0;0;0],Q)';
end

% Plot Track
figure(); hold on; grid on
title('Constant Acceleration - Ground Truth')
tiledlayout(6,1)
nexttile
plot(t_vec,xplus(1,:),'k',LineWidth=1)
nexttile
plot(t_vec,xplus(2,:),'k',LineWidth=1)
nexttile
plot(t_vec,xplus(3,:),'k',LineWidth=1)
nexttile
plot(t_vec,xplus(4,:),'k',LineWidth=1)
nexttile
plot(t_vec,xplus(5,:),'k',LineWidth=1)
nexttile
plot(t_vec,xplus(6,:),'k',LineWidth=1)

figure(); hold on; grid on
plot(xplus(1,:),xplus(4,:),'k',LineWidth=1)

save linear_const_a.mat xplus -mat

%% Track 3: Non-Linear Dynamics Model


% Initial state, x0 -> starts from rest
xplus = zeros(6,length(t_vec));
xplus(:,1) = [0;0.5;0;0;0.5;0];

% Generate GT a constant velocity trajectory
for k = 2:length(t_vec)
    xdot = NL_Dyn(t_vec(k-1),xplus(:,k-1));

    xplus(:,k) = xplus(:,k-1) + xdot*dt + mvnrnd([0;0;0;0;0;0],Q)';
end

% Plot Track
figure(); hold on; grid on
title('Non-Linear Dynamics Model - Ground Truth')
tiledlayout(6,1)
nexttile
plot(t_vec,xplus(1,:),'k',LineWidth=1)
nexttile
plot(t_vec,xplus(2,:),'k',LineWidth=1)
nexttile
plot(t_vec,xplus(3,:),'k',LineWidth=1)
nexttile
plot(t_vec,xplus(4,:),'k',LineWidth=1)
nexttile
plot(t_vec,xplus(5,:),'k',LineWidth=1)
nexttile
plot(t_vec,xplus(6,:),'k',LineWidth=1)

figure(); hold on; grid on
plot(xplus(1,:),xplus(4,:),'k',LineWidth=1)

save NL_circle.mat xplus -mat

%% Track 4: mWidar Sim Trajectory

% Initial state, x0 -> starts from rest
xplus = zeros(6,length(t_vec));
xplus(:,1) = [0;0.05;0;1;0.05;0];

% Generate GT a constant velocity trajectory
for k = 2:length(t_vec)
    xplus(:,k) = expm(A*dt)*xplus(:,k-1) + mvnrnd([0;0;0;0;0;0],Q)';
end

% Plot Track
figure(); hold on; grid on
title('Constant Velocty (mWidar sim) - Ground Truth')
tiledlayout(6,1)
nexttile
plot(t_vec,xplus(1,:),'k',LineWidth=1)
nexttile
plot(t_vec,xplus(2,:),'k',LineWidth=1)
nexttile
plot(t_vec,xplus(3,:),'k',LineWidth=1)
nexttile
plot(t_vec,xplus(4,:),'k',LineWidth=1)
nexttile
plot(t_vec,xplus(5,:),'k',LineWidth=1)
nexttile
plot(t_vec,xplus(6,:),'k',LineWidth=1)

figure(); hold on; grid on
plot(xplus(1,:),xplus(4,:),'k',LineWidth=1)

save linear_const_v_mWidarsim.mat xplus -mat

function xdot = NL_Dyn(t,x)

vx = x(2);
vy = x(5);
ax = x(3);
ay = x(6);

px_dot = vx*cos(pi/2*t);
py_dot = vy*sin(pi/2*t);
vx_dot = ax*cos(pi/2*t);
vy_dot = ay*sin(pi/2*t);
ax_dot = 0;
ay_dot = 0;

xdot = [px_dot;vx_dot;ax_dot;py_dot;vy_dot;ay_dot];

end