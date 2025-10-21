clear; clc; close all

% Test RK integrator with 2-body orbital problem

mu = 398600;
r0 = [7642; 170; 2186];
r_dot0 = [0.32; 6.91; 4.29];

pos0 = norm(r0); %km
v0 = norm(r_dot0); %km/s

init = [r0; r_dot0];

nrg_0 = v0^2/2 - mu/pos0;

a = -mu/(2*nrg_0); %km

P = 2*pi*sqrt(a^3/mu); %sec
h = 0.1;

t = 0:h:P;
xplus = zeros(6,length(t));

xplus(:,1) = init;

f = @(x,t) TwoBodyProblem(x,t);

for k=1:length(t)
    
    xplus(:,k+1) = RKIntegrator(f,xplus(:,k),h,t(k));

end

X = xplus(1,:);
Y = xplus(2,:);
Z = xplus(3,:);

figure(); hold on; grid on
plot3(X,Y,Z)

function [xdot] = TwoBodyProblem(x,t)

r = x(1:3);
rdot = x(4:6);

mu = 398600;
pos = norm(r);

rdoubledot = -(mu/pos^3)*r;

xdot = [rdot;rdoubledot];

end