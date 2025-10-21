clear; clc; close all


dt = 0.1;
t = 0:dt:20;

%% Set v and omega

rng(100)

v = zeros(1,length(t));
omega = zeros(1,length(t));

for k = 1:length(t)

    v(k) = 2*sin(pi*t(k)*0.1) + normrnd(0,0.01);
    omega(k) = 2*cos(2*pi*t(k)*0.1) + normrnd(0,0.01);

end

%% Get x,y & theta

xplus = generateTrack(t,v, omega);

figure();hold on; grid on
plot(xplus(1,:),xplus(2,:),'bo')
title("Track 1")

xplus = [xplus;v;omega];

save DubinsTurn.mat xplus -mat

%% Set v and omega

v = zeros(1,length(t));
omega = zeros(1,length(t));

for k = 1:length(t)

    v(k) = 0.05*t(k) + normrnd(0,0.01);
    
    if k >= 100 && k <= 150
        omega(k) = 0.01*t(k-99) + normrnd(0,0.001);
    elseif k >= 150
        omega(k) = omega(150)-0.01*t(k-149) + normrnd(0,0.001);
    end

end

%% Get x,y & theta

xplus = generateTrack(t,v, omega);

figure();hold on; grid on
plot(xplus(1,:),xplus(2,:),'bo')
title("Dubin Turn")

xplus = [xplus;v;omega];

save DubinsTrack.mat xplus -mat

%% Spiral

v = linspace(0,2,length(t));
omega = linspace(0,2*pi,length(t));

xplus = generateTrack(t,v, omega);

figure(); hold on; grid on
plot(xplus(1,:),xplus(2,:),'bo')
title('Dubin Spirals')

figure();hold on; grid on
tiledlayout(3,1)
nexttile
plot(t,xplus(1,:),'k',LineWidth=1)
nexttile
plot(t,xplus(2,:),'k',LineWidth=1)
nexttile
plot(t,xplus(3,:),'k',LineWidth=1)
title('Dubin Spirals')


%xplus = [xplus;v;omega];

%save DubinsBearing.mat xplus -mat

%% Constant V

v = 1*ones(1,length(t));
omega = linspace(0,0.25,length(t));

xplus = generateTrack(t,v, omega);

figure(); hold on; grid on
plot(xplus(1,:),xplus(2,:),'bo')
title('Constant V')

figure();hold on; grid on
tiledlayout(3,1)
nexttile
plot(t,xplus(1,:),'k',LineWidth=1)
nexttile
plot(t,xplus(2,:),'k',LineWidth=1)
nexttile
plot(t,xplus(3,:),'k',LineWidth=1)
title('Constant V')

xplus = [xplus;v;omega];

save DubinsConstV.mat xplus -mat

%% Constant Omega and V

v = 0.5*ones(1,length(t));
omega = 0.5*ones(1,length(t));

xplus = generateTrack(t,v, omega);

figure(); hold on; grid on
plot(xplus(1,:),xplus(2,:),'bo')
title('Circle')

figure();hold on; grid on
tiledlayout(3,1)
nexttile
plot(t,xplus(1,:),'k',LineWidth=1)
nexttile
plot(t,xplus(2,:),'k',LineWidth=1)
nexttile
plot(t,xplus(3,:),'k',LineWidth=1)
title('Circle')

xplus = [xplus;v;omega];

save DubinsCircle.mat xplus -mat


function [xplus] = generateTrack(t,v, omega)

    zeta = zeros(1,length(t));
    eta = zeros(1,length(t));
    theta = zeros(1,length(t));
    xplus = zeros(3,length(t));

    theta(1) = 0;

    dt = 0.1;

    for k = 2:length(t)

        zeta_dot = v(k)*cos(theta(k));
        eta_dot = v(k)*sin(theta(k));
        theta_dot = omega(k);

        xdot = [zeta_dot;eta_dot;theta_dot];

        xplus(:,k) = xplus(:,k-1) + xdot*dt;

        xplus(3,k) = wrapToPi(xplus(3,k));
        theta(k+1) = xplus(3,k);

    end
    
end
