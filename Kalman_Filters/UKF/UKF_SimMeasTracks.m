%% UKF_SimMeasTracks
%{

Produce a specialized track for the UKF image simulator that doesn't
approach the image generation boundry

%}

clear; clc; close all

% Smaller circle

tspan = linspace(0,7,70);

Circ = zeros(6,70);
for k=1:70

    traj = [0.25.*cos(tspan(k));
            -0.25.*sin(tspan(k));
            -0.25.*cos(tspan(k));
             0.25.*sin(tspan(k))+2;
             0.25.*cos(tspan(k));
             -0.25.*sin(tspan(k))];
    Circ(:,k) = traj;
end

figure()
plot(Circ(1,:),Circ(4,:))

figure()
tiledlayout(3,2)
nexttile
plot(tspan,Circ(1,:))
nexttile
plot(tspan,Circ(2,:))
nexttile
plot(tspan,Circ(3,:))
nexttile
plot(tspan,Circ(4,:))
nexttile
plot(tspan,Circ(5,:))
nexttile
plot(tspan,Circ(6,:))

save SimulatorCircleGT.mat Circ -mat

% Static

for k=1:50

    traj = [0; 0; 0; 2; 0; 0;];
    stat(:,k) = traj;
end

save Static.mat stat -mat

