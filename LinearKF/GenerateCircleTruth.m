function [Circ] = GenerateCircleTruth(x0)

c = physconst('LightSpeed'); %speed of light in m/s
dtsamp = 0.5*c*667e-12; %image frame subsampling step size for each Tx

%% *************** Circle ********************
% A circle has no State Space representation since it is entirely
% nonlinear. Its trajectory is instead defined by sinousoidal functions

%Initialize
tspan = linspace(0,10,100);

Circ = zeros(6,100);


% Simulate dynamics
for k=1:100

    traj = [1.5.*cos(tspan(k)) + x0(1) - 1.5;
            -1.5.*sin(tspan(k));
            -1.5.*cos(tspan(k));
             1.5.*sin(tspan(k))+ x0(4);
             1.5.*cos(tspan(k));
             -1.5.*sin(tspan(k))];
    Circ(:,k) = traj;
end

end

