function [Circ] = GenerateCircleTruth(x0,tMax)

c = physconst('LightSpeed'); %speed of light in m/s
dtsamp = 0.5*c*667e-12; %image frame subsampling step size for each Tx

%% *************** Circle ********************
% A circle has no State Space representation since it is entirely
% nonlinear. Its trajectory is instead defined by sinousoidal functions

%Initialize
tspan = 0:dtsamp:dtsamp*tMax;

Circ = zeros(6,length(tspan));


% Simulate dynamics
for k=1:length(tspan)

    traj = [1.5.*cos(tspan(k)) + x0(1) - 1.5;
            -1.5.*sin(tspan(k)) + x0(2);
            -1.5.*cos(tspan(k)) + x0(3) + 1.5;
             1.5.*sin(tspan(k))+ x0(4);
             1.5.*cos(tspan(k)) + x0(5) - 1.5;
             -1.5.*sin(tspan(k)) + x0(6)];
    Circ(:,k) = traj;
end

end

