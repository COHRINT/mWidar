function [Line] = GenerateLineTruth(x0)

Line = zeros(6,41);

c = physconst('LightSpeed'); %speed of light in m/s
dtsamp = 0.5*c*667e-12; %image frame subsampling step size for each Tx

%Line GT Dynamics Model
A_line = [0 1 0 0 0 0;
          0 0 1 0 0 0 ;
          0 0 0 0 0 0;
          0 0 0 0 1 0;
          0 0 0 0 0 1;
          0 0 0 0 0 0];

tspan = 0:dtsamp:4;

% Simulate
for k = 1:41
Line(:,k) = expm(A_line.*tspan(k))*x0;
end

end

