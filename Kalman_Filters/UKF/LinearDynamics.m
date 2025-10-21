function [x_hat] = LinearDynamics(x,dt)
%{
Inputs: 
x -> Prior state at time k
dt -> delta t between each sample

Outputs:
x_hat -> Dynamics State estimate at time k+1
%}

% Approx kinematics model
A = [0 1 0 0 0 0;
    0 0 1 0 0 0;
    0 0 0 0 0 0;
    0 0 0 0 1 0;
    0 0 0 0 0 1;
    0 0 0 0 0 0]; % x = [xpos xvel xacc ypos yvel yacc]

F = expm(A*dt);

x_hat = F*x;


end

