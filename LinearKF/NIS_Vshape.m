clear; close all; clc;

%% Load Relevant Data

load SimTraj_Vshape.mat -mat Pos
load TrueVelVshape.mat -mat trueVel 
load TrueAccVshape.mat -mat TrueAcc

Simulations = 100;

xPos = Pos(1,:);
yPos = Pos(2,:);

true = [xPos;yPos];

%% Initialize KF Matrices

c = physconst('LightSpeed'); %speed of light in m/s
dtsamp = 0.5*c*667e-12; %image frame subsampling step size for each Tx

% Initialization
xhat_0(1,1) = Pos(1,1)+0.1*randn(1); 
xhat_0(4,1) = Pos(2,1)+0.1*randn(1);

xhat_0(2,1) = 0.1*randn(1);
xhat_0(3,1) = 0.1*randn(1); 

xhat_0(5,1) = 0.1*randn(1); 
xhat_0(6,1) = 0.1*randn(1);

y = Pos(:,:); 
step = length(y);

P0 = diag([1 2 3 1 2 3]);

alpha = 0.05;

r1 = chi2inv(alpha/2,Simulations*6)/Simulations;
r2 = chi2inv(1-alpha/2,Simulations*6)/Simulations;

% KF Matrices
Gamma = [0 0;0 0;1 0;0 0;0 0;0 1];
H = [1 0 0 0 0 0;0 0 0 1 0 0];

O = zeros(6); %6x6 matrix of 0's

A = [0 1 0 0 0 0;
    0 0 1 0 0 0;
    0 0 0 0 0 0;
    0 0 0 0 1 0;
    0 0 0 0 0 1;
    0 0 0 0 0 0]; % x = [xpos xvel xacc ypos yvel yacc]

%Van Loans Method

W = [0.45 0.05;0.05 0.45]; 

Z = chol(W); %Cholskey decomp will fail if not pos def
R = 1.7*eye(2); 
Z = dtsamp* [-A Gamma*W*Gamma'; O A'];
eZ = expm(Z);
F = eZ(7:12,7:12)';
Q = F * eZ(1:6,7:12); %Signal Process Noise

% pre allocate space for NIS Statistic values
epsilon = zeros(Simulations,step-1);
Mean_epsilon = zeros(1,step-1);
xhat_plus = cell(Simulations,step);
P = cell(Simulations,step);

% MC simulation of innovation results
for N=1:Simulations
[xhat_plus(N,:),P(N,:),inov,inovCov] = mWidar_KF_PerfDA_Acc(xhat_0,y,P0,F,Q,R,step);

    for k=1:step-2
        % Get y_hat_minus, or expected measurment at timestep k+1
        y_hat_minus = H*F*xhat_plus{N,k};
        %error
        e = true(:,k+1) - y_hat_minus;
        % NIS Statistic
        epsilon(N,k) = e'/inovCov(:,:,k+1)*e;
    end
end

% Average epsilon over all MC runs
for k=1:step-1
    Mean_epsilon(k) = mean(epsilon(:,k));
end

% Chi-squre upper and lower bounds, r1 and r2 for 5% significance

alpha = 0.05;

% N * p DOF
r1 = chi2inv(alpha/2,2*Simulations)./(Simulations); 
r2 = chi2inv(1-alpha/2,2*Simulations)./(Simulations);

% Plot results

figure(); hold on; grid on;

plot(Mean_epsilon,'mo')
yline(r2,'k--',LineWidth=1)
yline(r1,'k--',LineWidth=1)

title('Chi-Squared NIS Test for consistency')
ylabel('NIS Statistic')
xlabel('Time Step k')