clear; close all; clc


%% Load Relevant Data

Simulations = 100; % # of MC runs

P0 = diag([0.75 0.75 1 0.75 0.75 1]);

%% Initialize KF Matrices

c = physconst('LightSpeed'); %speed of light in m/s
dtsamp = 0.5*c*667e-12; %image frame subsampling step size for each Tx

alpha = 0.05;

r1 = chi2inv(alpha/2,Simulations*6)/Simulations;
r2 = chi2inv(1-alpha/2,Simulations*6)/Simulations;

step = 100;

% KF Matrices
Gamma = [0 0 0 0;1 0 0 0;0 1 0 0;0 0 0 0;0 0 1 0;0 0 0 1];
H = [1 0 0 0 0 0;0 0 0 1 0 0];

O = zeros(6); %6x6 matrix of 0's

A = [0 1 0 0 0 0;
    0 0 1 0 0 0;
    0 0 0 0 0 0;
    0 0 0 0 1 0;
    0 0 0 0 0 1;
    0 0 0 0 0 0]; % x = [xpos xvel xacc ypos yvel yacc]

%Van Loans Method
W = [0.5 0 0 0;
    0 2.5 0 0;
    0 0 0.5 0;
    0 0 0 2.5];

 
Z = chol(W); %Cholskey decomp will fail if not pos def
R = 1.75*eye(2); 
Z = dtsamp* [-A Gamma*W*Gamma'; O A'];
eZ = expm(Z);
F = eZ(7:12,7:12)';
Q = F * eZ(1:6,7:12);

%% Q Tuning
% Circle
Q(1,1) = 0.25;
Q(4,4) = 0.25;
Z = chol(Q);
%% Preallocate variables

xhat_plus = cell(Simulations,step);
P = cell(Simulations,step);

err = zeros(6,step);

epsilon = zeros(Simulations,step);

epMC = zeros(1,step);
%% MC Simulation of KF

for i=1:Simulations

x0 = mvnrnd([0;0;0;0;0;0],P0)';

true = GenerateCircleTruth(x0);

xhat_0 = true(:,1);
y = [true(1,:);true(4,:)];

[xhat_plus(i,:),P(i,:),inov,inovCov] = mWidar_KF_PerfDA_Acc(xhat_0,y,P0,F,Q,R,step);

    for k=1:step
    % Error
    err(:,k) = xhat_plus{i,k} - true(:,k);

    % NEES Statistic
    Z = chol(P{i,k});
    epsilon(i,k) = err(:,k)'/P{i,k}*err(:,k);
   
    end

end

for k=1:step
epMC(k) = mean(epsilon(:,k));

end

inRange = 0;
for i=1:step
    if(r1<epMC(i)) && (r2>epMC(i))
        inRange = inRange+1;
    end
end
percent = inRange/step * 100;
str1 = [num2str(percent),'% in bounds'];

fprintf(str1)

figure(1);hold on; grid on;
plot(epMC,'o')
yline(r1,'r--',LineWidth=1)
yline(r2,'r--',LineWidth=1)
xlim([0 step])


xlabel('Time Step')
ylabel('NEES Statistic')
title('Chi-Squared NEES Test')

