clear; close all; clc

%initialize

%# of MC sims
Simulations = 100;

P0 = diag([0.5 0.75 1 0.5 0.75 1]);


%% Initialize KF Matrices

c = physconst('LightSpeed'); %speed of light in m/s
dtsamp = 0.5*c*667e-12; %image frame subsampling step size for each Tx

step = 100; % # of tims steps

alpha = 0.05; % Significance level, if NEES test passes, we declare KF consistent with 5% significance level

r1 = chi2inv(alpha/2,Simulations*6)/Simulations; %Chi^2 bounds
r2 = chi2inv(1-alpha/2,Simulations*6)/Simulations;

% KF Matrices
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
    0 1.5 0 0;
    0 0 0.5 0;
    0 0 0 1.5];

Gamma = [0 0 0 0;1 0 0 0;0 1 0 0;0 0 0 0;0 0 1 0;0 0 0 1]; 
Z = chol(W); %Cholskey decomp will fail if not pos def
R = 1.75*eye(2); 
Z = dtsamp* [-A Gamma*W*Gamma'; O A'];
eZ = expm(Z);
F = eZ(7:12,7:12)';
Q = F * eZ(1:6,7:12);

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

% Generate new initial condition each MC run
x0 = abs(mvnrnd([0;0;0;0;0;0],P0)');
true = GenerateVTruth(x0);
xhat_0 = true(:,1);
y = [true(1,:);true(4,:)];


[xhat_plus(i,:),P(i,:),inov,inovCov] = mWidar_KF_PerfDA_Acc(xhat_0,y,P0,F,Q,R,step);

    for k=1:step
    % Error
    err(:,k) = xhat_plus{i,k} - true(:,k);

    % NEES Statistic
    epsilon(i,k) = (err(:,k)'/P{i,k})*err(:,k);
   
    end

end

% Average of epsilon over each MC run
epMC = mean(epsilon);

% # of epsilon within our bounds
inRange = 0;
for i=1:step
    if(r1<epMC(i)) && (r2>epMC(i))
        inRange = inRange+1;
    end
end
percent = inRange/step * 100;

str1 = [num2str(percent),' % in bounds'];

% Plot
figure(1);hold on; grid on;
plot(epMC,'o')
yline(r1,'r--',LineWidth=1)
yline(r2,'r--',LineWidth=1)

xlabel('Time Step')
ylabel('NEES Statistic')
title('Chi-Squared NEES Test')
legend(str1,Location="southeast")


