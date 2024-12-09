clear; close all; clc

% Initialize
Simulations = 100;


%% Initialize KF Matrices

c = physconst('LightSpeed'); %speed of light in m/s
dtsamp = 0.5*c*667e-12; %image frame subsampling step size for each Tx

step = 41;

alpha = 0.05;

r1 = chi2inv(alpha/2,Simulations*6)/Simulations;
r2 = chi2inv(1-alpha/2,Simulations*6)/Simulations;

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

% Going to test multiple different W matrices, changing one diag at a time
W = [1 0 0 0;
    0 2 0 0;
    0 0 1 0;
    0 0 0 2];

Z = chol(W); %Cholskey decomp will fail if not pos def
R = 1.75*eye(2); 
Z = dtsamp* [-A Gamma*W*Gamma'; O A'];
eZ = expm(Z);
F = eZ(7:12,7:12)';
Q = F * eZ(1:6,7:12);

Q(1,1) = 0.1;
%Q(2,2) = 1;

Q(4,4) = 0.1;
%Q(5,5) = 1;

Z = chol(Q);
%% Preallocate variables


xhat_plus = cell(Simulations,step);
P = cell(Simulations,step);
% inov = cell(Simulations,tstep+1);
% inovCov = cell(Simulations,tstep+1);
%y = cell(Simulations,tstep+1);

err = zeros(6,step);

epsilon = zeros(Simulations,step);

epMC = zeros(1,step);
%% MC Simulation of KF

for i=1:Simulations

P0 = diag([0.5 0.75 1 0.5 0.75 1]);
x0 = mvnrnd([0;0;0;0;0;0],P0)';

true = GenerateLineTruth(x0);

xvel = true(2,:);
yvel = true(5,:);

xAcc = true(3,:);
yAcc = true(6,:);

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

figure(1);hold on; grid on;
plot(epMC,'o')
yline(r1,'r--',LineWidth=1)
yline(r2,'r--',LineWidth=1)


xlabel('Time Step')
ylabel('NEES Statistic')
title('Chi-Squared NEES Test')

legend(str1,Location="southeast")

% pause()
% clc

