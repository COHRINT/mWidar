clear; clc; close all

% Test script meant to validate and debug my UKF before using it against
% heavy uncertainty.
tStart = cputime;

load linear_Tracks/CircleResults.mat
%load DubinsTrack.mat

%% Set GT
%GT =[Pos(1,:);trueVel(1,:);TrueAcc(1,:);Pos(2,:);trueVel(2,:);TrueAcc(2,:)];
%GT = Circ;

GT = C.GT;
y = [GT(1,:);GT(4,:)];


%rng(10)
figure(66)
plot(GT(1,:),GT(2,:),'b',LineWidth=1)
title('Ground Truth')
xlabel('X')
ylabel('Y')

c = physconst('LightSpeed'); %speed of light in m/s
dtsamp = 0.5*c*667e-12; %image frame subsampling step size for each Tx

%% Initialize KF

P0 = diag([0.25 0.5 0.75 0.25 0.5 0.75]); % For Reg
%P0 = diag([0.025 0.05 0.075 0.025 0.05 0.075]); % For SimMeas

x0 = GT(:,1);

tMax = length(GT);
t = linspace(0,tMax/10,tMax);

% Measurment noise covariance
R = 0.5*eye(2);
%R = eye(2); % Sim Meas


% Add noise to our measurments
for i=1:length(y)
    yNew = y+mvnrnd([0;0],0.01*eye(2))';
end

%% Process Noise Covariance

Q = C.Q;

%%Circle tuning ~ save back to struct soon
Q(1,1) = 0.001;
Q(4,4) = 0.001;

Q(:,2) = 0.5*Q(:,2); Q(2,:) =  0.5*Q(2,:);
Q(:,5) =  0.5*Q(:,5); Q(5,:) =  0.5*Q(5,:);

Q(:,3) = 0.05*Q(:,3); Q(3,:) = 0.05*Q(3,:);
Q(:,6) = 0.05*Q(:,6); Q(6,:) = 0.05*Q(6,:);

% Q = diag([0.01;0.01;0.01;0.01;0.01]);
% P0 = diag([0.25;0.5;0.75;0.5;0.5]);

f = @(x) LinearDynamics(x,dtsamp);
%f = @(x) DubinsModel(dtsamp,x,Q);
%% Call UKF

X = zeros(height(GT),length(y));
P = cell(1,length(y));

X(:,1) = x0;
P{1} = P0;

% M = load("sampling.mat").M;
% G = load("recovery.mat").G;

for k = 2:length(y)

    k
    % Dynamics Step
    [x_minus,y_minus,Pxx,Pyy,Pxy] = UKF_DynamicsStepLinear(X(:,k-1),P{k-1},f,Q,R);
    %[x_minus,y_minus,Pxx,Pyy,Pxy] = UKF_DynamicsStepSimMeas(X(:,k-1),P{k-1},f,Q,R,M,G);
    
    %% DEBUGGING SIMULATOR

    %yminTEST(:,k) = y_minus;


    % Measurment Step
    [X(:,k),P{k}] = UKF_MeasurmentStep(x_minus,y_minus,Pxx,Pyy,Pxy,yNew(:,k));

end

figure(88); hold on; grid on
tiledlayout(2,2)
nexttile
hold on;
plot(yminTEST(1,:),'b',LineWidth=1)
plot(yNew(1,:),'r',LineWidth=1)
nexttile
hold on;
plot(yminTEST(2,:),'b',LineWidth=1)
plot(yNew(2,:),'r',LineWidth=1)
legend('Predicted Meas','Actual Measurment')
nexttile
hold on;
plot(yminTEST(1,:) - yNew(1,:),'k',LineWidth=1)

nexttile
hold on;
plot(yminTEST(2,:) - yNew(2,:),'k',LineWidth=1)



tEnd = cputime - tStart;

%% Plot Trajectory and Errors

% Preallocate variables
stdXpos = zeros(1,length(X));
stdXvel = zeros(1,length(X));
stdXacc = zeros(1,length(X));
stdYpos = zeros(1,length(X));
stdYvel = zeros(1,length(X));
stdYacc = zeros(1,length(X));

Xposerr = zeros(1,length(X));
Xvelerr = zeros(1,length(X));
Xaccerr = zeros(1,length(X));

Yposerr = zeros(1,length(X));
Yvelerr = zeros(1,length(X));
Yaccerr = zeros(1,length(X));

for k =1:length(X)
% 
    figure(1); clf; hold on; grid on

    posCov = [P{k}(1,1) P{k}(1,4); P{k}(4,1) P{k}(4,4)];
    muin = [X(1,k);X(4,k)];
    [Xellip, Yellip] = calc_gsigma_ellipse_plotpoints(muin,posCov,1,100);

    plot(X(1,k),X(4,k),'ms','MarkerSize',12,'LineWidth',1.2)
    plot(y(1,k),y(2,k),'mx','MarkerSize',10,'LineWidth',10)
    plot(yNew(1,k),yNew(2,k),'ro','MarkerSize',10,'LineWidth',1)
    plot(Xellip, Yellip,'--k')

    xlim([min(y(1,:))-1 max(y(1,:))+1]);
    ylim([min(y(2,:))-1 max(y(2,:))+1]);
    title(['Object @ k=',num2str(k)])
    
  
    % Errors and variances
    stdXpos(k) = P{k}(1,1)^2;
    stdXvel(k) = P{k}(2,2)^2;
    stdXacc(k) = P{k}(3,3)^2;

    stdYpos(k) = P{k}(4,4)^2;
    stdYvel(k) = P{k}(5,5)^2;
    stdYacc(k) = P{k}(6,6)^2;
    
    Xposerr(k) = X(1,k) - GT(1,k);
    Xvelerr(k) = X(2,k) - GT(2,k);
    Xaccerr(k) = X(3,k) - GT(3,k);

    Yposerr(k) = X(4,k) - GT(4,k);
    Yvelerr(k) = X(5,k) - GT(5,k);
    Yaccerr(k) = X(6,k) - GT(6,k);

end

% figure(); hold on; grid on
% a = animatedline(NaN,NaN,'Color','m','LineWidth',1,'LineStyle','--');
% b = animatedline(NaN,NaN,'Color','k','LineWidth',1,'LineStyle','-');
% drawnow
% 
% 
% for k=1:length(X)
%     
%     xlim([min(y(1,k))-1 max(y(1,k))+1]);
%     ylim([min(y(2,k))-1 max(y(2,k))+1]);
% 
%     addpoints(a,X(1,k),X(4,k))
%     addpoints(b,y(1,k),y(2,k))
%     drawnow;
%     pause(dtsamp/10)
% 
% end
% drawnow


%% Plot GT
figure(); hold on; grid on

subplot(2,3,1)
plot(t,GT(1,:),'r',LineWidth=1)
title('X pos GT')
subplot(2,3,2)
plot(t,GT(2,:),'r',LineWidth=1)
title('X vel GT')
subplot(2,3,3)
plot(t,GT(3,:),'r',LineWidth=1)
title('X acc GT')
subplot(2,3,4)
plot(t,GT(4,:),'r',LineWidth=1)
title('Y pos GT')
subplot(2,3,5)
plot(t,GT(5,:),'r',LineWidth=1)
title('Y vel GT')
subplot(2,3,6)
plot(t,GT(6,:),'r',LineWidth=1)
title('Y acc GT')

%% Error plots


% Position errors
figure(); hold on; grid on

sgtitle('V Trajectory')

subplot(2,3,1)
hold on
plot(Xposerr,'k',LineWidth=1)
plot(2*stdXpos,'k--',LineWidth=1)
plot(-2*stdXpos,'k--',LineWidth=1)
title('X Position Error')
xlim([0 length(X)])
subplot(2,3,4)
hold on
plot(Yposerr,'k',LineWidth=1)
plot(2*stdYpos,'k--',LineWidth=1)
plot(-2*stdYpos,'k--',LineWidth=1)
title('Y Position Error')
xlim([0 length(X)])
% Velocity errors
%figure(); hold on; grid on

subplot(2,3,2)
hold on
plot(Xvelerr,'k',LineWidth=1)
plot(2*stdXvel,'k--',LineWidth=1)
plot(-2*stdXvel,'k--',LineWidth=1)
title('X Velocity Error')
xlim([0 length(X)])
subplot(2,3,5)
hold on
plot(Yvelerr,'k',LineWidth=1)
plot(2*stdYvel,'k--',LineWidth=1)
plot(-2*stdYvel,'k--',LineWidth=1)
title('Y Velocity Error')
xlim([0 length(X)])
%figure(); hold on; grid on

subplot(2,3,3)
hold on
plot(Xaccerr,'k',LineWidth=1)
plot(2*stdXacc,'k--',LineWidth=1)
plot(-2*stdXacc,'k--',LineWidth=1)
title('X Acceleration Error')
xlim([0 length(X)])
subplot(2,3,6)
hold on
plot(Yaccerr,'k',LineWidth=1)
plot(2*stdYacc,'k--',LineWidth=1)
plot(-2*stdYacc,'k--',LineWidth=1)
title('Y Acceleration Error')
xlim([0 length(X)])

% figure(); hold on; grid on
% plot(GT(1,:),GT(4,:),'k',LineWidth=1)
% plot(X(1,:),X(4,:),'m',LineWidth=1)
% title('Ground Truth vs UKF Estimate')
% legend('Ground Truth', 'Augmented UKF Estimate')
