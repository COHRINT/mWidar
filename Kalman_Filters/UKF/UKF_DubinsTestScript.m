clear; clc; close all

% Test our Augmented-UKF with Dubins model
rng(100)
load dubins_Tracks/DubinsCircle.mat xplus

GT = xplus;
y = [GT(1,:);GT(2,:)];

c = physconst('LightSpeed'); %speed of light in m/s
dt = 0.5*c*667e-12; %image frame subsampling step size for each Tx

%% Initialize KF

P0 = diag([0.25 0.25 0.1 0.25 0.1]); % Tone down angle uncertainty
x0 = GT(:,1);

t = linspace(0,length(GT)*dt,length(GT));

% Measurment noise covariance
%R = 0.025*eye(height(y));
R = 0.25*eye(height(y));

% Add noise to our measurments

yNew = y + mvnrnd([0;0],0.001*eye(height(y)),length(y))';


%% Process Noise

Q = [1e-3 0 0 0 0;
    0 1e-3 0 0 0;
    0 0 0.1 0 1e-6;
    0 0 0 1e-3 1e-6;
    0 0 1e-6 1e-6 1e-5];

%Q = 100*Q;
chol(Q);

f = @(x) DubinsModel(dt,x);

%% Call UKF

X = zeros(height(GT),length(y));
P = cell(1,length(y));

X_A = zeros(height(GT),length(y));
P_A = cell(1,length(y));

X(:,1) = x0;
X_A(:,1) = x0;
P{1} = P0;
P_A{1} = P0;

for k = 2:length(yNew)
    
    k

    [x_minus,y_minus,Pxx,Pyy,Pxy] = UKF_DS_Dubin(X(:,k-1),P{k-1},f,Q,R);
    [x_minus_A,y_minus_A,Pxx_A,Pyy_A,Pxy_A] = AUKF_DynamicsStepDubin(X_A(:,k-1),P_A{k-1},f,Q,R);

    ymin(:,k) = y_minus;
    yminA(:,k) = y_minus_A;

    [X(:,k),P{k}] = UKF_MeasurmentStep(x_minus,y_minus,Pxx,Pyy,Pxy,yNew(:,k));
    [X_A(:,k),P_A{k}] = UKF_MeasurmentStep(x_minus_A,y_minus_A,Pxx_A,Pyy_A,Pxy_A,yNew(:,k));
    
    X(3,k) = wrapToPi(X(3,k));
    X_A(3,k) = wrapToPi(X_A(3,k));

end




figure(88); hold on; grid on
tiledlayout(2,2)
nexttile
hold on;
plot(ymin(1,:),'b',LineWidth=1)
plot(yminA(1,:),'m',LineWidth=1)
plot(yNew(1,:),'r',LineWidth=1)
nexttile
hold on;
plot(ymin(2,:),'b',LineWidth=1)
plot(yminA(2,:),'m',LineWidth=1)
plot(yNew(2,:),'r',LineWidth=1)
legend('Predicted Meas (Non-Aug)','Predicted Meas (Aug)', 'Actual Measurment')

nexttile
hold on;
title('X innovation')
plot(ymin(1,:) - yNew(1,:),'k',LineWidth=1)
plot(yminA(1,:) - yNew(1,:),'r',LineWidth=1)
legend('Non-Augmented Innovations','Augmented Innovations')

nexttile
hold on;
title('Y innovation')
plot(ymin(2,:) - yNew(2,:),'k',LineWidth=1)
plot(yminA(2,:) - yNew(2,:),'r',LineWidth=1)


%% Plot Trajectory and Errors

% Preallocate variables

err = X - GT;
err_A = X_A - GT;

for i = 1:length(err_A)
    if err_A(3,i) > pi || err_A(3,i) < -pi
        err_A(3,i) = wrapToPi(err_A(3,i));
    end
end

std_x = zeros(1,length(X));
std_y = zeros(1,length(X));
std_theta = zeros(1,length(X));
std_v = zeros(1,length(X));
std_omega = zeros(1,length(X));

std_xA = zeros(1,length(X));
std_yA = zeros(1,length(X));
std_thetaA = zeros(1,length(X));
std_vA = zeros(1,length(X));
std_omegaA = zeros(1,length(X));

for i = 1:length(X)
% % 
%     figure(1); clf; hold on; grid on
% 
%     posCov = [P_A{i}(1,1) P_A{i}(1,2); P_A{i}(2,1) P_A{i}(2,2)];
%     muin = [X_A(1,i);X_A(2,i)];
%     [Xellip, Yellip] = calc_gsigma_ellipse_plotpoints(muin,posCov,1,100);
% 
%     plot(X_A(1,i),X_A(2,i),'ms','MarkerSize',12,'LineWidth',1.2)
%     quiver(X_A(1,i),X_A(2,i),X_A(4,i)*cos(X_A(3,i)),X_A(4,i)*sin(X_A(3,i)))
%     quiver(GT(1,i),GT(2,i),GT(4,i)*cos(GT(3,i)),GT(4,i)*sin(GT(3,i)))
%     plot(y(1,i),y(2,i),'mx','MarkerSize',10,'LineWidth',10)
%     plot(yNew(1,i),yNew(2,i),'ro','MarkerSize',10,'LineWidth',1)
%     plot(Xellip, Yellip,'--k')
% 
%     xlim([min(y(1,:))-1 max(y(1,:))+1]);
%     ylim([min(y(2,:))-1 max(y(2,:))+1]);
%     title(['Object @ k=',num2str(i)])

    std_x(i) = sqrt(P{i}(1,1));
    std_y(i) = sqrt(P{i}(2,2));
    std_theta(i) = sqrt(P{i}(3,3));
    std_v(i) = sqrt(P{i}(4,4));
    std_omega(i) = sqrt(P{i}(5,5));

    std_xA(i) = sqrt(P_A{i}(1,1));
    std_yA(i) = sqrt(P_A{i}(2,2));
    std_thetaA(i) = sqrt(P_A{i}(3,3));
    std_vA(i) = sqrt(P_A{i}(4,4));
    std_omegaA(i) = sqrt(P_A{i}(5,5));

end

figure(44); hold on; grid on
plot(y(1,:),y(2,:),'b',LineWidth=1)
plot(X_A(1,:),X_A(2,:),'r--',LineWidth=0.5)
%plot(X_A(1,:)+2*std_xA,X_A(2,:)+2*std_yA,'k--',LineWidth=0.5)
%plot(X_A(1,:)-2*std_xA,X_A(2,:)-2*std_yA,'k--',LineWidth=0.5)
title('GT v. Estimated Track')
legend('GT','UKF Estimate')
xlabel('X Pos')
ylabel('Y Pos')

figure(); hold on; grid on
tiledlayout(5,1)

nexttile
hold on; grid on
title('X pos')
%plot(t,X(1,:),'k--')
plot(t,X_A(1,:),'r--')
plot(t,GT(1,:),'b',LineWidth=1)
xlim([0 t(end)])
legend('UKF Estimate (Augmented)','Ground Truth',Location='northwest')

nexttile
hold on; grid on
title('Y pos')
%plot(t,X(2,:),'k--')
plot(t,X_A(2,:),'r--')
plot(t,GT(2,:),'b',LineWidth=1)
xlim([0 t(end)])

nexttile
hold on; grid on
title('\theta')
%plot(t,X(3,:),'k--')
plot(t,X_A(3,:),'r--')
plot(t,GT(3,:),'b',LineWidth=1)
xlim([0 t(end)])

nexttile
hold on; grid on
title('v')
%plot(t,X(4,:),'k--')
plot(t,X_A(4,:),'r--')
plot(t,GT(4,:),'b',LineWidth=1)
xlim([0 t(end)])

nexttile
hold on; grid on
title('\omega')
%plot(t,X(5,:),'k--')
plot(t,X_A(5,:),'r--')
plot(t,GT(5,:),'b',LineWidth=1)
xlim([0 t(end)])

figure(); hold on; grid on
tiledlayout(5,1)

nexttile
hold on; grid on
title('X Pos error')
plot(t,err(1,:),'k',LineWidth=1)
plot(t,2*std_x,'k--',LineWidth=1)
plot(t,-2*std_x,'k--',LineWidth=1)
xlim([0 t(end)])

nexttile
hold on; grid on
title('Y Pos error')
plot(t,err(2,:),'k',LineWidth=1)
plot(t,2*std_y,'k--',LineWidth=1)
plot(t,-2*std_y,'k--',LineWidth=1)
xlim([0 t(end)])

nexttile
hold on; grid on
title('Bearing error')
plot(t,err(3,:),'k',LineWidth=1)
plot(t,2*std_theta,'k--',LineWidth=1)
plot(t,-2*std_theta,'k--',LineWidth=1)
ylim([-2*pi 2*pi])
xlim([0 t(end)])

nexttile
hold on; grid on
title('Translational Velocity error')
plot(t,err(4,:),'k',LineWidth=1)
plot(t,2*std_v,'k--',LineWidth=1)
plot(t,-2*std_v,'k--',LineWidth=1)
xlim([0 t(end)])

nexttile
hold on; grid on
title('Angular Velocity error')
plot(t,err(5,:),'k',LineWidth=1)
plot(t,2*std_omega,'k--',LineWidth=1)
plot(t,-2*std_omega,'k--',LineWidth=1)
xlim([0 t(end)])

figure(); hold on; grid on
tiledlayout(5,1)

nexttile
hold on; grid on
title('X Pos error')
plot(t,err_A(1,:),'b',LineWidth=1)
plot(t,2*std_xA,'b--',LineWidth=1)
plot(t,-2*std_xA,'b--',LineWidth=1)
xlim([0 t(end)])

nexttile
hold on; grid on
title('Y Pos error')
plot(t,err_A(2,:),'b',LineWidth=1)
plot(t,2*std_yA,'b--',LineWidth=1)
plot(t,-2*std_yA,'b--',LineWidth=1)
xlim([0 t(end)])

nexttile
hold on; grid on
title('Bearing error')
plot(t,err_A(3,:),'b',LineWidth=1)
plot(t,2*std_thetaA,'b--',LineWidth=1)
plot(t,-2*std_thetaA,'b--',LineWidth=1)
ylim([-2*pi 2*pi])
xlim([0 t(end)])

nexttile
hold on; grid on
title('Translational Velocity error')
plot(t,err_A(4,:),'b',LineWidth=1)
plot(t,2*std_vA,'b--',LineWidth=1)
plot(t,-2*std_vA,'b--',LineWidth=1)
xlim([0 t(end)])

nexttile
hold on; grid on
title('Angular Velocity error')
plot(t,err_A(5,:),'b',LineWidth=1)
plot(t,2*std_omegaA,'b--',LineWidth=1)
plot(t,-2*std_omegaA,'b--',LineWidth=1)
xlim([0 t(end)])
%% Perform NEES Statistic for m MC runs

% Will define x0 to be our GT corrupted by AWGN w/ cov P0

x0 = mvnrnd(GT(:,1),P0)';
m = 50;

NEES(x0, P0, m ,y, GT, Q, R, f)

