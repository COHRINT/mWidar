clear;clc;close all


%% Call/Initialize KF
% x = [xpos xvel xacc ypos yvel yacc]'

c = physconst('LightSpeed'); %speed of light in m/s
dtsamp = 0.5*c*667e-12; %image frame subsampling step size for each Tx

% State covaraince
P0 = diag([0.75 0.75 1 0.75 0.75 1]);
x0 = abs(mvnrnd([0;0;0;0;0;0],P0)');

% To change trajectories, comment out/in the appropriate generate truth
% function

Truth = GenerateCircleTruth(x0);
%Truth = GenerateLineTruth(x0);
%Truth = GenerateVTruth(x0);

% Unpacking truth model data, for error calculations
xvel = Truth(2,:);
yvel = Truth(5,:);

xAcc = Truth(3,:);
yAcc = Truth(6,:);

xhat_0 = Truth(:,1);

y = [Truth(1,:);Truth(4,:)];

% # of timesteps
step = length(y);

% KF Matrices
Gamma = [0 0 0 0;1 0 0 0;0 1 0 0;0 0 0 0;0 0 1 0;0 0 0 1];
H = [1 0 0 0 0 0;0 0 0 1 0 0];

O = zeros(6); %6x6 matrix of 0's

% Approx dynamics model
A = [0 1 0 0 0 0;
    0 0 1 0 0 0;
    0 0 0 0 0 0;
    0 0 0 0 1 0;
    0 0 0 0 0 1;
    0 0 0 0 0 0]; % x = [xpos xvel xacc ypos yvel yacc]

%Van Loans Method

% Each trajectory has a different W and Q matrix, see the appropriate NEES
% test for the best Q and W

W = [1 0 0 0;
    0 2 0 0;
    0 0 1 0;
    0 0 0 2];

test = chol(W); %Cholskey decomp will fail if W is not pos def

% Measurment noise covariance
R = 1.75*eye(2); 

% Van Loans Method
Z = dtsamp* [-A Gamma*W*Gamma'; O A'];
eZ = expm(Z);
F = eZ(7:12,7:12)';
Q = F * eZ(1:6,7:12);

% Q tuning
Q(1,1) = 0.1;
Q(4,4) = 0.1;

Z = chol(Q); %Ensuring Q is pos def

% KF function
[xhat_plus_3,P_3,inov3,inovCov3] = mWidar_KF_PerfDA_Acc(xhat_0,y,P0,F,Q,R,step);

%% Error and Plot KF Results

% Preallocate variables
err_xpos = zeros(1,length(y));
err_ypos = zeros(1,length(y));

varx = zeros(1,length(y));
vary = zeros(1,length(y));

err_xvel = zeros(1,length(y));
err_yvel = zeros(1,length(y));

err_xAcc = zeros(1,length(y));
err_yAcc = zeros(1,length(y));

varxvel = zeros(1,length(y));
varyvel = zeros(1,length(y));

varxAcc = zeros(1,length(y));
varyAcc = zeros(1,length(y));

Obj_traj = zeros(2,length(y));


for k=1:length(y)
  
%%Uncomment the following to see the filter vs truth trajectory

    % Ellipse calculation
%     posCov = [P_3{k}(1,1) P_3{k}(1,4); P_3{k}(4,1) P_3{k}(4,4)];
%     muin = [xhat_plus_3{k}(1,:);xhat_plus_3{k}(4,:)];
%     [X3, Y3] = calc_gsigma_ellipse_plotpoints(muin,posCov,1,100);
%     
%     figure(1); clf; hold on; grid on
%     % fig = figure(1);
% 
%     % Plot trajectories
%     plot(xhat_plus_3{k}(1,:),xhat_plus_3{k}(4,:),'ms','MarkerSize',12,'LineWidth',1.2)
%     plot(y(1,k),y(2,k),'mx','MarkerSize',10,'LineWidth',10)
%     quiver(xhat_plus_3{k}(1,:),xhat_plus_3{k}(4,:),xhat_plus_3{k}(2,:),xhat_plus_3{k}(5,:),'r')
%     quiver(y(1,k),y(2,k),xvel(k),yvel(k),'k')
%     plot(X3, Y3)
%     xlim([min(y(1,:))-1 max(y(1,:))+1]);
%     ylim([min(y(2,:))-1 max(y(2,:))+1]);
%     title(['Object @ k=',num2str(k)])
%     legend('KF Estimate','True Trajectory','Estimated Velocity','Covariance',Location='southwest')
%     xlabel('X Pos'); ylabel('Y Pos')
% % 
%     pause(0.01);



    % err and variances
    err_xpos(k) = xhat_plus_3{k}(1) - y(1,k);
    err_ypos(k) = xhat_plus_3{k}(4) - y(2,k);
    
    err_xvel(k) = xhat_plus_3{k}(2) - xvel(k);
    err_yvel(k) = xhat_plus_3{k}(5) - yvel(k);

    err_xAcc(k) = xhat_plus_3{k}(3) - xAcc(k);
    err_yAcc(k) = xhat_plus_3{k}(6) - yAcc(k);

    varx(k) = P_3{k}(1,1);
    vary(k) = P_3{k}(4,4);

    varxvel(k) = P_3{k}(2,2);
    varyvel(k) = P_3{k}(5,5);

    varxAcc(k) = P_3{k}(3,3);
    varyAcc(k) = P_3{k}(6,6);

    errors = [err_xpos;err_xvel;err_xAcc;err_ypos;err_yvel;err_yAcc];

%     Save to gifs
%     filename = 'C:\Users\bijan\Documents\COHIRNT\mWidar\KF_PerfectDA_VelTuning.gif';
%     frame = getframe(fig);
%     Mov{k} = frame2im(frame); 
%     [Fr, map] = rgb2ind(Mov{k},256);
%     if k == 1
%         imwrite(Fr,map,filename,"gif","LoopCount",Inf,"DelayTime",1);
%     else
%         imwrite(Fr,map,filename,"gif","WriteMode","append","DelayTime",1);
%     end
end


%% Error Plots
sdx3 = varx.^2; sdy3 = vary.^2;
sdxvel = varxvel.^2; sdyvel = varyvel.^2;
sdxAcc = varxAcc.^2; sdyAcc = varyAcc.^2;

% Position Error
figure(); 
subplot(2,1,1);
hold on;
plot(err_xpos,'r')
plot(2*sdx3,'r--')
plot(-2*sdx3,'r--')
title('X Position Error for Circle Trajectory')
legend('X Error','2\sigma')
xlabel('Timestep')
ylabel('Error')

xlim([1 length(y)])
subplot(2,1,2);
hold on;
plot(err_ypos,'b')
plot(2*sdy3,'b--')
plot(-2*sdy3,'b--')
title('Y Position Error for Circle Trajectory')
legend('Y Error','2\sigma')
xlabel('Timestep')
ylabel('Error')
%ylim([-2 2])
xlim([1 length(y)])


% Velocity Error
figure()
subplot(2,1,1); hold on
plot(err_xvel,'r')
plot(2*sdxvel,'b--')
plot(-2*sdxvel,'b--')
title('X vel Error for Circle Trajectory')
legend('X vel Error','2\sigma')
xlabel('Timestep')
ylabel('Error')
xlim([1 length(y)])

subplot(2,1,2); hold on
plot(err_yvel,'r')
plot(2*sdyvel,'b--')
plot(-2*sdyvel,'b--')
title('Y vel Error for Circle Trajectory')
legend('Y vel Error','2\sigma')
xlabel('Timestep')
ylabel('Error')
xlim([1 length(y)])


%Acceleration Error
figure()
subplot(2,1,1); hold on; grid on
plot(err_xAcc,'r')
plot(2*sdxAcc,'b--')
plot(-2*sdxAcc,'b--')
title('X Acc Error for Circle Trajectory')
legend('X Acc Error','2\sigma')
xlabel('Timestep')
ylabel('Error')
xlim([1 length(y)])


subplot(2,1,2); hold on; grid on
plot(err_yAcc,'r')
plot(2*sdyAcc,'b--')
plot(-2*sdyAcc,'b--')
title('Y Acc Error for Circle Trajectory')
legend('Y Acc Error','2\sigma')
xlabel('Timestep')
ylabel('Error')
xlim([1 length(y)])


%% Innovation

inovVarx3 = zeros(1,length(y)-1);
inovVary3 = zeros(1,length(y)-1);


for k=1:length(y)-1

inovVarx3(k) = inovCov3(1,1,k);
inovVary3(k) = inovCov3(2,2,k);

end

inov_stdx3 = inovVarx3.^2;
inov_stdy3 = inovVary3.^2;

figure(); hold on
subplot(2,1,1)
title('X Innovation ~ Circle')
hold on
plot(inov3(1,:),'r')
plot(2*inov_stdx3,'k--')
plot(-2*inov_stdx3,'k--')
xlim([1 length(Obj_traj)])
%ylim([-1 1])
legend('Innovation','2\sigma')
subplot(2,1,2)
title('Y Innovation ~ Circle')
hold on
plot(inov3(2,:),'r')
plot(2*inov_stdy3,'k--')
plot(-2*inov_stdy3,'k--')
xlim([1 length(Obj_traj)])
%ylim([-1 1])
legend('Innovation','2\sigma')


