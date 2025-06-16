function [] = mWidar_LinearKF_plot(KF,GT,t_vec)
%% Plot results from Kalman Filter

% Note: plots consider a state vector of [xpos,xvel,xacc,ypos,yvel,yacc].

n_k = size(GT,2);

X = zeros(6,n_k);
P = cell(1,n_k);
S = cell(1,n_k);
innov = zeros(2,n_k);
for i = 1:n_k
    X(:,i) = KF{i}.x;
    P{i} = KF{i}.P;
    S{i} = KF{i}.S;
    innov(:,i) = KF{i}.innov;
end

stdx = zeros(1,n_k);
stdy = zeros(1,n_k); 
stdvx = zeros(1,n_k);
stdvy = zeros(1,n_k); 
stdax = zeros(1,n_k);
stday = zeros(1,n_k);

std_xinnov = zeros(1,n_k);
std_yinnov = zeros(1,n_k);

err = X - GT;

for k = 1:n_k
    figure(66); clf; hold on; grid on

    posCov = [P{k}(1,1) P{k}(1,4); P{k}(4,1) P{k}(4,4)];
    muin = [X(1,k);X(4,k)];
    [Xellip, Yellip] = calc_gsigma_ellipse_plotpoints(muin,posCov,1,100);

    plot(X(1,k),X(4,k),'ms','MarkerSize',12,'LineWidth',1.2)
    quiver(X(1,k),X(4,k),X(2,k),X(5,k))
    quiver(GT(1,k),GT(4,k),GT(2,k),GT(5,k))
    plot(GT(1,k),GT(4,k),'mx','MarkerSize',10,'LineWidth',10)
    plot(Xellip, Yellip,'--k')

    xlim([min(GT(1,:))-1 max(GT(1,:))+1]);
    ylim([min(GT(4,:))-1 max(GT(4,:))+1]);
    title(['Object @ k=',num2str(k)])

    stdx(k) = sqrt(P{k}(1,1));
    stdy(k) = sqrt(P{k}(4,4));
    stdvx(k) = sqrt(P{k}(2,2));
    stdvy(k) = sqrt(P{k}(5,5));
    stdax(k) = sqrt(P{k}(3,3));
    stday(k) = sqrt(P{k}(6,6));
    std_xinnov(k) = sqrt(S{k}(1,1));
    std_yinnov(k) = sqrt(S{k}(2,2));
end

figure(); hold on; grid on
tiledlayout(6,1)
nexttile
hold on; grid on
plot(t_vec,err(1,:),'k',LineWidth=1);
plot(t_vec,2*stdx,'k--',LineWidth=1)
plot(t_vec,-2*stdx,'k--',LineWidth=1)
title('X Position [m]')

nexttile
hold on; grid on
plot(t_vec,err(2,:),'k',LineWidth=1);
plot(t_vec,2*stdvx,'k--',LineWidth=1)
plot(t_vec,-2*stdvx,'k--',LineWidth=1)
title('X Velocity [m/s]')

nexttile
hold on; grid on
plot(t_vec,err(3,:),'k',LineWidth=1);
plot(t_vec,2*stdax,'k--',LineWidth=1)
plot(t_vec,-2*stdax,'k--',LineWidth=1)
title('X Acceleration [m/s^2]')

nexttile
hold on; grid on
plot(t_vec,err(4,:),'k',LineWidth=1);
plot(t_vec,2*stdy,'k--',LineWidth=1)
plot(t_vec,-2*stdy,'k--',LineWidth=1)
title('Y Position [m]')

nexttile
hold on; grid on
plot(t_vec,err(5,:),'k',LineWidth=1);
plot(t_vec,2*stdvy,'k--',LineWidth=1)
plot(t_vec,-2*stdvy,'k--',LineWidth=1)
title('Y Velocity [m/s]')

nexttile
hold on; grid on
plot(t_vec,err(5,:),'k',LineWidth=1);
plot(t_vec,2*stday,'k--',LineWidth=1)
plot(t_vec,-2*stday,'k--',LineWidth=1)
title('Y Acceleration [m/s^2]')

figure(); hold on; grid on
tiledlayout(2,1)
nexttile
hold on; grid on
plot(t_vec,innov(1,:),'k',LineWidth=1);
plot(t_vec,2*std_xinnov,'k--',LineWidth=1)
plot(t_vec,-2*std_xinnov,'k--',LineWidth=1)
title('x innovations')

nexttile
hold on; grid on
plot(t_vec,innov(2,:),'k',LineWidth=1);
plot(t_vec,2*std_yinnov,'k--',LineWidth=1)
plot(t_vec,-2*std_yinnov,'k--',LineWidth=1)
title('y innovations')
end

%% More Functions for plotting

%calc_gsigma_ellipse_plotpoints.m
%%Does all the computations needed to plot 2D Gaussian ellipses properly.
%%Takes in the Gaussian mean (muin), cov matrix (Sigma, positive definite),
%%g-sigma value, and number of points to generate for plotting. 
function [X, Y] = calc_gsigma_ellipse_plotpoints(muin,Sigma,g,npoints)
%%align the Gaussian along its principal axes
[R,D,thetalocx] = subf_rotategaussianellipse(Sigma,g);
%%pick semi-major and semi-minor "axes" (lengths)
if Sigma(1)<Sigma(4) %use if sigxx<sigyy
    a = 1/sqrt(D(4)); 
    b = 1/sqrt(D(1)); 
elseif Sigma(1)>=Sigma(4)
    a= 1/sqrt(D(1)); %use if sigxx<sigyy
    b= 1/sqrt(D(4)); 
end

%%calculate points of ellipse:
mux = muin(1);
muy = muin(2);
if Sigma(2)~=0
    [X Y] = calculateEllipse(mux, muy, a, b, rad2deg(thetalocx),npoints); 
else %if there are no off-diagonal terms, then no rotation needed
    [X Y] = calculateEllipse(mux, muy, a, b, 0,npoints);
end

%%call the rotate gaussian ellipse thingy as a local subfunction (there is
%%also a separate fxn m-file for this, too)
function [R,D,thetalocx] = subf_rotategaussianellipse(Sigma,g)
P = inv(Sigma);
P = 0.5*(P+P'); %symmetrize

a11 = P(1);
a12 = P(2);
a22 = P(4);
c = -g^2;

mu = 1/(-c); %can define mu this way since b1=0,b2=0 b/c we are mean centered
m11 = mu*a11;
m12 = mu*a12;
m22 = mu*a22;

%%solve for eigenstuff
lambda1 = 0.5*(m11 + m22 + sqrt((m11-m22).^2 + 4*m12.^2) );
lambda2 = 0.5*(m11 + m22 - sqrt((m11-m22).^2 + 4*m12.^2) );
% % b = 1/sqrt(lambda1); %semi-minor axis for standard ellipse (length)
% % a = 1/sqrt(lambda2); %semi-major axis for standard ellipse (length)
D = diag([lambda2,lambda1]); %elements are 1/a^2 and 1/b^2, respectively
%%Choose the mahor axis direction of the ellipse
if m11>=m22
    u11 = lambda1 - m22;
    u12 = m12;
elseif m11<m22
    u11 = m12;
    u12 = lambda1 - m11;    
end
norm1 = sqrt( u11.^2 + u12.^2  );
U1 =  ([u11; u12]) / norm1; %major axis direction
U2 = [-u12; u11]; %minor axis direction
R = [U1,U2];
% if sum(sum(isnan(R)))>0
%    R = eye(2); %default hack for now in case of degenerate stuff
% end

thetalocx = 0.5*atan( -2*a12/(a22-a11) );
% if isnan(thetalocx)
%     thetalocx = 0; %default hack for now...
% end

end

end

function [X Y] = calculateEllipse(x, y, a, b, angle, steps) 
    %# This functions returns points to draw an ellipse 
    %# 
    %#  @param x     X coordinate 
    %#  @param y     Y coordinate 
    %#  @param a     Semimajor axis 
    %#  @param b     Semiminor axis 
    %#  @param angle Angle of the ellipse (in degrees) 
    %# 
 
    %error(nargchk(5, 6, nargin)); 
    narginchk(5, 6);
    if nargin<6, steps = 36; end 
 
    beta = angle * (pi / 180); 
    sinbeta = sin(beta); 
    cosbeta = cos(beta); 
 
    alpha = linspace(0, 360, steps)' .* (pi / 180); 
    sinalpha = sin(alpha); 
    cosalpha = cos(alpha); 
 
    X = x + (a * cosalpha * cosbeta - b * sinalpha * sinbeta); 
    Y = y + (a * cosalpha * sinbeta + b * sinalpha * cosbeta); 
 
    if nargout==1, X = [X Y]; end 
end 
