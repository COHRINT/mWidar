function [] = mWidar_FilterPlot(KF,GT,tvec,sim_signal,plot_traj,plot_mWidarimg,plot_error,plot_innov,save_traj,save_error,save_innov,xidx,yidx,state_list,color)
%% Plots trajectory, error, and innovations for a given state, ground truth and cov
%{
INPUTS:
X - i x n matrix of state where i is the dimension of s.s and n is the # of
time steps
P - cell array of covaraince at time step n
GT - i x n ground truth
y - measurments
innov - innovations
S - innovation Covaraince
tvec - time vector

Function is a mess I will fix it later
%}


innov = KF{1}{1}.innov;

n_PDAF = size(GT,2); % # of PDAFs to plot
n_k = size(GT{1},2); % # of time steps
n_i = size(GT{1},1); % # of states
n_y = size(innov,1); % # of observed states

std = cell(n_PDAF,n_i);
std_innov = cell(n_PDAF,n_y);

X = cell(1,n_PDAF);
P = cell(n_PDAF,n_k);
S = cell(n_PDAF,n_k);
y = cell(n_PDAF,n_k);
innov = cell(1,n_PDAF);

for k = 1:n_k
    for j = 1:n_PDAF
        X{j}(:,k) = KF{j}{k}.x;
        P{j,k} = KF{j}{k}.P;
        S{j,k} = KF{j}{k}.S;
        y{j,k} = KF{j}{k}.z;
        innov{j}(:,k) = KF{j}{k}.innov;
    end
end

for k = 1:n_k
    for i = 1:n_i
        for j = 1:n_PDAF
            std{j,i}(k) = sqrt(P{j,k}(i,i));
        end
    end
    for i = 1:n_y
        for j = 1:n_PDAF
            std_innov{j,i}(k) = sqrt(S{j,k}(i,i));
        end
    end
end

Lscene = 4;
npx = 128;
xgrid = linspace(-2,2,npx);
ygrid = linspace(0,Lscene,npx);
[pxgrid,pygrid] = meshgrid(xgrid,ygrid);

err = cell(1,n_PDAF);
for j = 1:n_PDAF
    err{j} = X{j} - GT{j};
end

if plot_traj
    if save_traj
        filename = input('Enter filename for trajectory plots:', "s");
        filename = append('Results/',filename);
    end
    for k = 1:n_k
        figure(66); clf; hold on; grid on
        for j = 1:n_PDAF
            posCov = [P{j,k}(xidx,xidx) P{j,k}(xidx,yidx); P{j,k}(yidx,xidx) P{j,k}(yidx,yidx)];
            muin = [X{j}(xidx,k);X{j}(yidx,k)];
            [Xellip, Yellip] = calc_gsigma_ellipse_plotpoints(muin,posCov,1,100);

            plot3(X{j}(xidx,k),X{j}(yidx,k),ones(length(GT),1),'ms','MarkerSize',12,'LineWidth',1.2)
            plot3(GT{j}(1,k),GT{j}(4,k),ones(length(GT),1),'mx','MarkerSize',10,'LineWidth',10)
            scatter3(y{j,k}(1,:),y{j,k}(2,:),ones(length(y{j,k}(1,:)),1), '*r')
            plot3(Xellip, Yellip,ones(length(Xellip),1),'--k')

            if plot_mWidarimg
                surf(pxgrid,pygrid,sim_signal{k}/(max(max(sim_signal{k}))),'EdgeColor','none')
            end
        end
%         xlim([min(GT(xidx,:))-1 max(GT(xidx,:))+1]);
%         ylim([min(GT(yidx,:))-1 max(GT(yidx,:))+1]);
        title(['Object @ k=',num2str(k)])
        %legend('Estimated Trajectory','','Ground Truth','Detections','Validation Region')
        

        if save_traj
            %%Turn fig into a gif
            fig = figure(66);
            frame = getframe(fig);
            Mov{k} = frame2im(frame); 
            [Fr, map] = rgb2ind(Mov{k},256);
            if k == 1
                imwrite(Fr,map,filename,"gif","LoopCount",Inf,"DelayTime",0.1);
            else
                imwrite(Fr,map,filename,"gif","WriteMode","append","DelayTime",0.1);
            end

        end

    end

    
end

if plot_error
    
    for j = 1:n_PDAF 
        figure(44+j); hold on; grid on
    
        tiledlayout(n_i,1)
        for i = 1:n_i
            nexttile
            sgtitle('Error Results')
            hold on; grid on
              
            plot(tvec,err{j}(i,:),color(j),LineWidth=1);
            plot(tvec,2*std{j,i},color(j),LineWidth=1,LineStyle="--")
            plot(tvec,-2*std{j,i},color(j),LineWidth=1,LineStyle="--")
            
        
            title(state_list(i))
        end

        if save_error
        filename = input('Enter filename for error plots: ', "s");
        filename = append('Results/',filename);
        print(filename,'-r300','-dpng')
        end
    end
    
    figure(12); hold on; grid on
    for j = 1:n_PDAF
        plot(X{j}(xidx,:),X{j}(yidx,:),color(j),LineWidth=1,LineStyle='-.')
        plot(GT{j}(xidx,:),GT{j}(yidx,:),'r--',LineWidth=1)
    end
    title("True Trajectory vs Estimated Trajectory")

    
end

if plot_innov
    for j = 1:n_PDAF
    figure(88+j); hold on; grid on
    tiledlayout(n_y,1);
    for i = 1:n_y
        nexttile
        sgtitle('Innovation Results')
        hold on; grid on
        plot(tvec(1:end-1),innov{j}(i,1:end-1),color(j),LineWidth=1)
        plot(tvec(1:end-1),2*std_innov{j,i}(1:end-1),color(j),LineWidth=1,LineStyle="--")
        plot(tvec(1:end-1),-2*std_innov{j,i}(1:end-1),color(j),LineWidth=1,LineStyle="--")
    end
    end
    if save_innov
        filename = input('Enter filename for innovation plots: ', "s");
        filename = append('Results/',filename);
        print(filename,'-r300','-dpng')
    end
end


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
