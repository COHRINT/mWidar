function [] = NEES(x0, P0, m ,y, GT, Q, R, f)
%%%%%% NEES %%%%%%%%%
%{
Function to generate a NEES statistic for any given state estimate x, with
covariance P. m denotes the number of MC runs for the given tracker

x0 and P0 are the initial state and covaraince, which will remain constant
throughout the MC runs

y will be the corrupted measurments, which will also be constant throughout
the MC runs

GT is our ground truth data for a given trajectory

%}

epsilon = zeros(m,length(y));

for i = 1:m

    X = zeros(height(GT),length(y));
   

    P = cell(1,length(y));
    
    X(:,1) = x0;
    P{1} = P0;

    for k = 2:length(y)
        % Filter
        [x_minus,y_minus,Pxx,Pyy,Pxy] = AUKF_DynamicsStepDubin(X(:,k-1),P{k-1},f,Q,R);
        [X(:,k),P{k}] = UKF_MeasurmentStep(x_minus,y_minus,Pxx,Pyy,Pxy,y(:,k));
    
        % Angle wrap
        X(3,k) = wrapToPi(X(3,k));
    end

    err = X - GT;

        
 
    for k = 1:length(y)
        if err(3,i) > pi || err(3,i) < -pi
            err(3,i) = wrapToPi(err(3,i));
        end
        epsilon(i,k) = err(:,k)'/P{k}*err(:,k);
    end

end

mean_epsilon = mean(epsilon,1);

alpha = 0.05;
r1 = chi2inv(alpha/2,m*5)/m;
r2 = chi2inv(1-alpha/2,m*5)/m;


figure(87);hold on; grid on;
plot(mean_epsilon,'-x')
yline(r1,'r--',LineWidth=1)
yline(r2,'r--',LineWidth=1)
xlim([0 length(GT)])
ylim([r1-1 r2+50])
xlabel('Time Step')
ylabel('NEES Statistic')
title('Chi-Squared NEES Test')


end

