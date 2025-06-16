function [] = NEES_KF(KF,GT,m)
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

n_k = size(KF,2);
epsilon = zeros(m,size(KF,2));


for i = 1:m

    X = zeros(height(GT),size(KF,2));
    P = cell(1,size(KF,2));
    
    X(:,1) = KF{1}.x;
    P{1} =  KF{1}.P;

    for k = 2:n_k
        % Filter
        [x_minus,y_minus,P_minus,S] = KF_DS(KF{k-1});

        KF{k-1}.S = S;

        [Xk,Pk,~] = KF_MS(KF{k-1},x_minus,y_minus,P_minus);

        X(:,k) = Xk;
        P{k} = Pk;
    end

    err = X - GT;
   
 
    for k = 1:n_k
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

