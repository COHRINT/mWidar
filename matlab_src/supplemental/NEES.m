
function [] = NEES(handle,initial_state,A,n,M,G)
%{
    NEES.m: perform NEES test for a given initial_state and filter class handle.
    handle -> class handle
    initial_state -> initial state mean x0 and covariance P0
    KF -> Struct contianing KF matrices Q, R, F, H, A
    n -> # of MC runs
    M,G -> sampling/ recovery matrix


%}

x0 = initial_state.x0;
P0 = initial_state.P0;
dt = 0.1;

detector = 'CFAR';

tvec = 0:dt:10;
n_t = size(tvec,2);

performance = cell(1,n_t);
X = zeros(6,n_t);
epsilon = zeros(n,n_t);


for i = 1:n
    
    %% Generate new G.T. sampling from gaussian initial state
    x = mvnrnd(x0,P0);
    GT = generate_track(x,A,tvec); % Generate track ensures all values lie in mWidar scene
    [y, signal] = sim_mWidar_image(n_t,GT,M,G,detector,false);

    Data.GT = GT;
    Data.y = y;
    Data.signal = signal;

    performance{1}.x = GT(:,1);
    X(:,1) = GT(:,1);
    performance{1}.P = P0;

    %% Run Filter
    for k = 2:n_t
        
        [X_est,P] = handle.timestep(performance{k-1}.x, performance{k-1}.P,y{k});

        % update performance
        performance{k}.x = X_est;
        performance{k}.P = P;
        
        X(:,k) = X_est;
    end

    %% NEES statistic

    err = X - GT;

    for k = 1:n_t
        epsilon(i,k) = err(:,k)'/performance{k}.P*err(:,k);
    end


end

%% Average over all MC runs and plot

mean_epsilon = mean(epsilon,1);

alpha = 0.05;
r1 = chi2inv(alpha/2,n*6)/n;
r2 = chi2inv(1-alpha/2,n*6)/n;


figure(87);hold on; grid on;
plot(mean_epsilon,'-x')
yline(r1,'r--',LineWidth=1)
yline(r2,'r--',LineWidth=1)
xlim([0 length(GT)])
%ylim([r1-1 r2+50])
xlabel('Time Step')
ylabel('NEES Statistic')
title('Chi-Squared NEES Test')

end