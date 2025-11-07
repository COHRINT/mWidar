function [X] = random_track(mean, cov, x0)

    PLOT_FLAG = false;
    
    % System Dynamics
    A = [0 0 1 0 0 0;
        0 0 0 1 0 0;
        0 0 0 0 1 0;
        0 0 0 0 0 1;
        0 0 0 0 0 0;
        0 0 0 0 0 0];
    
    % Maximum timespan, 50 time steps, 5 seconds. Will be cut off if position exits scene
    dt = 0.1;
    X = zeros(6,50);
    
    %% IDEA: Keep position/acc states fixed through MC runs, only vary velocity?

    x0_vel = mvnrnd(mean,cov);
    x0 = [x0(1:2);x0_vel';x0(5:6)];

    X(:,1) = x0;

    for i = 2:100
        X(:,i) = expm(A*dt)*X(:,i-1);
        
    end

    if PLOT_FLAG
        figure(67); hold on; grid on
        plot(X(1,:),X(2,:),'--',LineWidth=1)
        xlim([0 129])
        ylim([0 128])
    end
end