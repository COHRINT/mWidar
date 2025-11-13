function [Signal,Pos_states] = random_tracks(obj_case,M,G)
    %{
        random_tracks:
            For use in run_MC_Detector. Given a number of objects from 1 - 5, generate a track with the given number of objects
            
            Inputs:
            obj_case:

            5 different cases, each case keeps the objects from previous cases and adds a new one:

            1: A single object moving from the left side of the scene to the right.
            2: An object moving from the bottom left corner to the top right corner
            3: An object starting in the top right and moving in a parabola
            4: An object starting in the bottom right and moving to the top left
            5: An object starting at the top of the scene and moving straight down

            Each time the function is called, the track will be slightly different, with the velocity [vx, vy] being picked at random. The start position and acceleration is 
            the same. Each case will run 50 timesteps.

            M,G: sampling and recovery matrices for mWidar signal simulation

            Outputs:
            Signal, 128x128x50 array of mWidar signal at each timestep
            Pos_states, position states 

    %}
    
    % System Dynamics
    A = [0 0 1 0 0 0;
        0 0 0 1 0 0;
        0 0 0 0 1 0;
        0 0 0 0 0 1;
        0 0 0 0 0 0;
        0 0 0 0 0 0];
    
    % Ensure case is an integer
    obj_case = floor(obj_case);

    if obj_case > 5 || obj_case < 1
        fprintf("INVALID CASE, MUST BE INTEGER BETWEEN 1 AND 5")
        return;
    end

    % Maximum timespan, 50 time steps, 5 seconds. Will be cut off if position exits scene
    dt = 0.1;
    tvec = 0:dt:5;
    X = cell(obj_case); % Ground Truth
    Pos_states = zeros(obj_case,2,50); % Only position states, for signal generation
    Signal = zeros(128,128,50);

    
    % Define initial condtions

    % Track 1
    mu_v = [10; 0];

    % Control how spread out velocity is across different cases
    v_spread_x = 20; 
    v_spread_y = 5;

    cov_v = diag([v_spread_x v_spread_y]);
    vel_states = mvnrnd(mu_v, cov_v);

    x0(:,1) = [1 60 vel_states 0 0]';
    
    % Track 2
    mu_v = [10; 10];

    % Control how spread out velocity is across different cases
    v_spread_x = 15; 
    v_spread_y = 15;

    cov_v = diag([v_spread_x v_spread_y]);
    vel_states = mvnrnd(mu_v, cov_v);

    x0(:,2) = [1 25 vel_states 0 0]';

    % Track 3

    mu_v = [10; -25];
    % Control how spread out velocity is across different cases
    v_spread_x = 15; 
    v_spread_y = 15;

    cov_v = diag([v_spread_x v_spread_y]);
    vel_states = mvnrnd(mu_v, cov_v);

    x0(:,3) = [1 120 vel_states 0 5]';

    % Track 4

    mu_v = [-10; 10];
    v_spread_x = 15; 
    v_spread_y = 15;

    cov_v = diag([v_spread_x v_spread_y]);
    vel_states = mvnrnd(mu_v, cov_v);

    x0(:,4) = [120 25 vel_states 0 0]';

    % Track 5

    mu_v = [0; -10];
    v_spread_x = 5; 
    v_spread_y = 20;

    cov_v = diag([v_spread_x v_spread_y]);
    vel_states = mvnrnd(mu_v, cov_v);

    x0(:,5) = [64 120 vel_states 0 0]';

    for k = 1:50
        % Object binary scene
        S = zeros(128,128);

        for c = 1:obj_case
            switch c

                case 1
                % A single object moving from the left side of the scene to the right.

                X{c}(:,k) = expm(A*tvec(k))*x0(:,c);
                Pos_states(c,:,k) = floor([X{c}(1,k) X{c}(2,k)]);

                % Ensure object is still in scene, cut off generate at y = 20
                if Pos_states(c,1,k) <= 128 || Pos_states(c,1,k) >= 1 || Pos_states(c,2,k) <= 128 || Pos_states(c,2,k) >= 20
                    S(Pos_states(c,1,k), Pos_states(c,2,k)) = 1;
                end

                case 2
                %  An object moving from the bottom left corner to the top right corner
                
                X{c}(:,k) = expm(A*tvec(k))*x0(:,c);
                Pos_states(c,:,k) = floor([X{c}(1,k) X{c}(2,k)]);
                
                if Pos_states(c,1,k) <= 128 || Pos_states(c,1,k) >= 1 || Pos_states(c,2,k) <= 128 || Pos_states(c,2,k) >= 20
                    S(Pos_states(c,1,k), Pos_states(c,2,k)) = 1;
                end

                case 3
                % An object starting in the top right and moving in a parabola
                
                X{c}(:,k) = expm(A*tvec(k))*x0(:,c);
                Pos_states(c,:,k) = floor([X{c}(1,k) X{c}(2,k)]);
                
                if Pos_states(c,1,k) <= 128 || Pos_states(c,1,k) >= 1 || Pos_states(c,2,k) <= 128 || Pos_states(c,2,k) >= 20
                    S(Pos_states(c,1,k), Pos_states(c,2,k)) = 1;
                end

                case 4
                % An object starting in the bottom right and moving to the top left

                X{c}(:,k) = expm(A*tvec(k))*x0(:,c);
                Pos_states(c,:,k) = floor([X{c}(1,k) X{c}(2,k)]);

                if Pos_states(c,1,k) <= 128 || Pos_states(c,1,k) >= 1 || Pos_states(c,2,k) <= 128 || Pos_states(c,2,k) >= 20
                    S(Pos_states(c,1,k), Pos_states(c,2,k)) = 1;
                end

                case 5
                % An object starting at the top of the scene and moving straight down
                
                X{c}(:,k) = expm(A*tvec(k))*x0(:,c);
                Pos_states(c,:,k) = floor([X{c}(1,k) X{c}(2,k)]);

                if Pos_states(c,1,k) <= 128 || Pos_states(c,1,k) >= 1 || Pos_states(c,2,k) <= 128 || Pos_states(c,2,k) >= 20
                    S(Pos_states(c,1,k), Pos_states(c,2,k)) = 1;
                end

            end

            %% Signal Generation

            signal_flat = S';
            signal_flat = signal_flat(:);
            signal_flat = M * signal_flat;
            signal_flat = G' * signal_flat;
            sim_signal = reshape(signal_flat, 128, 128)';
            signal_original = imgaussfilt(sim_signal, 2);

            % Normalize signal [0 1]
            scaled_signal = (signal_original - min(signal_original(:)))/ (max(signal_original(:)) - min(signal_original(:)));
            Signal(:,:,k) = scaled_signal;

        end 
    end
end