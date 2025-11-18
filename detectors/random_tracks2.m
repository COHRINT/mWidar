function [Signal, Pos_states] = random_tracks2(obj_num, M, G)
    %{
        random_tracks:
            For use in run_MC_Detector. Given a number of objects from 1 - 5, generate a track with the given number of objects

            Inputs:
            obj_num: Number of ground truth objects
            M: Sampling matrix for mWidar signal simulation
            G: Recovery matrix for mWidar signal simulation

            Each time the function is called, the track will be slightly different, with the velocity [vx, vy] being picked at random. The start position and acceleration is
            the same. Each case will run 50 timesteps.

            M,G: sampling and recovery matrices for mWidar signal simulation

            Outputs:
            Signal, 128x128x50 array of mWidar signal at each timestep
            Pos_states, position states

    %}

    % System Dynamics (Continuous-time)
    A = [0 0 1 0 0 0;
         0 0 0 1 0 0;
         0 0 0 0 1 0;
         0 0 0 0 0 1;
         0 0 0 0 0 0;
         0 0 0 0 0 0];

    % Maximum timespan, 10 time steps, 1 second. Will be cut off if position exits scene
    dt = 0.1;
    tvec = 0:dt:1-dt;
    
    % Discrete-time state transition matrix: F = expm(A*dt)
    F = expm(A * dt);

    % Define return variables
    Full_states = NaN(obj_num, 6, length(tvec)); % Full state vector [x, y, vx, vy, ax, ay] for all objects
    Pos_states = zeros(obj_num, 2, length(tvec)); % Ground Truth position states
    Signal = zeros(128, 128, 2, length(tvec)); % 128x128 signal, 2 channels (unscaled and scaled), for each timestep

    % Limits for Position, Velocity, and Acceleration
    pos_limits = [1, 128; 20, 128];
    vel_limits = [-10, 10; -10, 10]; % make sure velocity is not too high -- they can move but expect fast sample rate
    acc_limits = [-1, 1; -1, 1]; % Super minor acceleration

    % Define initial conditions
    for i = 1:obj_num
        x0(:, i) = [randi(pos_limits(1, :)); randi(pos_limits(2, :)); ...
                        randi(vel_limits(1, :)); randi(vel_limits(2, :)); ...
                        randi(acc_limits(1, :)); randi(acc_limits(2, :))];
    end

    for k = 1:length(tvec)
        % Object binary scene
        S = zeros(128, 128);

        % Update state and place objects in scene
        for c = 1:obj_num
            % Update state
            if k == 1
                X{c} = x0(:, c);
            else % Deterministic update -- no process noise
                X{c} = F * X{c};
            end
            
            % Check if object is within bounds: 0 < posx < 128 and 20 < posy < 128
            posx = X{c}(2); % MATLAB indexing: posx is the second element
            posy = X{c}(1); % MATLAB indexing: posy is the first element
            % disp([posx, posy])
            
            if posx <= 0 || posx >= 128 || posy <= 20 || posy >= 128 || isnan(posx) || isnan(posy)
                % Object out of bounds - mark as deleted
                X{c} = NaN(6, 1);
                Full_states(c, :, k) = NaN(1, 6);
                Pos_states(c, :, k) = NaN(1, 2);
            else
                % Object is valid - store state and place in scene
                Full_states(c, :, k) = X{c};
                Pos_states(c, :, k) = X{c}(1:2); % Store position states
                S(round(posx), round(posy)) = 1; % Place object in scene
            end
        end

        %% Signal Generation
        signal_flat = S';
        signal_flat = signal_flat(:);
        signal_flat = M * signal_flat;
        signal_flat = G' * signal_flat;
        sim_signal = reshape(signal_flat, 128, 128)';
        signal_original = imgaussfilt(sim_signal, 2);

        Signal(:, :, 1, k) = signal_original; % Store unscaled signal for reference

        % Normalize signal [0 1] and apply nonlinearity
        signal_scaled = signal_original; % Use full signal for scaling
        signal_scaled(1:20,:) = NaN; % Focus on valid region
        scaled_signal = tanh(signal_scaled);
        scaled_signal = (signal_scaled - min(signal_scaled(:))) / (max(signal_scaled(:)) - min(signal_scaled(:)));
        
        Signal(:, :, 2, k) = scaled_signal;
    end

end
