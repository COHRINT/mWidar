classdef JPDA_KF < DA_Filter
    %{
    JPDA_KF: Joint Probabilistic Data Association Kalman Filter
    
    Description:
        Implements Joint PDA algorithms with standard KF updates to track multiple objects through clutter.
        Number of objects is assumed known, future implementations could allow for a variable number of objects
    
    Properties:
        x0 - Initial State of each object
        P0 - Initial cov of each object
        F, Q - System dynamics + process noise
        H, R - System measurment model + measurment noise
    Methods:
        JPDA_KF - Constructor
        timestep - process a single timestep for each obj
        prediction - standard kalman prediction for each object
        measurement_update  - JPDA measurement update with beta weighting
        generate_hypothesis - Generate all possible measurment to object hypothesis
        measurment_likelihood - Compute measurment liklihoods for each obj
        compute_joint_probabilities - Compute joint association probabilities
        Data_Association    - Compute JPDA beta coefficients from joint associations
        Validation          - Measurement gating using chi-squared test

    %}
    
    properties
    
        % Initial Conditions
        x0; % Initial State
        P0; % Initial Covariance

        % Sytem Model (Assumed to be the same for each object)
        F % STM
        Q % Process Noise
        R % Meas Noise
        H % Meas Fxn

        % Current Object States
        x_current % Current state estimate [N_x x N_t]
        P_current % Current covariance {N_t}[N_x x N_x] (N_t dimension cell array, N_x x N_x matrix per cell)

        % Intermediate Prediction Variables
        x_predicted % Predicted state [N_x x N_t]
        P_predicted % Predicted cov {N_t}[N_x x N_x]
        z_predicted % Predicted Meas [N_z x N_t]
        S_innovation % Innovation Cov {N_t}[N_z x N_z]

        % JPDA Specific parameters
        PD = 0.95 % Probability of detection
        gate_probability = 0.975 % Validation-gate probability P_G
        lambda_clutter = 2.5 % Expected clutter count per scan over the full measurement region
        measurement_space_area = 16 % Measurement-space area in m^2 (4m x 4m)
        nt % Number of objects

        % Control flags (inherited from DA_Filter)
        debug = false % Enable debug print statements
        DynamicPlot = false % Encable real - time visualization

        % Dynamic Plotting (inherited from DA_Filter) -> Will likely need seperate method for multiOBJ
        dynamic_figure_handle
        association_debug_figure_handle

        % Magnitude Liklihood
        pointlikelihood_mag

        % Cached JPDA debug data for the most recent update
        last_valid_measurements = []
        last_hypotheses = []
        last_joint_probabilities = []
        last_beta = []
        association_debug_top_k = 12

    end

    methods

        function obj = JPDA_KF(x0,P0,F,Q,R,H,t,pointliklihoodmag, varargin)
            % JPDAF Constructor for Joint Probabilistic DA KF
            %
            % SYNTAX:
            %   obj = JPDA_KF(x0,P0,F,Q,R,H)
            %   obj = JPDA_KF(..., 'Debug', true) -> NO dynamic plot functionality
            % INPUTS:
            %   x0 - Initial state estimate
            %   P0 - Initial cov estimate
            %   F - STM for ALL tracks
            %   Q - Process Noise for ALL tracks
            %   R - Meas Noise for ALL tracks
            %   H - Meas model for ALL tracks
            %   t - Number of tracks
            %   varargin - Name-value pairs: 'Debug', t/f
            % OUTPUTS:
            %   obj - JPDA_KF Object

            if nargin < 8
                error('JPDA_KF:InvalidInput', ...
                    'Requires 8 inputs: {x0, P0, F, Q, R, H, t, pointliklihoodmag}')
            end

            % Parse through varargin
            if ~isempty(varargin)
                options = DA_Filter.parseFilterOptions(varargin{:});
                obj.debug = options.Debug;
                obj.DynamicPlot = options.DynamicPlot;
            else
                obj.debug = false;
                obj.DynamicPlot = false;
            end

            % Store Filter Matrices
            obj.x0 = x0;
            obj.P0 = P0;
            obj.F = F;
            obj.Q = Q;
            obj.R = R;
            obj.H = H;

            % Number of tracks
            obj.nt = t;
            % Initialize current state
            obj.x_current = x0;
            obj.P_current = P0;

            % Mag likelihood lookup table
            obj.pointlikelihood_mag = pointliklihoodmag;

            if obj.DynamicPlot
                obj.initializeDynamicPlot('JPDA_KF Real-time Tracking', [100 100 900 700]);
            end

            % Debug
            if obj.debug
                fprintf('\n=== JPDA_KF INITIALIZATION === \n');
                fprintf('State dimention: %d\n',size(x0,1));
                fprintf('Measurement dimension: %d\n', size(H, 1));
                fprintf('Number of object tracks: %d\n',t);
                fprintf('============================\n\n');
            end
        end

        %% ========== TIMESTEP ==========
        function timestep(obj,measurements, signal, varargin)
            if ~obj.validateCommonInputs(measurements)
                error('JPDA_KF:InvalidMeasurements', 'Measurement input validation failed.');
            end

            if obj.debug
                fprintf('\n=== JPDA_KF TIMESTEP START ===\n')
                fprintf('Input: %d measurments \n', size(measurements,2));

                if ~isempty(measurements)

                    for i = 1:size(measurements, 2)
                        fprintf('  Meas %d: [%.3f, %.3f]\n', i, measurements(1, i), measurements(2, i));
                    end

                end

                fprintf('------------------------------\n');
            end

            % Prediction
            obj.prediction();

            % Measurment Update w JPDA
            obj.measurement_update(measurements,signal)

            if obj.DynamicPlot
                if ~isempty(varargin)
                    obj.updateDynamicPlot(measurements, varargin{1});
                else
                    obj.updateDynamicPlot(measurements);
                end
            else
                obj.timestep_counter = obj.timestep_counter + 1;
            end
        end

        %% ========== PREDICTION ==========
        function prediction(obj)
            if obj.debug
                fprintf('[PREDICTION] Applynig Kalman Dynamics...\n')
            end

            for t = 1:obj.nt
                
                % State Prediction
                obj.x_predicted(:,t) = obj.F * obj.x_current(:,t);
                
                % Measurment Prediction
                obj.z_predicted(:,t) = obj.H * obj.x_predicted(:,t);

                % Cov Prediction
                obj.P_predicted{t} = obj.F * obj.P_current{t} * obj.F' + obj.Q;

                % Innovation cov
                obj.S_innovation{t} = obj.H * obj.P_predicted{t} * obj.H' + obj.R;

                if obj.debug
                    fprintf('[PREDICTION] Predicted state for track %d: [%.4f, %.4f]\n',t,obj.x_predicted(1,t), obj.x_predicted(2,t))
                end
            end

            if obj.debug
                fprintf('[PREDICTION] Complete \n')
            end

        end

        %% ========== VALIDATION ==========
        function [valid_z, meas, validation_matrix, valid_meas_idx] = Validation(obj,z)
            meas = true(1, obj.nt); %default
            gamma = chi2inv(1-obj.gate_probability, size(obj.H, 1));
            valid_z = [];
            valid_meas_idx = [];
            validation_matrix_full = false(obj.nt, size(z, 2));

            if obj.debug
                fprintf('[VALIDATION] Processing %d measurments...\n',size(z,2))
            end
            
            for t = 1:obj.nt
                DETECTED = false; % Default -> flag to check if a given object has at least one measurment pass through the gate
                for j = 1:size(z,2)
                    detection = z(:,j);
                    Nu = (detection - obj.z_predicted(:,t))' /obj.S_innovation{t} *  (detection - obj.z_predicted(:,t));

                    if Nu < gamma
                        validation_matrix_full(t, j) = true;
                        DETECTED = true;
                        if obj.debug
                            fprintf('  Track %d, Meas %d: VALID (NIS=%.3f)\n', t, j, Nu);
                        end
                    else
                        if obj.debug
                            fprintf('  Track %d, Meas %d: REJECTED (NIS=%.3f > %.3f)\n', t, j, Nu, gamma);

                        end
                    end

                end

                if ~DETECTED
                    meas(t) = false;

                    if obj.debug
                        fprintf('   No detections passed validation ellipse for object %d \n',t)
                    end
                end

            end

            valid_meas_idx = find(any(validation_matrix_full, 1));
            valid_z = z(:, valid_meas_idx);
            validation_matrix = validation_matrix_full(:, valid_meas_idx);

        end

        %% ========== GENERATE_HYPOTHESES ==========
        function hypotheses = generate_hypotheses(obj,z,validation_matrix)
            m = size(z,2);

            idx = 0:m; % all possible measurment idx
            [grid_combo{1:obj.nt}] = ndgrid(idx); % t-dimensional grid of all possible measurment to track combos
            combo = cellfun(@(x) x(:), grid_combo, 'UniformOutput', false);
            all = [combo{:}]; % Flatten
            
            % Trim hypotheses to valid ones only
            valid = arrayfun(@(i) obj.isValidAssignment(all(i,:), validation_matrix), 1:size(all,1));
            hypotheses = all(valid, :); % n_hyp given by (m choose t)*2 + 2m+1 ? -> CHECK MATH ON THIS
        end

        %% ========== ISVALIDASSIGNMENT ==========
        function valid = isValidAssignment(obj,assign,validation_matrix)
            meas_used = assign(assign > 0);
            valid = length(meas_used) == length(unique(meas_used)); % Must be unique (can't have one measurment associate to two tracks, or vice versa)

            if ~valid
                return;
            end

            for t = 1:obj.nt
                meas_idx = assign(t);
                if meas_idx > 0 && ~validation_matrix(t, meas_idx)
                    valid = false;
                    return;
                end
            end
        end

        %% ========== MEASURMENT_LIKELIHOOD ==========
        function L = measurment_likelihood(obj,z,z_mag)
            L = zeros(obj.nt,size(z,2));

            % Smush z_mag into a col vector
            z_mag = z_mag(:);

            % Define spatial grid parameters (should match precomputed lookup table)
            npx = 128;
            xgrid = linspace(-2, 2, npx);
            ygrid = linspace(0, 4, npx);

            for t = 1:obj.nt
                for j = 1:size(z,2)

                    % Get index in lookup table
                    % Find measurement linear index
                    [~, meas_x_idx] = min(abs(xgrid - z(1,j)));
                    [~, meas_y_idx] = min(abs(ygrid - z(2,j)));
                    meas_linear_idx = sub2ind([npx, npx], meas_y_idx, meas_x_idx);
                    magnitude_likelihood = 1;

                    if ~isempty(obj.pointlikelihood_mag)
                        mag_liklihood_values = obj.pointlikelihood_mag(meas_linear_idx, :);
                        if numel(z_mag) == size(z, 2)
                            z_mag_value = z_mag(j);
                        elseif numel(z_mag) >= meas_linear_idx
                            z_mag_value = z_mag(meas_linear_idx);
                        else
                            z_mag_value = [];
                        end

                        if ~isempty(z_mag_value)
                            sigma_mag = max(mag_liklihood_values(2), eps);
                            magnitude_likelihood = normpdf(z_mag_value, mag_liklihood_values(1), sigma_mag);
                        end

                    end
                    L(t,j) = mvnpdf(z(:,j),obj.z_predicted(:,t),obj.S_innovation{t}) * magnitude_likelihood;

                    if obj.debug
                        fprintf('[LIKELIHOOD] Meas %d, likelihood to be associatiated with track %d: %.6f\n',j,t,L(t,j))
                    end
                end
            end
        end

        %% ========== COMPUTE_JOINT_PROBABILITIES ==========
        function joint_probs = compute_joint_probabilities(obj,hypotheses,L)
            
            n_hyp = size(hypotheses,1);
            joint_probs = zeros(n_hyp,1);
            clutter_density = obj.getClutterDensity();
            miss_weight = max(1 - obj.PD * obj.gate_probability, eps);

            for h = 1:n_hyp
                theta = hypotheses(h,:); % Current hypothesis

                joint_prob = 1;

                for t = 1:obj.nt
                    j = theta(t); % Detection j, assigned to track t

                    if j == 0 % Missed detecting track t
                        joint_prob = joint_prob * miss_weight;
                    else % detected track t (tau = 1)
                        joint_prob = joint_prob * (clutter_density)^(-1) * L(t,j) * obj.PD;
                    end
                end
                joint_probs(h) = joint_prob;
            end

            joint_probs = joint_probs / max(sum(joint_probs), eps);
            obj.last_hypotheses = hypotheses;
            obj.last_joint_probabilities = joint_probs;

            if obj.debug
                fprintf('[JOINT ASSOCIATION PROBABILITIES] Clutter density used: %.6f (rate %.3f / area %.3f)\n', ...
                    clutter_density, obj.lambda_clutter, obj.measurement_space_area);
                fprintf('[JOINT ASSOCIATION PROBABILITIES] Miss weight used: %.6f = 1 - PD * PG\n', miss_weight);
                % for h = 1:n_hyp
                %    fprintf('[JOINT ASSOCIATION PROBABILITIES] Probability of hypotheses %d: %.3f\n',h,joint_probs(h))
                %    fprintf('[JOINT ASSOCIATION PROBABILITIES] hypotheses %d: [',h)
                %    fprintf('%d',hypotheses(h,:))
                %    fprintf('] \n')
                % end
                fprintf('[JOINT ASSOCIATION PROBABILITIES] Sum check: %0.6f\n',sum(joint_probs))
            end
        end

        %% ========== DATA_ASSOCIATION ==========
        %{
            Computation of marginal beta probabilities
        %}
        function beta = Data_Association(obj,hypotheses,joint_probs,z)
            m = size(z,2);
            beta = zeros(obj.nt,m+1);
            n_hyp = size(hypotheses,1);

            for h = 1:n_hyp
                theta = hypotheses(h,:);
                for t = 1:obj.nt
                    j = theta(t);
                    if j == 0 % missed detection (beta0)
                        beta(t,m+1) = beta(t,m+1) + joint_probs(h);
                    else % detected
                        beta(t,j) = beta(t,j) + joint_probs(h);
                    end
                end
            end

            obj.last_beta = beta;

            if obj.debug
                obj.visualizeBetaProbabilities(beta, z);
                fprintf('[DATA ASSOCIATION] Beta coefficients: [');
                fprintf('%.3f ', beta);
                fprintf('], LAST VALUE IS BETA0\n');
                for t = 1:obj.nt
                    fprintf('[DATA ASSOCIATION] Sum check for object %d: %.6f\n', t,sum(beta(t,:)));
                end
            end
        end

        %% ========== MEASURMENT_UPDATE ==========
        function measurement_update(obj,z,signal)

            if obj.debug
                fprintf('[MEASURMENT UPDATE] Starting JPDA update...\n')
            end

            % Gate around predicted measurment
            [valid_z, meas, validation_matrix] = obj.Validation(z);

            if isempty(valid_z) % No valid measurments
                if obj.debug
                    fprintf('[MEASUREMENT UPDATE] No measurements - missed detection case\n');
                end

                % Keep prediction as final estimate for missed detection
                obj.x_current = obj.x_predicted;
                obj.P_current = obj.P_predicted;
                return;
            end

            for t = 1:obj.nt
                if ~meas(t)
                    if obj.debug
                        fprintf('[MEASURMENT UPDATE] object %d did not have any measurments pass through its validation ellipse ...\n',t)
                        % Not sure, use kalman prediction here?
                        %fprintf('[MEASURMENT UPDATE] Object %d will use kalman prediction, no measurment update for object %d \n\n',t,t)
                    end
                    %obj.x_current(:,t) = obj.x_predicted(:,t);
                    %obj.P_predicted{t} = obj.P_predicted{t};
                end
            end

            hypotheses = obj.generate_hypotheses(valid_z, validation_matrix);
            L = obj.measurment_likelihood(valid_z,signal);
            obj.last_valid_measurements = valid_z;

            joint_probs = obj.compute_joint_probabilities(hypotheses,L);

            beta = obj.Data_Association(hypotheses,joint_probs,valid_z);

            m = size(valid_z,2);
            n_z = size(obj.H, 1);

            for t = 1:obj.nt
                beta0 = beta(t,m+1);

                nu = zeros(n_z,m);
                innov = zeros(n_z,1);

                for j = 1:m
                    nu(:,j) = valid_z(:,j) - obj.z_predicted(:,t);
                    innov = innov + beta(t,j) * nu(:,j);
                end

                KK = obj.P_predicted{t} * obj.H' / obj.S_innovation{t};
                Pc = obj.P_predicted{t} - KK * obj.S_innovation{t} * KK';

                temp = 0;

                for j = 1:size(valid_z, 2)
                    temp = temp + (beta(t,j) * (nu(:, j) * nu(:, j)'));
                end

                P_tilde = KK* (temp - innov*innov') * KK';

                if obj.debug
                    fprintf('[MEASUREMENT UPDATE] Using %d measurements with beta weights for object %d \n', size(valid_z, 2),t);
                    fprintf('[MEASUREMENT UPDATE] Innovation norm: %.6f\n', norm(innov));
                end

                obj.x_current(:,t) = obj.x_predicted(:,t) + KK*innov;
                obj.P_current{t} = beta0 * obj.P_predicted{t} + (1 - beta0)*Pc + P_tilde;

                if obj.debug
                    fprintf('[MEASUREMENT UPDATE] Completed for object %d. State: [%.4f, %.4f]\n', t, obj.x_current(1,t), obj.x_current(2,t));
                end
            end

        end

        %% ========== STATE ESTIMATION ==========
        function [X_est,P_est] = getGaussianEstimate(obj)
            X_est = obj.x_current;
            P_est = obj.P_current;
            if obj.debug
                for t = 1:obj.nt
                    fprintf('[GAUSSIAN EST] Object %d, State=[%.4f, %.4f], det(P)=%.8f\n',...
                        t, X_est(1,t), X_est(2,t), det(P_est{t}));
                end
            end
        end

        function storeHistory(obj, measurements, varargin)
            if nargin > 2
                true_state = varargin{1};
            else
                true_state = [];
            end

            [x_est, P_est] = obj.getGaussianEstimate();

            entry = struct( ...
                'x_est', x_est, ...
                'P_est', P_est, ...
                'measurements', measurements, ...
                'true_state', true_state, ...
                'timestep_num', obj.getTimestepCount());

            if obj.store_full_history
                entry.x_predicted = obj.x_predicted;
                entry.P_predicted = obj.P_predicted;
                entry.z_predicted = obj.z_predicted;
                entry.S_innovation = obj.S_innovation;
            end

            if isempty(obj.history)
                obj.history = entry;
            else
                obj.history(end + 1) = entry;
            end
        end

        %% ========== VISUALIZATION ==========
        function visualize(obj,varargin)
            % Visualize the current filter state estimate
            %
            % SYNTAX:
            %   obj.visualize()
            %   obj.visualize(figure_handle, title_str)
            %   obj.visualize(figure_handle, title_str, measurements, true_state)
            %
            % INPUTS:
            %   figure_handle - (optional) Figure handle to plot in
            %   title_str     - (optional) Title string for plot
            %   measurements  - (optional) Current measurements [N_z x N_meas]
            %   true_state    - (optional) True state for comparison [N_x x N_t]
            %
            % DESCRIPTION:
            %   Creates visualization of Kalman filter state including state
            %   estimate, covariance ellipse, measurements, and true state.

            % Parse input arguments
            if nargin > 1 && ~isempty(varargin{1})
                figure(varargin{1});
            else
                figure;
            end

            if nargin > 2
                title_str = varargin{2};
            else
                title_str = 'JPDA-KF State Estimate';
            end

            if nargin > 3
                measurements = varargin{3};
            else
                measurements = [];
            end

            if nargin > 4
                true_state = varargin{4};
            else
                true_state = [];
            end

            cla; hold on;

            for t = 1:obj.nt
                % Plot State Estimate
                plot(obj.x_current(1,t),obj.x_current(2,t), 'ro', 'MarkerSize', 10, 'LineWidth', 3, ...
                    'DisplayName', 'KF Estimate');
                
                % Plot 1-sigma and 2-sigma cov ellipses
                if size(obj.P_current{t},1) >= 2
                    theta = linspace(0, 2 * pi, 100);
                    pos_cov = obj.P_current{t}(1:2,1:2);

                     % Eigenvalue decomposition for proper ellipse orientation
                    [V, D] = eig(pos_cov);

                    % 1-sigma ellipse (68% confidence)
                    sigma1_scale = sqrt(chi2inv(0.68, 2));
                    a1 = sigma1_scale * sqrt(D(1, 1));
                    b1 = sigma1_scale * sqrt(D(2, 2));
                    ellipse1_local = [a1 * cos(theta); b1 * sin(theta)];
                    ellipse1_global = V * ellipse1_local + obj.x_current(1:2,t);
                    plot(ellipse1_global(1, :), ellipse1_global(2, :), 'r-', 'LineWidth', 2, ...
                        'DisplayName', '1σ Covariance');

                    % 2-sigma ellipse (95% confidence)
                    sigma2_scale = sqrt(chi2inv(0.95, 2));
                    a2 = sigma2_scale * sqrt(D(1, 1));
                    b2 = sigma2_scale * sqrt(D(2, 2));
                    ellipse2_local = [a2 * cos(theta); b2 * sin(theta)];
                    ellipse2_global = V * ellipse2_local + obj.x_current(1:2,t);
                    plot(ellipse2_global(1, :), ellipse2_global(2, :), 'r--', 'LineWidth', 1.5, ...
                        'DisplayName', '2σ Covariance');
                end
                
                % Plot true state if provided
                if ~isempty(true_state)
                    plot(true_state(1,t), true_state(2,t), 'd', 'Color', 'm', ...
                        'MarkerSize', 8, 'LineWidth', 2, 'MarkerFaceColor', 'm', ...
                        'DisplayName', 'True Position');
                end
                
            end

            % Plot measurments
            if ~isempty(measurements)
                plot(measurements(1, :), measurements(2, :), '+', 'Color', [1 0.5 0], ...
                    'MarkerSize', 10, 'LineWidth', 3, 'DisplayName', 'Measurements');
            end

            % Formatting
            xlabel('X Position (m)');
            ylabel('Y Position (m)');
            title(title_str, 'Interpreter', 'latex');
            legend('Location', 'best');
            grid on;
            axis equal;
        end

        function visualizeBetaProbabilities(obj, beta, z)
            if isempty(beta)
                return;
            end

            n_tracks = size(beta, 1);
            n_meas = size(beta, 2) - 1;
            column_labels = [arrayfun(@(j) sprintf('M%d', j), 1:n_meas, 'UniformOutput', false), {'miss'}];

            if isempty(obj.association_debug_figure_handle) || ~isgraphics(obj.association_debug_figure_handle)
                obj.association_debug_figure_handle = figure( ...
                    'Name', 'JPDA Association Debugger', ...
                    'NumberTitle', 'off', ...
                    'Color', 'w');
            else
                figure(obj.association_debug_figure_handle);
            end

            clf(obj.association_debug_figure_handle);
            tl = tiledlayout(obj.association_debug_figure_handle, n_tracks + 1, 1, ...
                'TileSpacing', 'compact', 'Padding', 'compact');

            max_bar_value = max(beta(:));
            if max_bar_value <= 0
                max_bar_value = 1;
            end
            x_limit = min(1.0, 1.10 * max_bar_value);

            for t = 1:n_tracks
                ax_track = nexttile(tl, t);
                bar_values = beta(t, :);
                bar_handle = bar(ax_track, 1:(n_meas + 1), bar_values, ...
                    'FaceColor', [0.25 0.55 0.85], ...
                    'EdgeColor', [0.10 0.10 0.10], ...
                    'LineWidth', 1.0);
                bar_handle.FaceColor = 'flat';

                color_data = repmat([0.72 0.78 0.86], n_meas + 1, 1);
                [~, best_idx] = max(bar_values);
                color_data(best_idx, :) = [0.15 0.55 0.25];
                bar_handle.CData = color_data;

                ax_track.XTick = 1:(n_meas + 1);
                ax_track.XTickLabel = column_labels;
                ax_track.YLim = [0, x_limit];
                ax_track.YGrid = 'on';
                ax_track.Box = 'on';
                ylabel(ax_track, '\beta');

                if t == 1
                    title(ax_track, 'Marginal Association Probabilities (\beta) by Track');
                end

                xlabel(ax_track, sprintf('Track %d', t));
            end

            ax_table = nexttile(tl, n_tracks + 1);
            axis(ax_table, 'off');

            table_data = obj.buildBetaTableData(beta, z);

            uitable(obj.association_debug_figure_handle, ...
                'Data', table_data, ...
                'ColumnName', {'Track', 'Best Assoc.', 'Best \beta', '\beta_{miss}', 'Row Sum'}, ...
                'Units', 'normalized', ...
                'Position', ax_table.Position, ...
                'ColumnWidth', 'auto', ...
                'RowName', []);

            title(tl, sprintf('JPDA Beta Association Debugger | Step %d', obj.timestep_counter + 1), ...
                'FontWeight', 'bold');
            drawnow limitrate;
        end
    end

    methods (Access = private)
        function clutter_density = getClutterDensity(obj)
            clutter_density = obj.lambda_clutter / obj.measurement_space_area;
            clutter_density = max(clutter_density, eps);
        end

        function table_data = buildBetaTableData(obj, beta, z)
            n_tracks = size(beta, 1);
            n_meas = size(beta, 2) - 1;
            table_data = cell(n_tracks, 5);

            for t = 1:n_tracks
                if n_meas > 0
                    [best_beta, best_idx] = max(beta(t, 1:n_meas));
                    if ~isempty(z) && best_idx <= size(z, 2)
                        best_label = sprintf('M%d [%.3f, %.3f]', best_idx, z(1, best_idx), z(2, best_idx));
                    else
                        best_label = sprintf('M%d', best_idx);
                    end
                else
                    best_beta = 0;
                    best_label = 'none';
                end

                table_data{t, 1} = sprintf('Track %d', t);
                table_data{t, 2} = best_label;
                table_data{t, 3} = sprintf('%.6f', best_beta);
                table_data{t, 4} = sprintf('%.6f', beta(t, n_meas + 1));
                table_data{t, 5} = sprintf('%.6f', sum(beta(t, :)));
            end
        end
    end


end
