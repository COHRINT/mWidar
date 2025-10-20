classdef PDA_KF < DA_Filter
    % PDA_KF Probabilistic Data Association Kalman Filter
    %
    % DESCRIPTION:
    %   Implements PDA data association with standard Kalman filtering for
    %   single target tracking under measurement uncertainty. Handles multiple
    %   measurements per timestep using probabilistic association weights.
    %
    % PROPERTIES:
    %   x0, P0    - Initial state and covariance
    %   F, Q      - State transition and process noise matrices
    %   R, H      - Measurement noise and observation matrices
    %
    % METHODS:
    %   PDA_KF              - Constructor
    %   timestep            - Process single time step with PDA-KF algorithm
    %   prediction          - Standard Kalman prediction step
    %   measurement_update  - PDA measurement update with beta weighting
    %   Data_Association    - Compute PDA beta coefficients
    %   Validation          - Measurement gating using chi-squared test
    %
    % EXAMPLE:
    %   kf = PDA_KF(x0, P0, F, Q, R, H);
    %   [x_est, P_est] = kf.timestep(x_prev, P_prev, measurements);
    %
    % See also DA_Filter, GNN_KF, PDA_PF

    properties
        % Initial Conditions
        x0 % Initial State
        P0 % Initial Covariance

        % System Model
        F % Dynamics Matrix
        Q % Process Noise Covariance
        R % Measurement Noise Covariance
        H % Measurement Function Matrix

        % Internal State (for consistent API)
        x_current % Current state estimate [N_x x 1]
        P_current % Current covariance estimate [N_x x N_x]

        % Intermediate Computation Variables
        x_predicted % Predicted state [N_x x 1]
        P_predicted % Predicted covariance [N_x x N_x]
        z_predicted % Predicted measurement [N_z x 1]
        S_innovation % Innovation covariance [N_z x N_z]

        % PDA Parameters
        PD = 0.95 % Probability of detection
        PG = 0.95 % Gate probability
        lambda_clutter = 2.5 % Clutter density parameter

        % Control Flags (inherited from DA_Filter)
        debug = false % Enable debug output
        DynamicPlot = false % Enable real-time visualization

        % Dynamic Plotting (inherited from DA_Filter)
        dynamic_figure_handle % Figure handle for dynamic plotting

        % Magnitude Liklihood
        pointlikelihood_mag

        % Detection and Magnitude Liklihood storage for visualization
        detection_liklihood
        magnitude_liklihood
        composite_liklihood
        valid_z

    end

    methods

        %% ========== CONSTRUCTOR ==========
        function obj = PDA_KF(x0, P0, F, Q, R, H, pointliklihoodmag, varargin)
            % PDA_KF Constructor for Probabilistic Data Association Kalman Filter
            %
            % SYNTAX:
            %   obj = PDA_KF(x0, P0, F, Q, R, H)
            %   obj = PDA_KF(..., 'Debug', true, 'DynamicPlot', true)
            %
            % INPUTS:
            %   x0 - Initial state estimate [N_x x 1]
            %   P0 - Initial covariance matrix [N_x x N_x]
            %   F  - State transition matrix [N_x x N_x]
            %   Q  - Process noise covariance [N_x x N_x]
            %   R  - Measurement noise covariance [N_z x N_z]
            %   H  - Measurement matrix [N_z x N_x]
            %   varargin - Name-value pairs: 'Debug', true/false, 'DynamicPlot', true/false
            %
            % OUTPUTS:
            %   obj - Initialized PDA_KF object

            if nargin < 6
                error('PDA_KF:InvalidInput', 'Requires 6 inputs: {x0, P0, F, Q, R, H}');
            end

            % Parse options using parent class utility
            if nargin > 6
                options = DA_Filter.parseFilterOptions(varargin{:});
                obj.debug = options.Debug;
                obj.DynamicPlot = options.DynamicPlot;
            end

            % Store filter matrices
            obj.x0 = x0;
            obj.P0 = P0;
            obj.F = F;
            obj.Q = Q;
            obj.R = R;
            obj.H = H;

            % Initialize internal state
            obj.x_current = x0;
            obj.P_current = P0;
            
            obj.pointlikelihood_mag = pointliklihoodmag;

            if obj.debug
                fprintf('\n=== PDA_KF INITIALIZATION ===\n');
                fprintf('State dimension: %d\n', length(x0));
                fprintf('Measurement dimension: %d\n', size(H, 1));
                fprintf('Initial state: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]\n', x0);
                fprintf('============================\n\n');
            end

            % Initialize dynamic plotting if enabled
            obj.initializeDynamicPlot('PDA-KF Dynamic Tracking', [100, 100, 800, 600]);
        end

        %% ========== TIMESTEP ==========
        function timestep(obj, measurements, signal, varargin)
            % TIMESTEP Process single time step with PDA-KF algorithm
            %
            % SYNTAX:
            %   obj.timestep(measurements)
            %   obj.timestep(measurements, true_state)
            %
            % INPUTS:
            %   measurements - Current measurements [N_z x N_measurements]
            %   true_state   - (optional) True state for visualization
            %
            % DESCRIPTION:
            %   Implements complete PDA-KF algorithm:
            %   1. Prediction step using Kalman dynamics
            %   2. Measurement update with PDA data association
            %   Use getGaussianEstimate() to extract state estimates after timestep.
            %
            % See also prediction, measurement_update, Data_Association

            if obj.debug
                fprintf('\n=== PDA-KF TIMESTEP START ===\n');
                fprintf('Input: %d measurements\n', size(measurements, 2));

                if ~isempty(measurements)

                    for i = 1:size(measurements, 2)
                        fprintf('  Meas %d: [%.3f, %.3f]\n', i, measurements(1, i), measurements(2, i));
                    end

                end

                fprintf('------------------------------\n');
            end

            % Step 1: Prediction step
            obj.prediction();

            % Step 2: Measurement update with PDA
            obj.measurement_update(measurements,signal);

            if obj.debug
                [x_est, P_est] = obj.getGaussianEstimate();
                fprintf('\nOutput: State estimate [%.4f, %.4f] m\n', x_est(1), x_est(2));
                fprintf('        Covariance trace: %.6f\n', trace(P_est));
                fprintf('=== PDA-KF TIMESTEP END ===\n\n');
            end

            % Update dynamic plot if enabled
            if obj.DynamicPlot

                if nargin > 2
                    true_state = varargin{1};
                    obj.updateDynamicPlot(measurements, true_state);
                else
                    obj.updateDynamicPlot(measurements);
                end

            end

            obj.visualize(1,'PDA-KF State Estimate',[],signal)

   

        end

        %% ========== PREDICTION STEP ==========
        function prediction(obj)
            % PREDICTION Standard Kalman filter prediction step
            %
            % SYNTAX:
            %   obj.prediction()
            %
            % DESCRIPTION:
            %   Implements standard Kalman filter prediction equations using internal state:
            %   x_predicted = F * x_current
            %   P_predicted = F * P_current * F' + Q
            %   z_predicted = H * x_predicted
            %   S_innovation = H * P_predicted * H' + R
            %
            % MODIFIES:
            %   obj.x_predicted - Predicted state [N_x x 1]
            %   obj.P_predicted - Predicted covariance [N_x x N_x]
            %   obj.z_predicted - Predicted measurement [N_z x 1]
            %   obj.S_innovation - Innovation covariance [N_z x N_z]    
            %
            % See also timestep, measurement_update

            if obj.debug
                fprintf('[PREDICTION] Applying Kalman dynamics...\n');
            end

            % State prediction using current internal state
            obj.x_predicted = obj.F * obj.x_current;

            % Predicted measurement
            obj.z_predicted = obj.H * obj.x_predicted;

            % Covariance prediction
            obj.P_predicted = obj.F * obj.P_current * obj.F' + obj.Q;

            % Innovation covariance
            obj.S_innovation = obj.H * obj.P_predicted * obj.H' + obj.R;

            if obj.debug
                fprintf('[PREDICTION] Predicted state: [%.4f, %.4f]\n', obj.x_predicted(1), obj.x_predicted(2));
                fprintf('[PREDICTION] Complete.\n');
            end

        end

        %% ========== MEASUREMENT VALIDATION ==========
        function [meas] = Validation(obj, z)
            % VALIDATION Gate measurements using chi-squared validation ellipse
            %
            % SYNTAX:
            %   [valid_z, meas] = obj.Validation(z)
            %
            % INPUTS:
            %   z     - Raw measurements [N_z x N_measurements]
            %
            % OUTPUTS:
            %   valid_z - Validated measurements [N_z x N_valid]
            %   meas    - Boolean flag indicating if measurements are present
            %
            % DESCRIPTION:
            %   Applies chi-squared gating to filter out unlikely measurements.
            %   Uses 95% confidence level for validation ellipse.
            %   Uses internal predicted measurement and innovation covariance.
            %
            % See also timestep, Data_Association

            meas = true; % default
            gamma = chi2inv(0.95, 2); % 95 % confidence threshold
            obj.valid_z = [];

            if obj.debug
                fprintf('[VALIDATION] Processing %d measurements...\n', size(z, 2));
            end

            for j = 1:size(z, 2)
                detection = z(:, j);
                Nu = (detection - obj.z_predicted)' / obj.S_innovation * (detection - obj.z_predicted); % NIS statistic

                if Nu < gamma % Validation gate
                    obj.valid_z = [obj.valid_z detection]; % Append validated measurement

                    if obj.debug
                        fprintf('  Meas %d: VALID (NIS=%.3f)\n', j, Nu);
                    end

                else

                    if obj.debug
                        fprintf('  Meas %d: REJECTED (NIS=%.3f > %.3f)\n', j, Nu, gamma);
                    end

                end

            end

            if isempty(obj.valid_z)

                if obj.debug
                    fprintf('[VALIDATION] No valid measurements - missed detection\n');
                end

                meas = false; % missed detection
            else

                if obj.debug
                    fprintf('[VALIDATION] %d/%d measurements validated\n', size(obj.valid_z, 2), size(z, 2));
                end

            end

        end
        %% ========= Likelihood Calculation ===========
        function [L] = PDA_Likelihoods(obj,z_mag)
            % PDA_Likelihoods Compute measurment and magnitude likelihoods
            %
            % SYNTAX: [L] = obj.PDA_Likelihoods(z)
            %
            % INPUTS:
            %   z - Validated measurments [N_z x N_valid]
            %   z_mag - Raw signal for magnitude lookup
            %
            % OUTPUTS:
            %   L - Likelihood for given measurment and magnitude of signal

            if obj.debug
                fprintf('[LIKLIHOOD CALCULATION]')
            end
            
            % Define spatial grid parameters (should match precomputed lookup table)
            npx = 128;
            xgrid = linspace(-2, 2, npx);
            ygrid = linspace(0, 4, npx);
           

            % PDA parameters
            lambda = obj.lambda_clutter;
            oPD = obj.PD;

            % Pre-allocate space
            L = zeros(1, size(obj.valid_z, 2));
            
            % Smush z_mag into a col vector
            z_mag = z_mag(:);

            % Compute likelihood of each validated measurement
            for j = 1:size(obj.valid_z, 2)
                detection_likelihood = (mvnpdf(obj.valid_z(:, j), obj.z_predicted, obj.S_innovation) * oPD) / lambda;

                % Magnitude likelihood

                % Get index in lookup table
                % Find measurement linear index
                [~, meas_x_idx] = min(abs(xgrid - obj.valid_z(1,j)));
                [~, meas_y_idx] = min(abs(ygrid - obj.valid_z(2,j)));
                meas_linear_idx = sub2ind([npx, npx], meas_y_idx, meas_x_idx);

                mag_liklihood_values = obj.pointlikelihood_mag(meas_linear_idx, :);
                magnitude_likelihood = normpdf(z_mag(meas_linear_idx),mag_liklihood_values(1),mag_liklihood_values(2)); % Weighted slightly

               
                
                obj.detection_liklihood(j) = detection_likelihood;
                obj.magnitude_liklihood(j) = magnitude_likelihood;

                L(j) = detection_likelihood * magnitude_likelihood;
                
                obj.composite_liklihood(j) = L(j);

                if obj.debug
                    fprintf('  Meas %d: likelihood=%.6f\n', j, L(j));
                end

            end
            
            %L = sum(likelihood, 2); % sum of all likelihoods
        end
        %% ========== PROBABILISTIC DATA ASSOCIATION ==========
        function [beta, beta0] = Data_Association(obj,signal)
            % DATA_ASSOCIATION Compute PDA association probabilities
            %
            % SYNTAX:
            %   [beta, beta0] = obj.Data_Association(valid_z)
            %
            % INPUTS:
            %   valid_z - Validated measurements [N_z x N_valid]
            %
            % OUTPUTS:
            %   beta  - Association probabilities for each measurement [1 x N_valid]
            %   beta0 - Probability that all measurements are clutter
            %
            % DESCRIPTION:
            %   Computes PDA beta coefficients using likelihood ratios.
            %   Uses internal predicted measurement and innovation covariance.
            %   Each beta(i) represents the probability that measurement i
            %   is target-originated. beta0 is the probability that no
            %   measurements are target-originated (all clutter).
            %
            % See also timestep, measurement_update, Validation

            if obj.debug
                fprintf('[DATA ASSOCIATION] Computing beta coefficients...\n');
            end

            oPD = obj.PD;
            oPG = obj.PG;

            beta = zeros(1, size(obj.valid_z, 2));

            [likelihood] = PDA_Likelihoods(obj,signal);
            sum_likelihood = sum(likelihood, 2);

            % Compute beta coefficients
            for j = 1:size(obj.valid_z, 2)
                beta(j) = likelihood(j) / (1 - oPD * oPG + sum_likelihood);
            end

            % Compute beta0 (clutter hypothesis probability)
            beta0 = (1 - oPD * oPG) / (1 - oPD * oPG + sum_likelihood);

            if obj.debug
                fprintf('[DATA ASSOCIATION] Beta coefficients: [');
                fprintf('%.3f ', beta);
                fprintf('], beta0=%.3f\n', beta0);
                fprintf('[DATA ASSOCIATION] Sum check: %.6f\n', sum(beta) + beta0);
            end

        end

        %% ========== MEASUREMENT UPDATE ==========
        function measurement_update(obj, measurements,signal)
            % MEASUREMENT_UPDATE PDA measurement update step (standardized interface)
            %
            % SYNTAX:
            %   obj.measurement_update(measurements)
            %
            % INPUTS:
            %   measurements - Current measurements [N_z x N_measurements]
            %
            % DESCRIPTION:
            %   Implements PDA measurement update using internal state:
            %   1. Measurement validation (gating)
            %   2. Data association (beta coefficient computation)
            %   3. PDA measurement update with weighted innovation
            %   Updates internal state obj.x_current and obj.P_current
            %
            % MODIFIES:
            %   obj.x_current - Updated state estimate [N_x x 1]
            %   obj.P_current - Updated covariance estimate [N_x x N_x]
            %
            % See also timestep, prediction, Data_Association

            if obj.debug
                fprintf('[MEASUREMENT UPDATE] Starting PDA update...\n');
            end

            % Handle empty measurements case
            if isempty(measurements)

                if obj.debug
                    fprintf('[MEASUREMENT UPDATE] No measurements - missed detection case\n');
                end

                % Keep prediction as final estimate for missed detection
                obj.x_current = obj.x_predicted;
                obj.P_current = obj.P_predicted;
                return;
            end

            % Step 1: Measurement validation (gating)
            [ meas] = obj.Validation(measurements);

            % Step 2: Data association (compute beta coefficients)
            if meas
                [beta, beta0] = obj.Data_Association(signal);
            else
                % Missed detection case
                if obj.debug
                    fprintf('[MEASUREMENT UPDATE] No valid measurements - missed detection\n');
                end

                obj.x_current = obj.x_predicted;
                obj.P_current = obj.P_predicted;
                return;
            end

            % Step 3: PDA measurement update
            KK = 0;
            innov = 0;
            Pc = 0;
            P_tilde = 0;

            if meas == false % Missed detection case (redundant check but kept for clarity)
                innov = zeros(size(obj.z_predicted, 1), 1);
                beta0 = 1;
                KK = (obj.P_predicted * obj.H') / (obj.H * obj.P_predicted * obj.H' + obj.R);
                Pc = obj.P_predicted - KK * obj.S_innovation * KK';
                P_tilde = 0;

                if obj.debug
                    fprintf('[MEASUREMENT UPDATE] Missed detection case (beta0=1)\n');
                end

            else % Measurements available
                nu = zeros(size(obj.z_predicted, 1), size(obj.valid_z, 2));
                innov = zeros(size(obj.z_predicted, 1), 1);

                % Compute weighted innovation
                for j = 1:size(obj.valid_z, 2)
                    nu(:, j) = obj.valid_z(:, j) - obj.z_predicted;
                    innov = innov + beta(j) * nu(:, j); % Weighted sum of innovations
                end

                % Standard Kalman gain
                KK = obj.P_predicted * obj.H' / obj.S_innovation;

                % Updated covariance components
                Pc = obj.P_predicted - KK * obj.S_innovation * KK';

                % Spreading of innovations term
                temp = 0;

                for j = 1:size(obj.valid_z, 2)
                    temp = temp + (beta(j) * (nu(:, j) * nu(:, j)'));
                end

                P_tilde = KK * (temp - innov * innov') * KK';

                if obj.debug
                    fprintf('[MEASUREMENT UPDATE] Using %d measurements with beta weights\n', size(obj.valid_z, 2));
                    fprintf('[MEASUREMENT UPDATE] Innovation norm: %.6f\n', norm(innov));
                end

            end

            % Final PDA update equations - update internal state
            obj.x_current = obj.x_predicted + KK * innov;
            obj.P_current = beta0 * obj.P_predicted + (1 - beta0) * Pc + P_tilde;

            if obj.debug
                fprintf('[MEASUREMENT UPDATE] Complete. State: [%.4f, %.4f]\n', obj.x_current(1), obj.x_current(2));
            end

        end

        %% ========== STATE ESTIMATION ==========
        function [x_est, P_est] = getGaussianEstimate(obj)
            % GETGAUSSIANESTIMATE Extract Gaussian state estimate (DA_Filter interface)
            %
            % SYNTAX:
            %   [x_est, P_est] = obj.getGaussianEstimate()
            %
            % OUTPUTS:
            %   x_est - State estimate [N_x x 1]
            %   P_est - Covariance estimate [N_x x N_x]
            %
            % DESCRIPTION:
            %   For Kalman filter, returns the current internal state estimate.
            %   Required by DA_Filter abstract base class interface.
            %
            % See also timestep, measurement_update

            x_est = obj.x_current;
            P_est = obj.P_current;

            if obj.debug
                fprintf('[GAUSSIAN EST] State=[%.4f, %.4f], det(P)=%.8f\n', ...
                    x_est(1), x_est(2), det(P_est));
            end

        end

        %% ========== VISUALIZATION ==========

        %% TODO: Visualize Current Estimate and Uncertainty (already doing), with mWidar signal in the back (add), only plot validated mesurments for clarity
        %% Also add some sort of visualization technique for detection/mag liklihood
        function visualize(obj, varargin)
            % VISUALIZE Plot current filter state
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
            %   true_state    - (optional) True state for comparison [N_x x 1]
            %   signal        - (optional) Raw mWidar Signal
            %
            % DESCRIPTION:
            %   Creates visualization of Kalman filter state including state
            %   estimate, covariance ellipse, measurements, and true state.
            
            % Default Values
            signal = [];
            title_str = 'PDA-KF State Estimate';
            measurements = [];
            true_state = [];

            % Parse input arguments
            if nargin > 1 && ~isempty(varargin{1})
                figure(varargin{1});
            else
                figure(1);
            end

            if nargin > 2
                title_str = varargin{2};
            end

            if nargin > 3
                true_state = varargin{3};     
            end

            if nargin > 4
                signal = varargin{4};
            end

            tiledlayout(1,4)
            nexttile
            cla; hold on;

            % Plot state estimate
            plot(obj.x_current(1), obj.x_current(2), 'ro', 'MarkerSize', 10, 'LineWidth', 3, ...
                'DisplayName', 'KF Estimate');

            % Plot covariance ellipse (1-sigma and 2-sigma)
            if size(obj.P_current, 1) >= 2
                theta = linspace(0, 2 * pi, 100);
                pos_cov = obj.P_current(1:2, 1:2);

                % Eigenvalue decomposition for proper ellipse orientation
                [V, D] = eig(pos_cov);

                % 1-sigma ellipse (68% confidence)
                sigma1_scale = sqrt(chi2inv(0.68, 2));
                a1 = sigma1_scale * sqrt(D(1, 1));
                b1 = sigma1_scale * sqrt(D(2, 2));
                ellipse1_local = [a1 * cos(theta); b1 * sin(theta)];
                ellipse1_global = V * ellipse1_local + obj.x_current(1:2);
                plot(ellipse1_global(1, :), ellipse1_global(2, :), 'r-', 'LineWidth', 2, ...
                    'DisplayName', '1σ Covariance');

                % 2-sigma ellipse (95% confidence)
                sigma2_scale = sqrt(chi2inv(0.95, 2));
                a2 = sigma2_scale * sqrt(D(1, 1));
                b2 = sigma2_scale * sqrt(D(2, 2));
                ellipse2_local = [a2 * cos(theta); b2 * sin(theta)];
                ellipse2_global = V * ellipse2_local + obj.x_current(1:2);
                plot(ellipse2_global(1, :), ellipse2_global(2, :), 'r--', 'LineWidth', 1.5, ...
                    'DisplayName', '2σ Covariance');
            end

            % Plot measurements
            plot(obj.valid_z(1, :), obj.valid_z(2, :), '+', 'Color', [1 0.5 0], ...
                    'MarkerSize', 10, 'LineWidth', 3, 'DisplayName', 'Measurements');
            

            % Plot true state if provided
            if ~isempty(true_state)
                plot(true_state(1), true_state(2), 'd', 'Color', 'm', ...
                    'MarkerSize', 8, 'LineWidth', 2, 'MarkerFaceColor', 'm', ...
                    'DisplayName', 'True Position');
            end
            
            % Plot mWidar Signal if provided
            if ~isempty(signal)
                Lscene = 4;
                npx = 128;
                xgrid = linspace(-2, 2, npx);
                ygrid = linspace(0, Lscene, npx);
                [pxgrid, pygrid] = meshgrid(xgrid, ygrid);

                surf(pxgrid, pygrid, signal / (max(max(signal))), 'EdgeColor', 'none')
            end

            % Formatting
            xlabel('X Position (m)');
            ylabel('Y Position (m)');
            title(title_str, 'Interpreter', 'latex');
            legend('Location', 'best');
            grid on;
            
            % Set pinned axis limits for consistent positioning
            xlim([-2 2]);
            ylim([0 4]);
            axis square;
            
            % Likelihood vs Detection index
            nexttile
            cla; hold on
            bar(obj.detection_liklihood,'b')

            nexttile
            cla; hold on
            bar(obj.magnitude_liklihood,'r')

            nexttile
            cla; hold on
            stacked_liklihood = [obj.detection_liklihood;obj.magnitude_liklihood;obj.composite_liklihood];
            bar(stacked_liklihood)

            drawnow;
        end

    end

end
