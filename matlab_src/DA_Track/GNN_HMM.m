% GNN_HMM: Global Nearest Neighbor Data Association with Hidden Markov Model Tracking
% Combines GNN data association with HMM discrete state space filtering
% TODO: Implement GNN assignment algorithm and integrate with HMM state updates

classdef GNN_HMM
    properties
        % HMM Grid Parameters
        xgrid % Spatial grid x-coordinates
        ygrid % Spatial grid y-coordinates
        npx % Number of pixels/grid points per dimension
        npx2 % Total number of grid points (npx^2)
        pxyvec % Vectorized grid coordinates [npx2 x 2]
        
        % HMM Transition Models
        A_slow % Slow dynamics transition matrix [npx2 x npx2]
        A_fast % Fast dynamics transition matrix [npx2 x npx2]
        
        % HMM Likelihood Model
        pointlikelihood_image % Pre-computed likelihood model [npx2 x npx2]
        
        % Scene Parameters
        Xbounds % Scene X boundaries [min, max]
        Ybounds % Scene Y boundaries [min, max]
        
        % Filter State
        ptarget_prob % Current target probability distribution [npx2 x 1]
        
        % GNN Parameters
        gating_threshold % Chi-squared threshold for validation gate
        cost_matrix % Assignment cost matrix
        assignment_threshold % Threshold for valid assignments
    end

    methods
        function obj = GNN_HMM(xgrid, ygrid, A_slow, A_fast, pointlikelihood_image, Xbounds, Ybounds)
            %%% CLASS CONSTRUCTOR
            % Initialize GNN-HMM tracker with grid parameters and pre-computed models
            %
            % Args:
            %   xgrid - X-coordinate grid vector (or [] to use default)
            %   ygrid - Y-coordinate grid vector (or [] to use default)
            %   A_slow - Slow dynamics transition matrix (or [] to load default)
            %   A_fast - Fast dynamics transition matrix (or [] to load default)
            %   pointlikelihood_image - Pre-computed likelihood model (or [] to load default)
            %   Xbounds - Scene X boundaries [min, max] (or [] to use default [-2, 2])
            %   Ybounds - Scene Y boundaries [min, max] (or [] to use default [0, 4])
            
            % Handle variable number of inputs - allow defaults
            if nargin < 7 || isempty(Ybounds)
                Ybounds = [0, 4]; % Default Y bounds
            end
            if nargin < 6 || isempty(Xbounds)
                Xbounds = [-2, 2]; % Default X bounds
            end
            if nargin < 5
                pointlikelihood_image = [];
            end
            if nargin < 4
                A_fast = [];
            end
            if nargin < 3
                A_slow = [];
            end
            if nargin < 2 || isempty(ygrid)
                % Create default grid
                Lscene = 4; % Physical length of scene in m (square shape)
                npx = 128; % Number of pixels in image (same in x&y dims)
                ygrid = linspace(Ybounds(1), Ybounds(2), npx);
            end
            if nargin < 1 || isempty(xgrid)
                % Create default grid
                Lscene = 4; % Physical length of scene in m (square shape)  
                npx = 128; % Number of pixels in image (same in x&y dims)
                xgrid = linspace(Xbounds(1), Xbounds(2), npx);
            end
            
            obj.xgrid = xgrid;
            obj.ygrid = ygrid;
            obj.npx = length(xgrid);
            obj.npx2 = obj.npx^2;
            obj.Xbounds = Xbounds;
            obj.Ybounds = Ybounds;
            
            % Create vectorized grid coordinates
            [pxgrid, pygrid] = meshgrid(xgrid, ygrid);
            obj.pxyvec = [pxgrid(:), pygrid(:)];
            
            % Load HMM model parameters if not provided
            if isempty(A_slow) || isempty(A_fast) || isempty(pointlikelihood_image)
                fprintf('Loading default HMM model parameters...\n');
                
                % Try to load from relative paths (assuming we're in DA_Track folder)
                base_path = '../../testTrackers/data/';
                
                % Load slow dynamics transition matrix
                if isempty(A_slow)
                    try
                        slow_file = fullfile(base_path, 'precalc_imagegridHMMSTMn15.mat');
                        load(slow_file, 'A');
                        obj.A_slow = A;
                        clear A;
                        fprintf('  Loaded A_slow from %s\n', slow_file);
                    catch ME
                        error('Failed to load default A_slow matrix: %s\nFile: %s', ME.message, slow_file);
                    end
                else
                    obj.A_slow = A_slow;
                end
                
                % Load fast dynamics transition matrix
                if isempty(A_fast)
                    try
                        fast_file = fullfile(base_path, 'precalc_imagegridHMMSTMn30.mat');
                        load(fast_file, 'A');
                        obj.A_fast = A;
                        clear A;
                        fprintf('  Loaded A_fast from %s\n', fast_file);
                    catch ME
                        error('Failed to load default A_fast matrix: %s\nFile: %s', ME.message, fast_file);
                    end
                else
                    obj.A_fast = A_fast;
                end
                
                % Load likelihood model
                if isempty(pointlikelihood_image)
                    try
                        like_file = fullfile(base_path, 'precalc_imagegridHMMEmLike.mat');
                        load(like_file, 'pointlikelihood_image');
                        obj.pointlikelihood_image = pointlikelihood_image;
                        fprintf('  Loaded pointlikelihood_image from %s\n', like_file);
                    catch ME
                        error('Failed to load default pointlikelihood_image: %s\nFile: %s', ME.message, like_file);
                    end
                else
                    obj.pointlikelihood_image = pointlikelihood_image;
                end
                
                % Validate loaded parameters
                if size(obj.A_slow, 1) ~= obj.npx2 || size(obj.A_fast, 1) ~= obj.npx2
                    error('Transition matrix dimensions (%dx%d) do not match grid size (%d)', ...
                        size(obj.A_slow, 1), size(obj.A_slow, 2), obj.npx2);
                end
                
                if size(obj.pointlikelihood_image, 1) ~= obj.npx2 || size(obj.pointlikelihood_image, 2) ~= obj.npx2
                    error('Likelihood model dimensions (%dx%d) do not match grid size (%d)', ...
                        size(obj.pointlikelihood_image, 1), size(obj.pointlikelihood_image, 2), obj.npx2);
                end
                
                fprintf('Successfully loaded and validated default HMM matrices.\n');
            else
                % Use provided parameters
                obj.A_slow = A_slow;
                obj.A_fast = A_fast;
                obj.pointlikelihood_image = pointlikelihood_image;
                fprintf('Using provided HMM model parameters.\n');
            end
            
            % Initialize GNN parameters
            obj.gating_threshold = chi2inv(0.95, 2); % 95% confidence gate
            obj.assignment_threshold = 1e-6; % Minimum assignment probability
            
            % Initialize target probability (will be set properly in first update)
            obj.ptarget_prob = sparse(obj.npx2, 1);
            
            fprintf('GNN_HMM tracker initialized with %dx%d grid (%.4fm resolution)\n', ...
                obj.npx, obj.npx, obj.xgrid(2) - obj.xgrid(1));
        end

        function obj = initialize_target_probability(obj, init_pos, init_sigma)
            %%% Initialize target probability distribution around given position
            %
            % Args:
            %   init_pos - Initial position [x, y]
            %   init_sigma - Initial uncertainty (scalar or 2x2 covariance)
            
            obj.ptarget_prob = sparse(obj.npx2, 1);
            
            if isscalar(init_sigma)
                sigma_cov = init_sigma^2 * eye(2);
            else
                sigma_cov = init_sigma;
            end
            
            % Initialize with Gaussian distribution around initial position
            for i = 1:obj.npx2
                [row, col] = ind2sub([obj.npx, obj.npx], i);
                x_pos = obj.xgrid(col);
                y_pos = obj.ygrid(row);
                grid_pos = [x_pos; y_pos];
                
                dist_sq = (grid_pos - init_pos(:))' / sigma_cov * (grid_pos - init_pos(:));
                obj.ptarget_prob(i) = exp(-0.5 * dist_sq);
            end
            
            % Normalize
            obj.ptarget_prob = obj.ptarget_prob / sum(obj.ptarget_prob);
        end

        %% timestep: Iterate GNN-HMM through a single timestep
        %{
            args:
                measurements -> matrix of all detections [2 x num_detections]
                use_fast_dynamics -> boolean to select transition model
            outputs:
                obj -> Updated object with new target probability
                assignment -> GNN assignment result
                mmse_estimate -> MMSE position estimate [x; y]
                map_estimate -> MAP position estimate [x; y]
        %}
        function [obj, assignment, mmse_estimate, map_estimate] = timestep(obj, measurements, use_fast_dynamics)
            
            % Default to slow dynamics if not specified
            if nargin < 3
                use_fast_dynamics = false;
            end
            
            %% ========== PREDICTION STEP ==========
            obj = obj.prediction_step(use_fast_dynamics);
            
            %% ========== GNN DATA ASSOCIATION STEP ==========
            % TODO: Implement GNN data association algorithm
            assignment = obj.gnn_data_association(measurements);
            
            %% ========== MEASUREMENT UPDATE STEP ==========
            obj = obj.measurement_update_step(assignment, measurements);
            
            %% ========== STATE ESTIMATION ==========
            [mmse_estimate, map_estimate] = obj.compute_estimates();
        end

        %% Prediction Step: Apply HMM state transition model
        %{
            args:
                use_fast_dynamics -> boolean to select transition model
            outputs:
                obj -> Updated object with predicted probability
        %}   
        function obj = prediction_step(obj, use_fast_dynamics)
            % Apply state transition model (dynamics)
            if use_fast_dynamics
                obj.ptarget_prob = obj.A_fast * obj.ptarget_prob;
            else
                obj.ptarget_prob = obj.A_slow * obj.ptarget_prob;
            end
        end

        %% GNN Data Association: Global Nearest Neighbor assignment
        %{
            args:
                measurements -> matrix of detections [2 x num_detections]
            outputs:
                assignment -> assignment result structure
                    .assigned_measurement -> index of assigned measurement (0 if no assignment)
                    .assignment_cost -> cost of assignment
                    .gated_measurements -> indices of measurements passing validation gate
        %}
        function assignment = gnn_data_association(obj, measurements)
            % TODO: Implement GNN data association algorithm
            
            % Initialize assignment structure
            assignment = struct();
            assignment.assigned_measurement = 0; % No assignment by default
            assignment.assignment_cost = inf;
            assignment.gated_measurements = [];
            
            if isempty(measurements)
                return;
            end
            
            % Placeholder implementation - select closest measurement to predicted position
            % In full implementation, this would use Hungarian algorithm or similar
            
            % Get predicted position (MMSE of current probability)
            predicted_pos = sum(obj.pxyvec .* repmat(obj.ptarget_prob, [1, 2]), 1)';
            
            % Compute distances to all measurements
            num_measurements = size(measurements, 2);
            distances = zeros(1, num_measurements);
            
            for i = 1:num_measurements
                distances(i) = norm(measurements(:, i) - predicted_pos);
            end
            
            % Apply validation gate (simplified chi-squared test)
            % TODO: Implement proper Mahalanobis distance gating
            gate_threshold = sqrt(obj.gating_threshold) * 0.2; % Simplified distance threshold
            gated_indices = find(distances <= gate_threshold);
            
            assignment.gated_measurements = gated_indices;
            
            if ~isempty(gated_indices)
                % Select closest gated measurement
                [min_dist, min_idx] = min(distances(gated_indices));
                assignment.assigned_measurement = gated_indices(min_idx);
                assignment.assignment_cost = min_dist;
            end
            
            % TODO: Implement full GNN algorithm:
            % 1. Compute cost matrix for all target-measurement pairs
            % 2. Apply Hungarian algorithm for optimal assignment
            % 3. Handle multiple targets and track management
        end

        %% Measurement Update: Apply HMM likelihood update with assigned measurement
        %{
            args:
                assignment -> assignment result from GNN
                measurements -> matrix of all measurements
            outputs:
                obj -> Updated object with posterior probability
        %}
        function obj = measurement_update_step(obj, assignment, measurements)
            
            if assignment.assigned_measurement == 0
                % No measurement assigned - no update needed
                % Probability distribution remains at prediction
                return;
            end
            
            % Get assigned measurement
            assigned_meas = measurements(:, assignment.assigned_measurement);
            
            % Find closest grid point to measurement for likelihood lookup
            [~, meas_x_idx] = min(abs(obj.xgrid - assigned_meas(1)));
            [~, meas_y_idx] = min(abs(obj.ygrid - assigned_meas(2)));
            meas_linear_idx = sub2ind([obj.npx, obj.npx], meas_y_idx, meas_x_idx);
            
            % Get likelihood function from pre-computed model
            likelihood_raw = obj.pointlikelihood_image(meas_linear_idx, :)';
            
            % Apply Gaussian mask around measurement for improved localization
            sf = 0.15; % scaling factor for Gaussian mask
            meas_pos = [assigned_meas(1), assigned_meas(2)];
            gaussmask = mvnpdf(obj.pxyvec, meas_pos, sf * eye(2));
            gaussmask(gaussmask < 0.1 * max(gaussmask)) = 0; % threshold small values
            likelihood = likelihood_raw .* gaussmask;
            
            % Compute posterior: P(x_k | z_1:k) ∝ P(z_k | x_k) * P(x_k | z_1:k-1)
            obj.ptarget_prob = obj.ptarget_prob .* likelihood;
            
            % Normalize probability distribution
            obj.ptarget_prob = obj.ptarget_prob / sum(obj.ptarget_prob);
        end

        %% Compute Estimates: Calculate MMSE and MAP estimates from posterior
        %{
            outputs:
                mmse_estimate -> MMSE position estimate [x; y]
                map_estimate -> MAP position estimate [x; y]
        %}
        function [mmse_estimate, map_estimate] = compute_estimates(obj)
            
            % Compute MMSE estimate (mean of posterior)
            mmse_estimate = sum(obj.pxyvec .* repmat(obj.ptarget_prob, [1, 2]), 1)';
            
            % Compute MAP estimate (mode of posterior)
            [~, map_idx] = max(obj.ptarget_prob);
            map_estimate = obj.pxyvec(map_idx, :)';
        end

        %% Get current probability distribution (for visualization)
        function prob_dist = get_probability_distribution(obj)
            prob_dist = reshape(obj.ptarget_prob, [obj.npx, obj.npx]);
        end

    end

end