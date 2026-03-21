classdef HMM < handle
    % HMM Grid-Based Hidden Markov Model base class
    %
    % DESCRIPTION:
    %   Implements a standard (no data association) grid-based HMM for single
    %   target tracking. The state is a discrete probability distribution over
    %   a 128x128 spatial grid. Designed to serve as the inner filter for each
    %   particle in an RBPF_HMM; data association is handled externally.
    %
    %   The predict/update cycle mirrors the KF interface:
    %     prediction()          -- propagate prior via A_transition
    %     measurement_update(z) -- apply likelihood lookup, no DA weighting
    %     timestep(z)           -- single-call predict + update
    %
    % PROPERTIES:
    %   ptarget_prob            - Current probability distribution [npx2 x 1]
    %   prior_prob              - Copy of prior after prediction [npx2 x 1]
    %   likelihood_prob         - Likelihood grid from last update [npx2 x 1]
    %   posterior_prob          - Copy of posterior after update [npx2 x 1]
    %   A_transition            - HMM state transition matrix [npx2 x npx2]
    %   pointlikelihood_image   - Precomputed likelihood lookup table [npx2 x npx2]
    %
    % METHODS:
    %   HMM                      - Constructor
    %   timestep                 - Process single time step (predict + update)
    %   prediction               - HMM prediction step via transition matrix
    %   measurement_update       - Standard (no-DA) likelihood measurement update
    %   likelihoodLookup         - Detection likelihood grid with optional Gaussian mask
    %   magnitudeLikelihoodGrid  - Magnitude likelihood grid from signal frame
    %   getGaussianEstimate      - Extract MMSE mean and covariance from grid
    %   getMAPEstimate           - Extract MAP position estimate from grid
    %   getEntropy               - Compute Shannon entropy of distribution
    %   validateState            - Check normalization and consistency
    %   copyHMM (Static)         - Deep copy an HMM object (required for RBPF)
    %
    % EXAMPLE:
    %   hmm = HMM(x0, A_transition, pointlikelihood_image);
    %   [x_est, P_est] = hmm.timestep(z);
    %
    %   % With uniform prior (no initial position known):
    %   hmm = HMM([], A_transition, pointlikelihood_image);
    %
    %   % With hybrid (detection + magnitude) likelihood:
    %   hmm = HMM(x0, A_transition, pointlikelihood_image, ...
    %             'PointlikelihoodMag', pointlikelihood_mag, ...
    %             'MagnitudeWeight', 0.1);
    %   [x_est, P_est] = hmm.timestep(z, z_mag);
    %
    % See also PDA_HMM, PDA_PF, KF

    properties
        % Grid Parameters (hardcoded defaults matching project convention)
        grid_size = 128         % Grid size (128x128)
        npx2                    % Total grid points (128^2 = 16384)

        % Scene Parameters (hardcoded defaults matching project convention)
        Xbounds = [-2, 2]       % X bounds of scene in metres
        Ybounds = [0, 4]        % Y bounds of scene in metres

        % Grid State
        ptarget_prob            % Current probability distribution [npx2 x 1]

        % Snapshots for visualization / RBPF weight computation
        prior_prob              % Prior after prediction [npx2 x 1]
        likelihood_prob         % Likelihood grid from last update [npx2 x 1]
        posterior_prob          % Posterior after update [npx2 x 1]

        % HMM Model
        A_transition            % State transition matrix [npx2 x npx2]
        pointlikelihood_image   % Detection likelihood lookup table [npx2 x npx2]

        % Hybrid (composite) likelihood -- optional
        % pointlikelihood_mag stores [mu, sigma] per grid cell [npx2 x 2].
        % When provided, the final likelihood is:
        %   L_composite = L_detection .* (magnitude_weight * L_magnitude)
        % where L_magnitude(i) = normpdf(z_mag(i), mu(i), sigma(i)).
        pointlikelihood_mag     % Magnitude likelihood params table [npx2 x 2] ([] = disabled)
        magnitude_weight = 1.0  % Scalar weight on magnitude likelihood (tune to balance vs detection)

        % Spatial Grid
        xgrid                   % X coordinate grid [1 x 128]
        ygrid                   % Y coordinate grid [1 x 128]
        pxgrid                  % X meshgrid [128 x 128]
        pygrid                  % Y meshgrid [128 x 128]
        pxyvec                  % Vectorized coordinates [npx2 x 2]
        dx                      % X grid resolution (m/cell)
        dy                      % Y grid resolution (m/cell)

        % Control Flags
        debug = false           % Enable verbose debug output
    end

    methods

        %% ========== CONSTRUCTOR ==========
        function obj = HMM(x0, A_transition, pointlikelihood_image, varargin)
            % HMM Constructor for grid-based Hidden Markov Model
            %
            % SYNTAX:
            %   obj = HMM(x0, A_transition, pointlikelihood_image)
            %   obj = HMM([], A_transition, pointlikelihood_image)
            %   obj = HMM(x0, A_transition, pointlikelihood_image, ...
            %             'PointlikelihoodMag', PLmag, 'MagnitudeWeight', 0.1, 'Debug', true)
            %
            % INPUTS:
            %   x0                    - Initial position estimate [2 x 1], metres.
            %                           Pass [] for a uniform prior over the full grid.
            %   A_transition          - HMM transition matrix [npx2 x npx2]
            %   pointlikelihood_image - Detection likelihood lookup table [npx2 x npx2]
            %
            % OPTIONAL NAME-VALUE PAIRS:
            %   'Debug'              - true/false (default false)
            %   'PointlikelihoodMag' - [npx2 x 2] table of [mu, sigma] per grid cell
            %                          for hybrid (composite) likelihood. [] = disabled.
            %   'MagnitudeWeight'    - Scalar weight applied to magnitude likelihood
            %                          before product with detection likelihood.
            %                          Default 1.0. Set to 0.1 to match PDA_PF behaviour.
            %
            % OUTPUTS:
            %   obj - Initialized HMM object

            if nargin < 3
                error('HMM:InvalidInput', ...
                    'Requires at least 3 inputs: {x0, A_transition, pointlikelihood_image}');
            end

            % Parse optional name-value pairs
            p = inputParser;
            addParameter(p, 'Debug',              false, @islogical);
            addParameter(p, 'PointlikelihoodMag', [],    @(x) isempty(x) || isnumeric(x));
            addParameter(p, 'MagnitudeWeight',    1.0,   @(x) isnumeric(x) && isscalar(x) && x > 0);
            parse(p, varargin{:});
            obj.debug            = p.Results.Debug;
            obj.magnitude_weight = p.Results.MagnitudeWeight;

            % Initialize grid parameters
            obj.npx2 = obj.grid_size ^ 2;

            % Build spatial grid
            obj.xgrid  = linspace(obj.Xbounds(1), obj.Xbounds(2), obj.grid_size);
            obj.ygrid  = linspace(obj.Ybounds(1), obj.Ybounds(2), obj.grid_size);
            [obj.pxgrid, obj.pygrid] = meshgrid(obj.xgrid, obj.ygrid);
            obj.pxyvec = [obj.pxgrid(:), obj.pygrid(:)];
            obj.dx = obj.xgrid(2) - obj.xgrid(1);
            obj.dy = obj.ygrid(2) - obj.ygrid(1);

            % Validate and store detection likelihood matrix
            if size(A_transition, 1) ~= obj.npx2 || size(A_transition, 2) ~= obj.npx2
                error('HMM:InvalidDimensions', ...
                    'A_transition must be [%d x %d], got [%d x %d]', ...
                    obj.npx2, obj.npx2, size(A_transition, 1), size(A_transition, 2));
            end

            if size(pointlikelihood_image, 1) ~= obj.npx2 || size(pointlikelihood_image, 2) ~= obj.npx2
                error('HMM:InvalidDimensions', ...
                    'pointlikelihood_image must be [%d x %d], got [%d x %d]', ...
                    obj.npx2, obj.npx2, size(pointlikelihood_image, 1), size(pointlikelihood_image, 2));
            end

            obj.A_transition          = A_transition;
            obj.pointlikelihood_image = pointlikelihood_image;

            % Validate and store magnitude likelihood table (optional)
            pl_mag = p.Results.PointlikelihoodMag;
            if ~isempty(pl_mag)
                if size(pl_mag, 1) ~= obj.npx2 || size(pl_mag, 2) ~= 2
                    error('HMM:InvalidDimensions', ...
                        'PointlikelihoodMag must be [%d x 2], got [%d x %d]', ...
                        obj.npx2, size(pl_mag, 1), size(pl_mag, 2));
                end
                obj.pointlikelihood_mag = pl_mag;
            else
                obj.pointlikelihood_mag = [];
            end

            % Initialize probability distribution
            if isempty(x0)
                obj.initUniformPrior();
            else
                obj.initGaussianPrior(x0);
            end

            if obj.debug
                fprintf('\n=== HMM INITIALIZATION ===\n');
                fprintf('Grid:      %dx%d (%.4f m/cell x, %.4f m/cell y)\n', ...
                    obj.grid_size, obj.grid_size, obj.dx, obj.dy);
                fprintf('Scene:     X[%.1f, %.1f], Y[%.1f, %.1f] m\n', ...
                    obj.Xbounds(1), obj.Xbounds(2), obj.Ybounds(1), obj.Ybounds(2));
                if isempty(x0)
                    fprintf('Prior:     Uniform over full grid\n');
                else
                    fprintf('Prior:     Gaussian centered at [%.3f, %.3f]\n', x0(1), x0(2));
                end
                if ~isempty(obj.pointlikelihood_mag)
                    fprintf('Likelihood: HYBRID (detection x magnitude, weight=%.3f)\n', obj.magnitude_weight);
                else
                    fprintf('Likelihood: Detection only\n');
                end
                fprintf('==========================\n\n');
            end
        end

        %% ========== PRIOR INITIALIZATION HELPERS ==========
        function initUniformPrior(obj)
            % INITUNIFORMPRIOR Initialize with a uniform distribution over the grid
            obj.ptarget_prob = ones(obj.npx2, 1) / obj.npx2;

            if obj.debug
                fprintf('-> Initialized uniform prior (1/%d per cell)\n', obj.npx2);
            end
        end

        function initGaussianPrior(obj, x0)
            % INITGAUSSIANPRIOR Initialize with a Gaussian distribution centred at x0
            %
            % INPUTS:
            %   x0 - Initial position [2 x 1] in metres

            sigma_init = 0.3; % Initial uncertainty in metres (matches PDA_HMM)

            % Vectorized Gaussian evaluation over all grid cells
            diff_x  = obj.pxyvec(:, 1) - x0(1);
            diff_y  = obj.pxyvec(:, 2) - x0(2);
            dist_sq = diff_x .^ 2 + diff_y .^ 2;
            obj.ptarget_prob = exp(-dist_sq / (2 * sigma_init ^ 2));

            % Normalize
            obj.ptarget_prob = obj.ptarget_prob / sum(obj.ptarget_prob);

            if obj.debug
                fprintf('-> Initialized Gaussian prior at [%.3f, %.3f] (sigma=%.3f m)\n', ...
                    x0(1), x0(2), sigma_init);
            end
        end

        %% ========== TIMESTEP ==========
        function [x_est, P_est] = timestep(obj, z, z_mag)
            % TIMESTEP Process a single time step with the HMM filter
            %
            % SYNTAX:
            %   [x_est, P_est] = obj.timestep(z)
            %   [x_est, P_est] = obj.timestep(z, z_mag)   % hybrid likelihood
            %
            % INPUTS:
            %   z     - Detection measurement(s) [2 x N_meas].
            %           Pass [] for a missed detection (prediction-only step).
            %   z_mag - (optional) Raw signal frame for magnitude likelihood.
            %           Accepts [128 x 128] image or [npx2 x 1] vector -- vectorized
            %           internally. Ignored if PointlikelihoodMag was not provided at
            %           construction. Pass [] to force detection-only even when table exists.
            %
            % OUTPUTS:
            %   x_est - MMSE position estimate [2 x 1]
            %   P_est - Position covariance [2 x 2]
            %
            % NOTE:
            %   For RBPF use, call prediction() and measurement_update() separately
            %   so the RBPF can intercept the likelihood for particle weighting.
            %
            % See also prediction, measurement_update, getGaussianEstimate

            if nargin < 3
                z_mag = [];
            end

            % Prediction step
            obj.prediction();

            % Measurement update step
            obj.measurement_update(z, z_mag);

            % Return Gaussian estimate
            [x_est, P_est] = obj.getGaussianEstimate();
        end

        %% ========== PREDICTION STEP ==========
        function prediction(obj)
            % PREDICTION HMM prediction step via state transition matrix
            %
            % SYNTAX:
            %   obj.prediction()
            %
            % ALGORITHM:
            %   P(x_k | z_{1:k-1}) = A_transition * P(x_{k-1} | z_{1:k-1})
            %
            % MODIFIES:
            %   obj.ptarget_prob - Updated predicted (prior) distribution
            %   obj.prior_prob   - Snapshot of predicted distribution

            if obj.debug
                fprintf('[PREDICTION] Applying transition matrix... prior sum=%.6f\n', ...
                    full(sum(obj.ptarget_prob)));
            end

            % Propagate through transition model
            obj.ptarget_prob = obj.A_transition * obj.ptarget_prob;

            % Re-normalize to guard against floating-point drift
            obj.ptarget_prob = obj.ptarget_prob / sum(obj.ptarget_prob);

            % Snapshot for visualization / RBPF weight computation
            obj.prior_prob = obj.ptarget_prob;

            if obj.debug
                fprintf('[PREDICTION] Complete. Prior sum=%.6f, max=%.6f\n', ...
                    full(sum(obj.ptarget_prob)), full(max(obj.ptarget_prob)));
            end
        end

        %% ========== MEASUREMENT UPDATE ==========
        function measurement_update(obj, z, z_mag)
            % MEASUREMENT_UPDATE Standard (no-DA) Bayesian likelihood update
            %
            % SYNTAX:
            %   obj.measurement_update(z)
            %   obj.measurement_update(z, z_mag)   % hybrid: detection x magnitude
            %
            % INPUTS:
            %   z     - Detection measurement(s) [2 x N_meas]. [] = missed detection.
            %   z_mag - (optional) Raw signal frame [128 x 128] or [npx2 x 1].
            %           Only used when obj.pointlikelihood_mag is loaded.
            %
            % ALGORITHM:
            %   Detection only:
            %     P(x_k | z_{1:k}) ∝ L_det(z | x) * P(x_k | z_{1:k-1})
            %
            %   Hybrid (detection x magnitude):
            %     P(x_k | z_{1:k}) ∝ L_det(z | x) .* L_mag(z_mag | x) * P(x_k | z_{1:k-1})
            %
            %   For N_meas > 1 the detection likelihood grids are summed before
            %   the update (sensor fusion, NOT data association -- DA lives in RBPF).
            %
            % MODIFIES:
            %   obj.ptarget_prob    - Updated posterior distribution
            %   obj.likelihood_prob - Combined likelihood grid (for diagnostics)
            %   obj.posterior_prob  - Snapshot of posterior

            if nargin < 3
                z_mag = [];
            end

            % ---- Missed detection: keep prediction unchanged ----
            if isempty(z)
                if obj.debug
                    fprintf('[MEAS UPDATE] No measurements -- keeping prediction.\n');
                end
                obj.likelihood_prob = ones(obj.npx2, 1) / obj.npx2; % uninformative
                obj.posterior_prob  = obj.ptarget_prob;
                return;
            end

            N_meas = size(z, 2);

            if obj.debug
                fprintf('[MEAS UPDATE] %d measurement(s). Hybrid=%d\n', ...
                    N_meas, ~isempty(z_mag) && ~isempty(obj.pointlikelihood_mag));
            end

            % ---- Step 1: Detection likelihood grid ----
            % Sum over all detection measurements (sensor fusion, not DA).
            likelihood_det = zeros(obj.npx2, 1);

            for i = 1:N_meas
                likelihood_i  = obj.likelihoodLookup(z(:, i));
                likelihood_det = likelihood_det + likelihood_i;

                if obj.debug
                    fprintf('  Det meas %d [%.3f, %.3f]: max=%.6f, sum=%.6f\n', ...
                        i, z(1, i), z(2, i), full(max(likelihood_i)), full(sum(likelihood_i)));
                end
            end

            % ---- Step 2: Magnitude likelihood grid (optional) ----
            use_hybrid = ~isempty(z_mag) && ~isempty(obj.pointlikelihood_mag);

            if use_hybrid
                likelihood_mag = obj.magnitudeLikelihoodGrid(z_mag);
                % Composite: element-wise product, magnitude scaled by magnitude_weight
                likelihood_total = likelihood_det .* (obj.magnitude_weight * likelihood_mag);

                if obj.debug
                    fprintf('[HYBRID] mag: max=%.6f, sum=%.6f | composite: max=%.6f\n', ...
                        max(likelihood_mag), sum(likelihood_mag), max(likelihood_total));
                end
            else
                likelihood_total = likelihood_det;
            end

            % ---- Step 3: Bayesian update ----
            obj.ptarget_prob = obj.ptarget_prob .* likelihood_total + eps;

            % Normalize
            obj.ptarget_prob = obj.ptarget_prob / sum(obj.ptarget_prob);

            % Snapshots for diagnostics / visualization
            obj.likelihood_prob = likelihood_total;
            obj.posterior_prob  = obj.ptarget_prob;

            if obj.debug
                fprintf('[MEAS UPDATE] Complete. Posterior sum=%.6f, max=%.6f\n', ...
                    full(sum(obj.ptarget_prob)), full(max(obj.ptarget_prob)));
            end
        end

        %% ========== LIKELIHOOD LOOKUP ==========
        function likelihood_grid = likelihoodLookup(obj, measurement)
            % LIKELIHOODLOOKUP Detection likelihood grid for a single measurement
            %
            % SYNTAX:
            %   likelihood_grid = obj.likelihoodLookup(measurement)
            %
            % INPUTS:
            %   measurement - Single detection [2 x 1] in metres
            %
            % OUTPUTS:
            %   likelihood_grid - P(z_det | x) for all grid cells [npx2 x 1]
            %
            % DESCRIPTION:
            %   1. Snaps measurement to nearest grid cell (row of pointlikelihood_image).
            %   2. Applies a Gaussian spatial mask centred on the measurement to
            %      suppress whisker artifacts in the far-field of the likelihood image.
            %
            % See also measurement_update, magnitudeLikelihoodGrid

            % Snap measurement to nearest grid indices
            [~, meas_x_idx] = min(abs(obj.xgrid - measurement(1)));
            [~, meas_y_idx] = min(abs(obj.ygrid - measurement(2)));
            meas_linear_idx = sub2ind([obj.grid_size, obj.grid_size], meas_y_idx, meas_x_idx);

            % Bounds check
            if meas_linear_idx < 1 || meas_linear_idx > size(obj.pointlikelihood_image, 1)
                error('HMM:IndexError', ...
                    'Measurement linear index %d out of bounds [1, %d]', ...
                    meas_linear_idx, size(obj.pointlikelihood_image, 1));
            end

            % Extract row as column vector; add eps to avoid log(0) downstream
            likelihood_grid = obj.pointlikelihood_image(meas_linear_idx, :)' + eps;

            % ---- GAUSSIAN SPATIAL MASK ------------------------------------------------
            % Suppresses whisker/sidelobe artifacts far from the measurement location.
            % The mask is a Gaussian centred at the measurement; cells below 10 % of the
            % peak are zeroed.  Adjust sf (metres) to control the suppression radius.
            % To DISABLE masking: comment out from here ...
            sf = 0.15; % Gaussian half-width in metres (matches PDA_HMM / PDA_PF)
            diff_x   = obj.pxyvec(:, 1) - measurement(1);
            diff_y   = obj.pxyvec(:, 2) - measurement(2);
            dist_sq  = diff_x .^ 2 + diff_y .^ 2;
            gaussmask = exp(-dist_sq / (2 * sf ^ 2));
            gaussmask(gaussmask < 0.1 * max(gaussmask)) = 0; % threshold small values
            likelihood_grid = likelihood_grid .* gaussmask;
            % ... to here to DISABLE masking.
            % ---- END GAUSSIAN SPATIAL MASK --------------------------------------------

            if obj.debug
                fprintf('    likelihoodLookup: grid_idx=%d, max=%.6f, sum=%.6f\n', ...
                    meas_linear_idx, full(max(likelihood_grid)), full(sum(likelihood_grid)));
            end
        end

        %% ========== MAGNITUDE LIKELIHOOD GRID ==========
        function likelihood_mag = magnitudeLikelihoodGrid(obj, z_mag)
            % MAGNITUDELIKELIHOODGRID Magnitude likelihood grid from signal frame
            %
            % SYNTAX:
            %   likelihood_mag = obj.magnitudeLikelihoodGrid(z_mag)
            %
            % INPUTS:
            %   z_mag - Raw signal frame. Accepted formats:
            %             [128 x 128] image  -- vectorized internally (column-major)
            %             [npx2 x 1]  vector -- used as-is
            %
            % OUTPUTS:
            %   likelihood_mag - P(z_mag | x) for all grid cells [npx2 x 1]
            %
            % DESCRIPTION:
            %   For each grid cell i, evaluates:
            %     L_mag(i) = normpdf( z_mag(i), mu(i), sigma(i) )
            %   where [mu(i), sigma(i)] = pointlikelihood_mag(i, :).
            %
            %   This is the grid-HMM equivalent of the PDA_PF magnitude likelihood:
            %   the question asked is "is the signal at cell i consistent with what
            %   the sensor model predicts at cell i?" -- a state-conditional quantity.
            %
            % NOTE:
            %   Requires obj.pointlikelihood_mag to be loaded (non-empty).
            %
            % See also measurement_update

            if isempty(obj.pointlikelihood_mag)
                error('HMM:NoMagTable', ...
                    'pointlikelihood_mag not loaded. Pass ''PointlikelihoodMag'' to constructor.');
            end

            % Vectorize input: accept [128 x 128] image or [npx2 x 1] vector
            z_vec = z_mag(:); % column-major flatten -- same indexing as pxyvec

            if numel(z_vec) ~= obj.npx2
                error('HMM:DimMismatch', ...
                    'z_mag must contain %d elements (got %d).', obj.npx2, numel(z_vec));
            end

            % Read precomputed [mu, sigma] for each grid cell
            mu_grid    = obj.pointlikelihood_mag(:, 1); % [npx2 x 1]
            sigma_grid = obj.pointlikelihood_mag(:, 2); % [npx2 x 1]

            % Evaluate Gaussian likelihood cell-by-cell (vectorized)
            % normpdf(x, mu, sigma) = (1/(sigma*sqrt(2pi))) * exp(-0.5*((x-mu)/sigma)^2)
            likelihood_mag = normpdf(z_vec, mu_grid, sigma_grid) + eps;

            if obj.debug
                fprintf('[MAG LIKELIHOOD] max=%.6f, sum=%.6f, min=%.2e\n', ...
                    max(likelihood_mag), sum(likelihood_mag), min(likelihood_mag));
            end
        end

        %% ========== STATE ESTIMATION ==========
        function [x_est, P_est] = getGaussianEstimate(obj)
            % GETGAUSSIANESTIMATE Extract MMSE mean and covariance from grid distribution
            %
            % SYNTAX:
            %   [x_est, P_est] = obj.getGaussianEstimate()
            %
            % OUTPUTS:
            %   x_est - MMSE position estimate [2 x 1]
            %   P_est - Position covariance [2 x 2]
            %
            % NOTE:
            %   Also used by the RBPF to propagate a Gaussian summary of each
            %   particle's HMM for importance weight computation.
            %
            % See also getMAPEstimate, timestep

            % MMSE estimate: weighted mean of grid positions
            prob_full = full(obj.ptarget_prob);
            x_est = sum(obj.pxyvec .* repmat(prob_full, [1, 2]), 1)';

            % Covariance: E[xx^T] - mu*mu^T
            second_moment = reshape( ...
                sum([obj.pxyvec(:, 1) .^ 2, ...
                     obj.pxyvec(:, 1) .* obj.pxyvec(:, 2), ...
                     obj.pxyvec(:, 2) .* obj.pxyvec(:, 1), ...
                     obj.pxyvec(:, 2) .^ 2] .* repmat(prob_full, [1, 4]), 1), [2, 2]);

            P_est = second_moment - x_est * x_est';

            % Symmetrize and regularize
            P_est = 0.5 * (P_est + P_est') + 1e-8 * eye(2);

            if obj.debug
                fprintf('[GAUSSIAN EST] Mean=[%.4f, %.4f], trace(P)=%.6f\n', ...
                    x_est(1), x_est(2), trace(P_est));
            end
        end

        %% ========== MAP ESTIMATION ==========
        function [x_map, map_prob] = getMAPEstimate(obj)
            % GETMAPESTIMATE Get Maximum A Posteriori position estimate
            %
            % SYNTAX:
            %   [x_map, map_prob] = obj.getMAPEstimate()
            %
            % OUTPUTS:
            %   x_map    - MAP position estimate [2 x 1]
            %   map_prob - Probability at MAP grid cell

            [map_prob, map_idx] = max(obj.ptarget_prob);
            x_map = obj.pxyvec(map_idx, :)';

            if obj.debug
                fprintf('[MAP EST] Position=[%.4f, %.4f], prob=%.6f\n', ...
                    x_map(1), x_map(2), full(map_prob));
            end
        end

        %% ========== ENTROPY ==========
        function entropy = getEntropy(obj)
            % GETENTROPY Compute Shannon entropy of current distribution
            %
            % SYNTAX:
            %   entropy = obj.getEntropy()
            %
            % OUTPUTS:
            %   entropy - H = -sum(p * log(p))

            p_safe = full(obj.ptarget_prob) + eps;
            entropy = -sum(p_safe .* log(p_safe));

            if obj.debug
                fprintf('[ENTROPY] H=%.4f\n', entropy);
            end
        end

        %% ========== VALIDATION ==========
        function validateState(obj)
            % VALIDATESTATE Check normalization and non-negativity of distribution
            %
            % SYNTAX:
            %   obj.validateState()

            prob_sum = full(sum(obj.ptarget_prob));

            if abs(prob_sum - 1.0) > 1e-6
                warning('HMM:Normalization', ...
                    'Distribution not normalized: sum=%.8f', prob_sum);
            end

            if any(full(obj.ptarget_prob) < 0)
                warning('HMM:NegativeProbability', 'Negative probabilities detected.');
            end

            if obj.debug
                fprintf('[VALIDATE] sum=%.8f, min=%.2e, max=%.6f\n', ...
                    prob_sum, full(min(obj.ptarget_prob)), full(max(obj.ptarget_prob)));
            end
        end

    end

    methods (Static)

        %% ========== DEEP COPY ==========
        function new_hmm = copyHMM(orig)
            % COPYHMM Deep copy an HMM object
            %
            % SYNTAX:
            %   new_hmm = HMM.copyHMM(orig)
            %
            % INPUTS:
            %   orig - Source HMM object
            %
            % OUTPUTS:
            %   new_hmm - Independent HMM object with separate memory
            %
            % CRITICAL:
            %   MATLAB handle classes share memory by reference. Without this,
            %   RBPF resampling would create particles that all point to the same
            %   HMM object -- identical to the KF.copyKF requirement.
            %
            %   The large read-only lookup tables (A_transition,
            %   pointlikelihood_image) are shared between the copy and the
            %   original; only the mutable distribution state is duplicated.
            %
            % See also KF.copyKF

            % Re-use the stored matrices (read-only lookup tables -- safe to share).
            new_hmm = HMM([], orig.A_transition, orig.pointlikelihood_image, ...
                          'Debug',              orig.debug, ...
                          'PointlikelihoodMag', orig.pointlikelihood_mag, ...
                          'MagnitudeWeight',    orig.magnitude_weight);

            % Copy the mutable distribution state
            new_hmm.ptarget_prob    = orig.ptarget_prob;
            new_hmm.prior_prob      = orig.prior_prob;
            new_hmm.likelihood_prob = orig.likelihood_prob;
            new_hmm.posterior_prob  = orig.posterior_prob;
        end

    end

end
