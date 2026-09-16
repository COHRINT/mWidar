
%%% Full TBD filter implementation
%%% Public entry points are run() and show()
%%% TBD_PF.run(scenario) will output a results struct, TBD_PF.show(R, scenario) plots it
%%%
%%% Conventions
%%%   - Particle state is [px; vx; py; vy] in METERS, matching the scenario truth.
%%%   - Signals are indexed z(row, col) = z(y, x), matching simulator / visualize.
%%%   - Sigma and pDist are in pixels; Sigma is scaled by dx inside the likelihood.

classdef TBD_PF < TBD

    methods
        %%% RUN TBD_PF
        %%% Main entry point. Takes the scenario struct from environment.setup()
        %%% (or just a 1 x K cell of npx x npx signals) and runs the bootstrap
        %%% PF over every frame.
        %%%
        %%%   R = pf.run(scenario)
        %%%
        %%%   R.particles   5 x N x K, posterior particle set [px; vx; py; vy; E] at each k
        %%%   R.pE          1 x K, existence probability = fraction of alive particles
        %%%   R.exist       1 x K logical, track declared where pE > pEthresh
        %%%   R.N, R.K      # of particles and # of timesteps run
        %%%
        %%% Metrics can be appended to R later, everything here is just the
        %%% raw filter output.
        function R = run(obj, scenario)

            %%% Measurements
            if isstruct(scenario)
                z = scenario.signal;
            elseif iscell(scenario)
                z = reshape(scenario, 1, []);
            else
                error('TBD_PF:run', 'expected a scenario struct or a 1 x K cell of signals');
            end
            K = numel(z);
            if K ~= obj.K
                obj.debug_print(sprintf("scenario has K=%d frames but obj.K=%d, running over the scenario", K, obj.K));
            end

            if ~isempty(obj.seed)
                rng(obj.seed);
            end

            obj.debug_print(sprintf("run: K=%d frames, N=%d particles, frame size %s", ...
                K, obj.N, mat2str(size(z{1}))));
            tStart = tic;

            %%% Initialize, every particle dead with undefined state
            prior = [nan(4, obj.N); false(1, obj.N)];

            R = struct();
            R.N         = obj.N;
            R.K         = K;
            R.particles = nan(5, obj.N, K);
            R.pE        = nan(1, K);
            R.exist     = false(1, K);

            %%% Filter
            for k = 1:K
                post = obj.timestep(prior, z{k});

                R.particles(:,:,k) = post;
                R.pE(k)    = mean(post(5,:));
                R.exist(k) = R.pE(k) > obj.pEthresh;

                obj.debug_print(sprintf("k=%d/%d  pE=%.3f  exist=%d", k, K, R.pE(k), R.exist(k)));

                prior = post; % Carry posterior forward as next step's prior
            end

            %%% Summary
            kDecl = find(R.exist);
            if isempty(kDecl)
                obj.debug_print(sprintf("run: done in %.2fs, track never declared (max pE=%.3f)", ...
                    toc(tStart), max(R.pE)));
            else
                obj.debug_print(sprintf("run: done in %.2fs, track declared %d/%d steps, first k=%d last k=%d", ...
                    toc(tStart), numel(kDecl), K, kDecl(1), kDecl(end)));
            end

        end

        %%% Quick look at a run. Draws the vis.plot_TBD dashboard (track over
        %%% the energy map, existence, per-axis position, error, particle
        %%% count) and optionally plays the frame-by-frame history with the
        %%% particle cloud. Everything is delegated to vis, same as
        %%% environment.show.
        %%%
        %%%   fig = pf.show(R, scenario)
        %%%   [fig, figAnim] = pf.show(R, scenario, 'Animate', true, 'Save', 'figs/run1')
        %%%
        %%% scenario is whatever was handed to run(). Pass [] to plot without
        %%% truth; the panels that need it are then skipped.
        %%%
        %%% Options
        %%%   'Animate'   also play the time history (default false, slow for big K)
        %%%   'Save'      base path with no extension. Writes <base>_tbd.png
        %%%               and, with Animate, <base>_history.gif
        %%%   'FPS'       animation frame rate (default vis.fps)
        %%%   'Trail'     # of past samples drawn in the animation (default 15)
        %%%   'Title'     override the auto title
        function [fig, figAnim] = show(obj, R, scenario, varargin)

            if nargin < 3
                scenario = [];
            end

            p = inputParser;
            addParameter(p, 'Animate', false, @islogical);
            addParameter(p, 'Save', '', @(x) ischar(x) || isstring(x));
            addParameter(p, 'FPS', obj.vis.fps);
            addParameter(p, 'Trail', 15);
            addParameter(p, 'Title', '');
            parse(p, varargin{:});
            opt = p.Results;

            figAnim = [];

            res = obj.to_vis_results(R, scenario);

            ttl = opt.Title;
            if isempty(ttl)
                ttl = sprintf('TBD-PF: N = %d, K = %d, Pb = %.3g, Ps = %.3g, pEthresh = %.2f', ...
                              R.N, R.K, obj.Pb, obj.Ps, obj.pEthresh);
            end

            savePath = '';
            if ~isempty(opt.Save)
                savePath = char(opt.Save) + "_tbd.png";
            end

            obj.debug_print(sprintf("show: dashboard, truth=%d signals=%d", ...
                ~isempty(res.truth), ~isempty(res.signals)));

            % Particle state is in meters (see sample_new), so tell vis that
            % regardless of what the axes are set to.
            fig = obj.vis.plot_TBD(res, ...
                'DataUnits', 'meters', ...
                'pEthresh',  obj.pEthresh, ...
                'Title',     ttl, ...
                'Save',      savePath);

            if opt.Animate
                if isempty(res.signals)
                    obj.debug_print("show: no signals to animate over, skipping Animate");
                    return
                end
                animPath = '';
                if ~isempty(opt.Save)
                    animPath = char(opt.Save) + "_history.gif";
                end
                obj.debug_print(sprintf("show: animating %d frames at %g fps", R.K, opt.FPS));
                figAnim = obj.vis.animate_time_history(res.signals, ...
                    'Truth',     res.truth, ...
                    'Particles', R.particles, ...
                    'pE',        R.pE, ...
                    'DataUnits', 'meters', ...
                    'Trail',     opt.Trail, ...
                    'FPS',       opt.FPS, ...
                    'Title',     ttl, ...
                    'Save',      animPath);
            end

        end
    end

    methods (Hidden)

        function w = importance_weights(obj, Y, z)

            X = Y(1:4);
            E = Y(5);

            w = 1;

            if(E) %% E = 1

                px = X(1);
                py = X(3);

                if px > obj.max_x || px < obj.min_x
                    w = 0;
                    return
                end

                % Going to set weights for particles below y = 0.25 to 0 to avoid sensor array
                if py > obj.max_y || py < 0.25
                    w = 0;
                    return
                end

                % Particle's pixel index (meters -> index, grid is uniform so this is exact)
                ix = round((px - obj.xgrid(1)) / obj.dx) + 1;
                iy = round((py - obj.ygrid(1)) / obj.dy) + 1;

                % Neighborhood window, clamped to valid pixel indices
                xvec = max(1, ix-obj.pDist):min(obj.npx, ix+obj.pDist);
                yvec = max(1, iy-obj.pDist):min(obj.npx, iy+obj.pDist);

                % Same window in meters, for the PSF distance
                xm = obj.xgrid(xvec);
                ym = obj.ygrid(yvec);

                for i = 1:numel(xvec)
                    for j = 1:numel(yvec)
                        idx = [xvec(i), yvec(j)]; % [col row] for indexing z
                        pix = [xm(i), ym(j)];     % meters
                        %% TODO: Add functionality here to choose likelihood function
                        % Gaussian for now
                        w = w*obj.guass_likelihood([px py], pix, idx, z);
                    end
                end

            else %% E = 0
                w = 1;
            end
        end

        %%% Likelihood ratio for one pixel (Ristic 11.20 style).
        %%%   pos  [px py] particle position in meters
        %%%   pix  [x y]   pixel center in meters
        %%%   idx  [col row] pixel index into z
        function l = guass_likelihood(obj, pos, pix, idx, z)
            px = pos(1);
            py = pos(2);

            ii = pix(1);
            jj = pix(2);

            sig = obj.Sigma * obj.dx; % Sigma is given in pixels
            hh = obj.Ip * exp( - ((ii -px)^2 + ((jj -py)^2))/ (2 * sig^2));
            zz = z(idx(2), idx(1)); % rows = y, cols = x
            l = exp(-(hh*(hh - 2*zz))/(2 * obj.NoiseSTD^2));
        end

        %%% Run through one timestep of TBD_PF, return posterior set of particles
        %%% Uniform weights (bootstrap PF)
        function post = timestep(obj, prior, z)
            x_minus = prior(1:4,:);
            E_minus = prior(5,:);

            % Regime transition
            E_plus = obj.RT(E_minus);
            x_plus = nan(4,obj.N);
            w_tilde = zeros(1,obj.N);
            Y = nan(5,obj.N);

            for n = 1:obj.N

                % New born particles
                if E_plus(n) && ~E_minus(n)
                    x_plus(:,n) = obj.sample_new(z);
                elseif E_plus(n) && E_minus(n)
                    x_plus(:,n) = obj.dynamics(x_minus(:,n));
                end

                % Evaluate importance weights
                Y(:, n) = [x_plus(:,n); E_plus(n)];
                w_tilde(n) = obj.importance_weights(Y(:,n),z);
            end

            if obj.debug
                born  = nnz(E_plus & ~E_minus);
                kept  = nnz(E_plus &  E_minus);
                died  = nnz(~E_plus & E_minus);
                w0    = nnz(E_plus & w_tilde == 0); % alive but zero weight (OOB or likelihood underflow)
                ess   = sum(w_tilde)^2 / max(sum(w_tilde.^2), eps);
                birthMode = "signal";
                if ~any(z(:) > obj.gamma)
                    birthMode = "uniform (no pixels above gamma)";
                end
                obj.debug_print(sprintf("timestep: born=%d kept=%d died=%d alive=%d w0=%d | w max=%.3g ESS=%.1f | birth=%s", ...
                    born, kept, died, nnz(E_plus), w0, max(w_tilde), ess, birthMode));
            end

            w = obj.normalize(w_tilde);
            post = obj.resample(Y,w);
        end

        %%% Per particle

        function X = sample_new(obj, z)
            % Birth proposal: pick a pixel above the threshold uniformly at
            % random and place the particle at its center in meters.
            % z is indexed z(row, col) = z(y, x), matching simulator.
            idx = find(z > obj.gamma);

            if isempty(idx)
                % Nothing above gamma, fall back to uniform over the scene
                % (timestep reports this once per frame, so no print here)
                px = obj.min_x + (obj.max_x - obj.min_x) * rand();
                py = obj.min_y + (obj.max_y - obj.min_y) * rand();
            else
                [r, c] = ind2sub(size(z), idx(randi(numel(idx))));
                px = obj.xgrid(c);
                py = obj.ygrid(r);
            end

            vx = obj.v_min + (obj.v_max - obj.v_min) * rand();
            vy = obj.v_min + (obj.v_max - obj.v_min) * rand();

            X = [px; vx; py; vy]; % Random new alive particle
        end


        function X_plus = dynamics(obj, X_minus)
            X_plus = obj.F * X_minus + mvnrnd(zeros(4,1), obj.Q)';
        end

        function post = resample(obj,pre, w)
            post = nan(5,obj.N);

            C = cumsum(w);
            u = rand()/obj.N + (0:obj.N-1)/obj.N;

            for n = 1:obj.N
                idx = find(u(n) < C, 1, 'first');

                if isempty(idx)
                    idx = obj.N;
                end
                post(:,n) = pre(:,idx);
            end

        end

        function w_n = normalize(obj, w)
            t = sum(w);
            if t == 0
                % Fall back to uniform so resample keeps the whole set instead
                % of collapsing onto particle N (cdf is all zeros otherwise)
                obj.debug_print("unable to normalize PF weights, all weights == 0, using uniform")
                w_n = ones(size(w)) / numel(w);
                return;
            end

            w_n = w./t;

        end

        %%% Pack a run's output plus the scenario it ran on into the results
        %%% struct visualize.plot_TBD understands. Anything missing (no
        %%% scenario, or a bare cell of signals) is simply left empty.
        function res = to_vis_results(obj, R, scenario)
            res = visualize.results_template();
            res.particles = R.particles;
            res.pE        = R.pE;
            res.pEthresh  = obj.pEthresh;
            res.t         = (0:R.K-1) * obj.dt;

            if isstruct(scenario)
                res.signals = cat(3, scenario.signal{:});
                res.truth   = scenario.truth;
                res.Etruth  = scenario.exist;
                if isfield(scenario, 'tvec') && numel(scenario.tvec) == R.K
                    res.t = scenario.tvec;
                end
            elseif iscell(scenario) && ~isempty(scenario)
                res.signals = cat(3, scenario{:});
            end
        end

    end
end