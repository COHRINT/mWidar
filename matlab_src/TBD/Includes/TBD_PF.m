
%%% Full TBD filter implementation
%%% Public entry points are run() and show()
%%% TBD_PF.run(scenario) will output a results struct, TBD_PF.show(R, scenario) plots it
%%%
%%% Conventions
%%%   - Particle state is [px; vx; py; vy; I] in METERS, matching the scenario
%%%     truth, plus the target intensity I. Particle sets carry existence E as
%%%     a 6th row: [px; vx; py; vy; I; E].
%%%   - Signals are indexed z(row, col) = z(y, x), matching simulator / visualize.
%%%   - Sigma and pDist are in pixels; Sigma is scaled by dx inside the likelihood.
%%%   - Weights are always normalized to sum to 1, and R.particles(:,:,k) is
%%%     always paired with R.weights(:,k). After a resample that pairing is a
%%%     uniform 1/N; between resamples it is not, so anything read off the
%%%     cloud has to be weighted (see visualize.particle_estimate).

classdef TBD_PF < TBD

    properties
        %%% Resampling policy, set by run()
        bootstrap          % true  -> resample every step (classic bootstrap PF)
                           % false -> SIS, resample only when ESS/N drops
        RESAMPLE_THRESHOLD % ESS/N at or below which the adaptive filter resamples
    end

    methods
        %%% RUN TBD_PF
        %%% Main entry point. Takes the scenario struct from environment.setup()
        %%% (or just a 1 x K cell of npx x npx signals) and runs the bootstrap
        %%% PF over every frame.
        %%%
        %%%   R = pf.run(scenario)
        %%%   R = pf.run(scenario, 'Bootstrap', false, 'ESS', 0.5)
        %%%
        %%% Options
        %%%   'Bootstrap' true resamples every step (the classic bootstrap PF,
        %%%               weights uniform by construction). false runs SIS with
        %%%               adaptive resampling: weights are carried forward and
        %%%               only flattened once the cloud degenerates.
        %%%               Default false.
        %%%   'ESS'       resampling threshold as a FRACTION of N, in (0, 1].
        %%%               Resample when ESS/N <= this. Default 0.5, the usual
        %%%               N/2 rule. Ignored when 'Bootstrap' is true.
        %%%
        %%%   R.particles   6 x N x K, posterior particle set [px; vx; py; vy; I; E] at each k
        %%%   R.weights     N x K, normalized posterior weight of each particle
        %%%   R.ess         1 x K, ESS of the weights BEFORE the resample decision
        %%%   R.resampled   1 x K logical, frames where the filter resampled
        %%%   R.pE          1 x K, existence probability = weight mass on E = 1
        %%%   R.exist       1 x K logical, track declared where pE > pEthresh
        %%%   R.N, R.K      # of particles and # of timesteps run
        %%%
        %%% Metrics can be appended to R later, everything here is just the
        %%% raw filter output.
        function R = run(obj, scenario, varargin)

            %%% Resampling policy. Resampling every step throws information
            %%% away whenever the frame was uninformative, so the default is to
            %%% only do it once the weights have actually degenerated.
            p = inputParser;
            addParameter(p, 'Bootstrap', false, @islogical);
            addParameter(p, 'ESS', 0.5, @(x) isscalar(x) && isnumeric(x) && x > 0 && x <= 1);
            parse(p, varargin{:});

            %%% TBD_PF is a value class, so these land on run()'s own copy of
            %%% obj and never reach the caller's filter object. timestep() sees
            %%% them (it is called on that copy), but show() has to read the
            %%% policy back off R, which is why run() records it there.
            obj.bootstrap = p.Results.Bootstrap;
            obj.RESAMPLE_THRESHOLD = p.Results.ESS;

            if obj.bootstrap && ~ismember('ESS', p.UsingDefaults)
                warning('TBD_PF:essIgnored', ...
                    ['''Bootstrap'' is true, so ''ESS'' = %.3g is ignored and every ', ...
                     'step resamples. Pass ''Bootstrap'', false to use the threshold.'], ...
                    obj.RESAMPLE_THRESHOLD);
            end

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

            if obj.bootstrap
                mode = "bootstrap (resample every step)";
            else
                mode = sprintf("adaptive (resample when ESS <= %.2f * N = %.0f)", ...
                    obj.RESAMPLE_THRESHOLD, obj.RESAMPLE_THRESHOLD * obj.N);
            end
            obj.debug_print(sprintf("run: K=%d frames, N=%d particles, frame size %s, %s", ...
                K, obj.N, mat2str(size(z{1})), mode));
            tStart = tic;

            %%% Initialize, every particle dead with undefined state and an
            %%% equal share of the weight
            prior   = [nan(5, obj.N); false(1, obj.N)];
            w_prior = ones(1, obj.N) / obj.N;

            R = struct();
            R.N         = obj.N;
            R.K         = K;
            R.particles = nan(6, obj.N, K);
            R.weights   = nan(obj.N, K);
            R.ess       = nan(1, K);
            R.resampled = false(1, K);
            R.pE        = nan(1, K);
            R.exist     = false(1, K);
            R.bootstrap = obj.bootstrap;
            R.essThresh = obj.RESAMPLE_THRESHOLD;

            %%% Filter
            for k = 1:K
                [post, w, ess, didResample] = obj.timestep(prior, w_prior, z{k});

                R.particles(:,:,k) = post;
                R.weights(:,k)     = w(:);
                R.ess(k)           = ess;
                R.resampled(k)     = didResample;
                %%% Existence is the weight mass sitting on E = 1, not the
                %%% particle count: between resamples the two disagree.
                R.pE(k)    = sum(w(post(6,:) ~= 0));
                R.exist(k) = R.pE(k) > obj.pEthresh;

                obj.debug_print(sprintf("k=%d/%d  pE=%.3f  exist=%d  ESS/N=%.3f  resampled=%d", ...
                    k, K, R.pE(k), R.exist(k), ess / obj.N, didResample));

                prior   = post; % Carry posterior forward as next step's prior
                w_prior = w;    % ... and its weights, that is the whole point
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
            obj.debug_print(sprintf("run: resampled %d/%d frames, median ESS/N = %.3f", ...
                nnz(R.resampled), K, median(R.ess, 'omitnan') / obj.N));

        end

        %%% Quick look at a run. Draws the vis.plot_TBD dashboard (track over
        %%% the energy map, existence, per-axis position, error, particle
        %%% count, and measurement SNR vs time when the scenario recorded it)
        %%% and optionally plays the frame-by-frame history with the
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
        %%%   'WeightColormap' colormap shading the particles by weight in the
        %%%               animation (default 'gray'); '' draws them in the flat
        %%%               particle color instead
        %%%   'WeightScale'    'log' (default) or 'linear' scale for that shading.
        %%%               The likelihood ratio is a product over a pixel window,
        %%%               so on a linear scale one particle is white and the
        %%%               other N-1 are black.
        %%%   'Intensity' with Animate, put the weighted distribution of the
        %%%               particles' target intensity I in a second panel next
        %%%               to the scene, so <base>_history.gif carries both.
        %%%               Bars are posterior weight, not particle count, and
        %%%               the E = 0 mass gets a grey bar of its own on the
        %%%               left, so everything on screen sums to 1 and the live
        %%%               bars sum to pE. Default true.
        %%%   'IntensityBins'  # of bins over [I_min, I_max] (default 24)
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
            addParameter(p, 'WeightColormap', 'gray');
            addParameter(p, 'WeightScale', 'log');
            addParameter(p, 'Intensity', true, @islogical);
            addParameter(p, 'IntensityBins', 24, @(x) isscalar(x) && isnumeric(x) && x >= 2);
            parse(p, varargin{:});
            opt = p.Results;

            figAnim = [];

            res = obj.to_vis_results(R, scenario);

            ttl = opt.Title;
            if isempty(ttl)
                ttl = sprintf('TBD-PF: N = %d, K = %d, Pb = %.3g, Ps = %.3g, pEthresh = %.2f, %s', ...
                              R.N, R.K, obj.Pb, obj.Ps, obj.pEthresh, obj.resample_tag(R));
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
                'ESSThresh', obj.essThresh_of(R), ...
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
                obj.debug_print(sprintf("show: animating %d frames at %g fps, intensity panel=%d", ...
                    R.K, opt.FPS, opt.Intensity));
                figAnim = obj.vis.animate_time_history(res.signals, ...
                    'Truth',     res.truth, ...
                    'Particles', R.particles, ...
                    'Weights',   res.weights, ...
                    'pE',        R.pE, ...
                    'SNR',       res.snr, ...
                    'DataUnits', 'meters', ...
                    'Trail',     opt.Trail, ...
                    'FPS',       opt.FPS, ...
                    'WeightColormap', opt.WeightColormap, ...
                    'WeightScale',    opt.WeightScale, ...
                    'Intensity',     opt.Intensity, ...
                    'IntensityRow',  5, ...
                    'IntensityLim',  [obj.I_min, obj.I_max], ...
                    'IntensityBins', opt.IntensityBins, ...
                    'Title',     ttl, ...
                    'Save',      animPath);
            end

        end
    end

    methods (Hidden)

        function w = importance_weights(obj, Y, z)

            X = Y(1:5);
            E = Y(6);

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
                        w = w*obj.guass_likelihood([px py], pix, idx, z, X(5));
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
        function l = guass_likelihood(obj, pos, pix, idx, z, I)
            px = pos(1);
            py = pos(2);

            ii = pix(1);
            jj = pix(2);

            sig = obj.Sigma * obj.dx; % Sigma is given in pixels
            hh = I * exp( - ((ii -px)^2 + ((jj -py)^2))/ (2 * sig^2));
            zz = z(idx(2), idx(1)); % rows = y, cols = x
            l = exp(-(hh*(hh - 2*zz))/(2 * obj.NoiseSTD^2));
        end

        %%% Run through one timestep of TBD_PF.
        %%%
        %%%   [post, w, ESS, didResample] = obj.timestep(prior, w_prior, z)
        %%%
        %%% prior / post are 6 x N particle sets, w_prior / w the matching
        %%% normalized weights. ESS is measured on w BEFORE the resample
        %%% decision, so it is the number that drove it; didResample says what
        %%% the filter decided.
        function [post, w, ESS, didResample] = timestep(obj, prior, w_prior, z)
            x_minus = prior(1:5,:);
            E_minus = prior(6,:);

            if nargin < 4
                error('TBD_PF:timestep', ...
                    'timestep now takes the prior weights: timestep(prior, w_prior, z)');
            end
            w_prior = reshape(w_prior, 1, []);

            z = obj.preprocess(z);

            % Regime transition
            E_plus = obj.RT(E_minus);
            x_plus = nan(5,obj.N);
            L = zeros(1,obj.N);   % this frame's likelihood ratio, per particle
            Y = nan(6,obj.N);

            for n = 1:obj.N

                % New born particles
                if E_plus(n) && ~E_minus(n)
                    x_plus(:,n) = obj.sample_new(z);
                elseif E_plus(n) && E_minus(n)
                    x_plus(:,n) = obj.dynamics(x_minus(:,n));
                end

                % Evaluate importance weights
                Y(:, n) = [x_plus(:,n); E_plus(n)];
                L(n) = obj.importance_weights(Y(:,n),z);
            end

            %%% Sequential importance sampling: the new weight is the old
            %%% weight times this frame's likelihood ratio. A resample flattens
            %%% w_prior back to 1/N, so whenever the previous step resampled
            %%% this collapses to the bootstrap weights and the two filters
            %%% agree. Skipping the multiply is what breaks the non-bootstrap
            %%% path, the evidence from every un-resampled frame is lost.
            w_tilde = w_prior .* L;

            w   = obj.normalize(w_tilde);
            ESS = obj.get_ESS(w);

            didResample = obj.bootstrap || (ESS <= obj.RESAMPLE_THRESHOLD * obj.N);
            if didResample
                [post, w] = obj.resample(Y, w);
            else
                post = Y;
            end

            if obj.debug
                born  = nnz(E_plus & ~E_minus);
                kept  = nnz(E_plus &  E_minus);
                died  = nnz(~E_plus & E_minus);
                w0    = nnz(E_plus & w_tilde == 0); % alive but zero weight (OOB or likelihood underflow)
                birthMode = "signal";
                if ~any(z(:) > obj.gamma)
                    birthMode = "uniform (no pixels above gamma)";
                end
                obj.debug_print(sprintf("timestep: born=%d kept=%d died=%d alive=%d w0=%d | L max=%.3g ESS/N=%.3f resampled=%d | birth=%s", ...
                    born, kept, died, nnz(E_plus), w0, max(L), ESS / obj.N, didResample, birthMode));
            end

        end

        %%% Effective sample size, Kong's 1/sum(w^2) written scale-free so it
        %%% works on normalized or unnormalized weights. Ranges over [1, N].
        function ESS = get_ESS(~, w)
            s1 = sum(w);
            s2 = sum(w.^2);
            if ~(s2 > 0) || ~isfinite(s1)
                ESS = NaN; % every weight zero, caller treats this as "no info"
                return
            end
            ESS = s1^2 / s2;
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
            I = obj.I_min + (obj.I_max - obj.I_min) * rand();
            X = [px; vx; py; vy; I]; % Random new alive particle
        end


        function X_plus = dynamics(obj, X_minus)
            X_plus = obj.F * X_minus + mvnrnd(zeros(5,1), obj.Q)';
            % I is a random walk, keep it in the range births are drawn from
            X_plus(5) = min(max(X_plus(5), obj.I_min), obj.I_max);
        end

        %%% Measurement preprocessing, applied once per frame before births
        %%% and the likelihood. The likelihood assumes z = h + zero-mean
        %%% noise, but the simulator min-max normalizes each frame, so with a
        %%% target present the background sits near 0.8 and nearly every
        %%% pixel clears gamma. Subtracting the frame median and rescaling the
        %%% peak back to 1 puts the background at ~0 and keeps I on [0, 1].
        %%% A blank frame (no target, no noise) is all zeros and passes
        %%% through unchanged.
        function z = preprocess(obj, z)
            if strcmp(obj.Background, 'none')
                return
            end
            b  = median(z(:));
            pk = max(z(:)) - b;
            if pk > 0
                z = (z - b) / pk;
            end
        end

        %%% Systematic resampling. w must already be normalized. The
        %%% returned weights are flat 1/N, which is what lets the SIS recursion
        %%% in timestep() reduce to the bootstrap weights on the next frame.
        function [post, w] = resample(obj, pre, w)
            post = nan(6,obj.N);

            C = cumsum(w);
            C(end) = 1; % cumsum round-off can leave C(end) just under u(N)
            u = (rand() + (0:obj.N-1)) / obj.N;

            %%% u is sorted, so one pass over the cdf is enough. A find() per
            %%% particle is O(N^2) and shows up badly at N = 5000.
            i = 1;
            for n = 1:obj.N
                while u(n) > C(i) && i < obj.N
                    i = i + 1;
                end
                post(:,n) = pre(:,i);
            end

            w = ones(1,obj.N) / obj.N;
        end

        function w_n = normalize(obj, w)
            w = reshape(double(w), 1, []);

            %%% The likelihood ratio is a product over a (2*pDist+1)^2 window of
            %%% exponentials, so it can overflow on a strong frame. Clamp the
            %%% infinities, drop the NaNs, then divide by the max before the sum
            %%% so a large-but-finite set cannot overflow on the way to 1.
            w(isinf(w) & w > 0) = realmax;
            w(~isfinite(w)) = 0;

            m = max(w);
            if m > 0
                w = w / m;
            end

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

        %%% Which resampling policy produced R, for figure titles. Reads it
        %%% off R rather than obj so a reloaded result still labels correctly.
        function tag = resample_tag(obj, R)
            boot = obj.bootstrap;
            if isfield(R, 'bootstrap')
                boot = R.bootstrap;
            end
            if boot
                tag = 'bootstrap';
            else
                tag = sprintf('resample at ESS/N <= %.2f', obj.essThresh_of(R));
            end
        end

        function thr = essThresh_of(obj, R)
            thr = obj.RESAMPLE_THRESHOLD;
            if isfield(R, 'essThresh')
                thr = R.essThresh;
            end
            if isempty(thr)
                thr = 0.5;
            end
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

            %%% Runs produced before the adaptive resampler existed have none
            %%% of these; vis drops the panels that need them.
            if isfield(R, 'weights')
                res.weights = R.weights;
            end
            if isfield(R, 'ess')
                res.ess = R.ess;
            end
            if isfield(R, 'resampled')
                res.resampled = R.resampled;
            end

            if isstruct(scenario)
                res.signals = cat(3, scenario.signal{:});
                res.truth   = scenario.truth;
                res.Etruth  = scenario.exist;
                % Scenarios built before SNR was recorded have no field; vis
                % drops the SNR panel and labels when res.snr is empty.
                if isfield(scenario, 'snr')
                    res.snr = scenario.snr;
                end
                if isfield(scenario, 'tvec') && numel(scenario.tvec) == R.K
                    res.t = scenario.tvec;
                end
            elseif iscell(scenario) && ~isempty(scenario)
                res.signals = cat(3, scenario{:});
            end
        end

    end
end