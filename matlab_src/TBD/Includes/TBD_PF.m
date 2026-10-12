
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

        %%Forward model matrices, used for model based likelihood
        RECOVERY_FILEPATH % G -> Recovery matrix
        SAMPLING_FILEPATH; % M -> sampling matrix
        TARGETCONTRIBUTION_FILEPATH

        M % sampling
        G % recovert
        TC % TargetContribution
        C % will be computed per measurment

        %%% Choose which likelihood function to use
        %%% Options: gauss for the gaussian psf assumption
        %%% model for using the mWidar model
        %%% maybe more someday, comparing these would be cool
        likelihood
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
            addParameter(p, 'Likelihood', 'model', @(x) ismember(x, {'gauss', 'model'}))
            addParameter(p, 'Var', 1)
            parse(p, varargin{:});

            %%% TBD_PF is a value class, so these land on run()'s own copy of
            %%% obj and never reach the caller's filter object. timestep() sees
            %%% them (it is called on that copy), but show() has to read the
            %%% policy back off R, which is why run() records it there.
            obj.bootstrap = p.Results.Bootstrap;
            obj.RESAMPLE_THRESHOLD = p.Results.ESS;
            obj.likelihood = p.Results.Likelihood;
            var = p.Results.Var;

            if var ~= 0
                obj.NoiseSTD = sqrt(var);
            else
                obj.NoiseSTD = 0.1; % some small value
            end
            %%% Filepaths
            obj.RECOVERY_FILEPATH = "supplemental/recovery.mat";
            obj.SAMPLING_FILEPATH = "supplemental/sampling.mat";
            obj.TARGETCONTRIBUTION_FILEPATH = "TBD/supp/TargetContribution.mat";

            if strcmp(obj.likelihood, 'model')
                ds = load(obj.RECOVERY_FILEPATH);
                obj.G = ds.G;

                ds = load(obj.SAMPLING_FILEPATH);
                obj.M = ds.M;

                ds = load(obj.TARGETCONTRIBUTION_FILEPATH);
                obj.TC = ds.L;
            end

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
            R.likelihood = obj.likelihood;
            %%% 'model' only: the per-frame profile log-LR map the particles
            %%% were scored against, stored RELATIVE to its own maximum so
            %%% single precision keeps full resolution near the peak (the raw
            %%% values are ~1e7, which single would round to ~4 nats).
            %%% R.llrMax(k) holds the max that was subtracted.
            R.llr    = [];
            R.llrMax = [];
            if strcmp(obj.likelihood, 'model')
                R.llr    = nan(obj.npx, obj.npx, K, 'single');
                R.llrMax = nan(1, K);
            end
            %%% Where the particles born on each frame landed (E went 0 -> 1),
            %%% taken before that frame's resample: R.births{k} is 2 x nBorn
            %%% [px; py] in meters. Plotting only: lets the animation show the
            %%% birth process, which the posterior particles mostly can't
            %%% (a newborn rarely survives the resample).
            R.births = cell(1, K);

            %%% Filter
            for k = 1:K
                [post, w, ess, didResample, llrMap, births] = obj.timestep(prior, w_prior, z{k});
                if ~isempty(llrMap)
                    R.llrMax(k)   = max(llrMap(:));
                    R.llr(:,:,k)  = single(llrMap - R.llrMax(k));
                end
                R.births{k} = births;

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
        %%% the energy map, existence, per-axis position, error, occupied
        %%% cells, target intensity vs time, ESS, and matched-filter SNR vs
        %%% time when the scenario recorded it) and optionally plays the
        %%% frame-by-frame history with the particle cloud. Everything is
        %%% delegated to vis, same as environment.show.
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
        %%%   'LikelihoodMap'  with Animate and a 'model' run, put the per-frame
        %%%               profile log-LR map (R.llr) in a second panel beside
        %%%               the raw frame, same overlays on both. Drawn as
        %%%               -log10(1 + dLL), dLL = max LL - LL, so the best cell
        %%%               is 0 and each step down the colorbar is a factor of
        %%%               10 in log-LR deficit. Default true.
        %%%   'ShowBirths'  draw the E = 1 particles that carry zero weight as
        %%%               small grey dots, and every particle born this frame
        %%%               (R.births, taken before the resample) as a white
        %%%               diamond, so the birth process is visible. Default
        %%%               true. Newborns almost always carry exactly zero
        %%%               weight (their log-LR is thousands of nats below the
        %%%               cloud) and are dropped by the next resample, so
        %%%               without this they never show.
        %%%   'WeightColormap' colormap shading the particles by weight in the
        %%%               animation. Default '' (flat particle color): with the
        %%%               cell-constant 'model' likelihood every particle in a
        %%%               cell shares a weight, and after a resample they are
        %%%               all 1/N, so the shading carries almost nothing.
        %%%   'WeightScale'    'log' (default) or 'linear' scale for that shading.
        %%%   'Intensity' with Animate, put the weighted distribution of the
        %%%               particles' target intensity I in a second panel next
        %%%               to the scene. Default false: with 'model' I is the
        %%%               fitted C/E, so the panel is one bar at ~1. Still
        %%%               useful for 'gauss', where I is sampled.
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
            addParameter(p, 'LikelihoodMap', true, @islogical);
            addParameter(p, 'ShowBirths', true, @islogical);
            addParameter(p, 'WeightColormap', '');
            addParameter(p, 'WeightScale', 'log');
            addParameter(p, 'Intensity', false, @islogical);
            addParameter(p, 'IntensityBins', 24, @(x) isscalar(x) && isnumeric(x) && x >= 2);
            parse(p, varargin{:});
            opt = p.Results;

            figAnim = [];

            res = obj.to_vis_results(R, scenario);

            ttl = opt.Title;
            if isempty(ttl)
                lik = '';
                if isfield(R, 'likelihood') && ~isempty(R.likelihood)
                    lik = [', ' char(R.likelihood) ' likelihood'];
                end
                ttl = sprintf('TBD-PF: N = %d, K = %d, Pb = %.3g, Ps = %.3g, pEthresh = %.2f, %s%s', ...
                              R.N, R.K, obj.Pb, obj.Ps, obj.pEthresh, obj.resample_tag(R), lik);
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
                %%% Right-hand panel: the likelihood map, when the run has one
                second = {};
                if opt.LikelihoodMap
                    [D, lo] = obj.llr_display(R);
                    if isempty(D)
                        obj.debug_print("show: no likelihood map in R (not a 'model' run), raw frame only");
                    else
                        second = {'Signals2',    D, ...
                                  'CLim2',       [lo 0], ...
                                  'Colormap2',   'parula', ...
                                  'ImageLabel2', '$-\log_{10}(1 + \max LL - LL)$', ...
                                  'PanelTitles', {'raw frame', 'profile log-LR map'}};
                    end
                end
                births = {};
                if opt.ShowBirths
                    births = {'ShowZeroWeight', true};
                    if isfield(R, 'births') && ~isempty(R.births)
                        births = [births, {'Births', R.births}];
                    end
                end
                obj.debug_print(sprintf("show: animating %d frames at %g fps, likelihood panel=%d, births=%d", ...
                    R.K, opt.FPS, ~isempty(second), opt.ShowBirths));

                %%% Raw frames are unnormalized, and the pedestal level swings
                %%% ~3x with target range, so one run-wide color scale paints
                %%% most frames a single flat color. Per-frame limits are
                %%% display only; the filter still sees the raw frame.
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
                    'Title',     ttl, ...
                    'CLim',          'frame', ...
                    'ImageLabel',    'raw signal (per-frame limits)', ...
                    second{:}, births{:}, ...
                    'Intensity',     opt.Intensity, ...
                    'IntensityRow',  5, ...
                    'IntensityLim',  [obj.I_min, obj.I_max], ...
                    'IntensityBins', opt.IntensityBins, ...
                    'Save',          animPath);
            end

        end
    end

    methods (Hidden)

        function [logL, Ihat] = importance_weights(obj, Y, z)

            X = Y(1:5);
            E = Y(6);
            logL = 0; Ihat = NaN;
            if ~E, return; end % skip non existing


            px = X(1);
            py = X(3);

            if px > obj.max_x || px < obj.min_x
                logL = -Inf;
                return
            end

            % Going to set weights for particles below y = 0.25 to 0 to avoid sensor array
            if py > obj.max_y || py < 0.25
                logL = -Inf;
                return
            end

            % Particle's pixel index (meters -> index, grid is uniform so this is exact)
            ix = round((px - obj.xgrid(1)) / obj.dx) + 1;
            iy = round((py - obj.ygrid(1)) / obj.dy) + 1;

            if strcmp(obj.likelihood, 'model')
                [logL, Ihat] = obj.model_likelihood(iy,ix, X(5));
                return
            end

            % Neighborhood window, clamped to valid pixel indices
            xvec = max(1, ix-obj.pDist):min(obj.npx, ix+obj.pDist);
            yvec = max(1, iy-obj.pDist):min(obj.npx, iy+obj.pDist);

            % Same window in meters, for the PSF distance
            xm = obj.xgrid(xvec);
            ym = obj.ygrid(yvec);
            
            
            for i = 1:numel(xvec)
                for j = 1:numel(yvec)
                    logL = logL + obj.guass_likelihood([px py], [xm(i), ym(j)], ...
                                                    [xvec(i), yvec(j)], z, X(5));
                end
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
            l = (-(hh*(hh - 2*zz))/(2 * obj.NoiseSTD^2));
        end

        function [l, Ihat] = model_likelihood(obj, r, c, I) % doesn't need z since C is computed already

            Cv = obj.C(r,c);
            TCv = obj.TC(r,c);
            Ihat = min(max(Cv / TCv, obj.I_min), obj.I_max);
            l = (Cv - 0.5*TCv) / obj.NoiseSTD^2;   % log domain


        end
        %%% Profile log-LR for EVERY cell at once, from this frame's C: the
        %%% same expression model_likelihood evaluates per particle, so the
        %%% map is exactly what the particles are scored against. Rows below
        %%% y = 0.25 m are NaN, matching the region importance_weights zeroes.
        %%% Only used for plotting (R.llr); the filter itself still calls
        %%% model_likelihood per particle.
        function LL = llr_map(obj)
            Ihat = min(max(obj.C ./ obj.TC, obj.I_min), obj.I_max);
            LL = (Ihat .* obj.C - 0.5 * Ihat.^2 .* obj.TC) / obj.NoiseSTD^2;
            LL(obj.ygrid(:) < 0.25, :) = NaN;   % rows = y
        end

        %%% Run through one timestep of TBD_PF.
        %%%
        %%%   [post, w, ESS, didResample, llrMap, births] = obj.timestep(prior, w_prior, z)
        %%%
        %%% prior / post are 6 x N particle sets, w_prior / w the matching
        %%% normalized weights. ESS is measured on w BEFORE the resample
        %%% decision, so it is the number that drove it; didResample says what
        %%% the filter decided. llrMap is the npx x npx profile log-LR map
        %%% (llr_map) for 'model', [] for 'gauss'. births is 2 x nBorn, the
        %%% [px; py] of every particle born this frame, taken before the
        %%% resample. llrMap and births are for plotting only.
        function [post, w, ESS, didResample, llrMap, births] = timestep(obj, prior, w_prior, z)
            llrMap = [];
            x_minus = prior(1:5,:);
            E_minus = prior(6,:);

            if nargin < 4
                error('TBD_PF:timestep', ...
                    'timestep now takes the prior weights: timestep(prior, w_prior, z)');
            end
            w_prior = reshape(w_prior, 1, []);
            

            % Regime transition
            E_plus = obj.RT(E_minus);
            x_plus = nan(5,obj.N);
            % L = zeros(1,obj.N);   % this frame's likelihood ratio, per particle
            Y = nan(6,obj.N);

            % compute c_k
            % only if likelihood is model
            if strcmp(obj.likelihood, 'model')
                zt = z.'; % row major ordering
                c = obj.M' * (obj.G * zt(:));
                obj.C = reshape(c, obj.npx, obj.npx).';
                llrMap = obj.llr_map();
            end

            [birthIdx, nValid] = obj.birth_cells(z);
            % Want to check below with book:
            logBirthCorr = log(numel(birthIdx) / nValid); 

            logL = zeros(1, obj.N);
            for n = 1:obj.N

                % New born particles
                if E_plus(n) && ~E_minus(n)
                    x_plus(:,n) = obj.sample_new(birthIdx);
                elseif E_plus(n) && E_minus(n)
                    x_plus(:,n) = obj.dynamics(x_minus(:,n));
                end

                % Evaluate importance weights
                Y(:, n) = [x_plus(:,n); E_plus(n)];
                [logL(n), ~] = obj.importance_weights(Y(:,n),z);

                %if ~isnan(Ihat), Y(5, n) = Ihat; end   % keep I in the state = fitted intensity
            end

            %%% Update
            w = obj.normalize_log(log(w_prior) + logL);
            ESS = obj.get_ESS(w);

            %%% Where this frame's newborns landed, BEFORE the resample: a
            %%% newborn almost never survives one, so the posterior alone
            %%% can't show the birth process. Plotting only.
            births = Y([1 3], E_plus & ~E_minus);   % 2 x nBorn [px; py], meters

            didResample = obj.bootstrap || (ESS <= obj.RESAMPLE_THRESHOLD * obj.N);
            if didResample
                [post, w] = obj.resample(Y, w);
            else
                post = Y;
            end

            if obj.debug
                nBorn = nnz(E_plus & ~E_minus);
                kept  = nnz(E_plus &  E_minus);
                died  = nnz(~E_plus & E_minus);
                w0    = nnz(E_plus & logL == -Inf); % alive but zero weight (OOB or likelihood underflow)
                birthMode = sprintf("top %.0f%% of valid cells (%d)", 100*(1 - obj.gamma), numel(birthIdx));
                obj.debug_print(sprintf("timestep: born=%d kept=%d died=%d alive=%d w0=%d | L max=%.3g ESS/N=%.3f resampled=%d | birth=%s", ...
                    nBorn, kept, died, nnz(E_plus), w0, max(logL), ESS / obj.N, didResample, birthMode));
            end

        end

        function w_n = normalize_log(obj, logw)
            logw = reshape(double(logw), 1, []);
            logw(isnan(logw)) = -Inf;
            m = max(logw);
            if ~isfinite(m)
                obj.debug_print("all log weights -Inf, using uniform")
                w_n = ones(size(logw)) / numel(logw);
                return
            end
            w = exp(logw - m);
            w_n = w / sum(w);
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

        % the brightest (1 - gamma) fraction of the valid pixels in this frame
        % rows below y = 0.25 are excluded always

        function [idx, nValid] = birth_cells(obj,z)
            validRow = obj.ygrid(:) >= 0.25;
            zv = z;
            zv(~validRow,:) = -Inf; % never chosen
            nValid = nnz(validRow)*obj.npx;
            nKeep = max(1, ceil((1-obj.gamma) * nValid));
            [~, idx] = maxk(zv(:), nKeep);
        end
        %%% Per particle

        function X = sample_new(obj, idx)
            % Birth proposal: pick a pixel above the threshold uniformly at
            % random and place the particle at its center in meters.
            % z is indexed z(row, col) = z(y, x), matching simulator.

            %%% Now that we are not normalizing, gamma should act as a percentange

            [r, c] = ind2sub([obj.npx obj.npx], idx(randi(numel(idx))));
            px = obj.xgrid(c);
            py = obj.ygrid(r);
            

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


        %%% Systematic resampling. w must already be normalized. The
        %%% returned weights are flat 1/N, which is what lets the SIS recursion
        %%% in timestep() reduce to the bootstrap weights on the next frame.
        function [post, w] = resample(obj, pre, w)
            post = nan(6,obj.N);

            Cdf = cumsum(w);
            Cdf(end) = 1; % cumsum round-off can leave C(end) just under u(N)
            u = (rand() + (0:obj.N-1)) / obj.N;

            %%% u is sorted, so one pass over the cdf is enough. A find() per
            %%% particle is O(N^2) and shows up badly at N = 5000.
            i = 1;
            for n = 1:obj.N
                while u(n) > Cdf(i) && i < obj.N
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

        %%% Display version of the per-frame profile log-LR map (R.llr), for
        %%% the right-hand panel of the animation.
        %%%
        %%% R.llr(:,:,k) = LL - max(LL) <= 0. The deficit spans ~0 to ~1e7
        %%% nats in one frame (neighbor cells ~1e2-1e3, mirror cell ~1e3,
        %%% far cells ~1e7), so it is drawn as -log10(1 + deficit): 0 at the
        %%% best cell, -1 at a 9-nat deficit, -3 at ~1e3, and so on. lo is the
        %%% floor of the color scale, fixed over the run so frames compare.
        %%% Excluded rows (y < 0.25 m) are NaN in R.llr and set to lo.
        %%% D is [] when R has no map (a 'gauss' run).
        function [D, lo] = llr_display(~, R)
            D = [];
            lo = -1;
            if ~isfield(R, 'llr') || isempty(R.llr)
                return
            end

            %%% Deficit >= 0. Clamp with a mask, not max(0, .): max ignores
            %%% NaN, which would turn the excluded rows into "best cell".
            dLL = -double(R.llr);
            dLL(dLL < 0) = 0;
            D = -log10(1 + dLL);
            m = min(D(isfinite(D)));
            if ~isempty(m) && m < 0
                lo = floor(m);
            end
            D(~isfinite(D)) = lo;
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