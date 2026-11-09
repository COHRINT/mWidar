
%%% Full TBD filter implementation
%%% Public entry points are run() and show()
%%% TBD_PF.run(scenario, env) will output a results struct, TBD_PF.show(R, scenario) plots it
%%%
%%% Conventions
%%%   - Particle state is [px; vx; py; vy] in METERS, matching the scenario
%%%     truth. Particle sets carry existence E as a 5th row:
%%%     [px; vx; py; vy; E].
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

        %%% Plotting only, set by run('DebugPlots', ...). When true, timestep
        %%% also returns a snapshot of its intermediate variables for the
        %%% visualize.debug_* plots. It never changes what the filter computes.
        debugPlots = false

        %%Forward model matrices, used for model based likelihood
        RECOVERY_FILEPATH % G -> Recovery matrix
        SAMPLING_FILEPATH; % M -> sampling matrix
        TARGETCONTRIBUTION_FILEPATH

        M % sampling
        G % recovery
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
        %%% (or just a 1 x K cell of npx x npx signals) and runs the PF over
        %%% every frame (SIS with adaptive resampling unless 'Bootstrap').
        %%%  
        %%%   R = pf.run(scenario, env)
        %%%   R = pf.run(scenario, env, 'Bootstrap', false, 'ESS', 0.5)
        %%%
        %%% Options
        %%%   'Bootstrap' true resamples every step (the classic bootstrap PF,
        %%%               weights uniform by construction). false runs SIS with
        %%%               adaptive resampling: weights are carried forward and
        %%%               only flattened once the cloud degenerates.
        %%%               Default false.
        %%%   'ESS'       resampling threshold as a FRACTION of N, in (0, 1].
        %%%               Resample when ESS/N <= this. Default 0.005, far
        %%%               below the usual N/2 rule so newborns get a chance
        %%%               before the cloud is resampled away. Ignored when
        %%%               'Bootstrap' is true.
        %%%   'DebugPlots'  draw step-by-step debug figures (existence
        %%%               transitions, births, dynamics, importance weights,
        %%%               normalization, resampling) plus a whole-run summary.
        %%%               Plotting only: R is identical with it on or off.
        %%%               Default false. See visualize.debug_step.
        %%%   'DebugFrames' frames to draw the per-step figures for. Default []
        %%%               = every frame. The summary always covers every frame.
        %%%   'DebugName'   run sub folder, default a timestamp
        %%%   'DebugDir'    root folder, default TBD/test/figs/debug. Figures
        %%%               land in <DebugDir>/<DebugName>/<NN_stage>/k###_<stage>.png
        %%%   'DebugVisible' leave the debug figures open on screen (default
        %%%               false: drawn off screen, saved, then closed)
        %%%
        %%%   R.particles  5 x N x K, posterior particle set [px; vx; py; vy; E] at each k
        %%%   R.weights     N x K, normalized posterior weight of each particle
        %%%   R.weightsPre  N x K, normalized weights before the resample
        %%%                 decision (plotting only, see show 'AnimateWeights')
        %%%   R.ess         1 x K, ESS of the weights BEFORE the resample decision
        %%%   R.resampled   1 x K logical, frames where the filter resampled
        %%%   R.pE          1 x K, existence probability = weight mass on E = 1
        %%%   R.exist       1 x K logical, track declared where pE > pEthresh
        %%%   R.N, R.K      # of particles and # of timesteps run
        %%%
        %%% Metrics can be appended to R later, everything here is just the
        %%% raw filter output.
        function R = run(obj, scenario, env, varargin)

            %%% Resampling policy. Resampling every step throws information
            %%% away whenever the frame was uninformative, so the default is to
            %%% only do it once the weights have actually degenerated.
            p = inputParser;
            addParameter(p, 'Bootstrap', false, @islogical);
            addParameter(p, 'ESS', 0.3, @(x) isscalar(x) && isnumeric(x) && x > 0 && x <= 1); % Particles are often tightly grouped, small ESS allows for births to get a chance
            addParameter(p, 'Likelihood', 'model', @(x) ismember(x, {'gauss', 'model'}))
            addParameter(p, 'NoiseSTD', 1000);
            addParameter(p, 'DebugPlots', false, @islogical);
            addParameter(p, 'DebugFrames', [], @(x) isempty(x) || (isnumeric(x) && all(x >= 1)));
            addParameter(p, 'DebugName', '', @(x) ischar(x) || isstring(x));
            addParameter(p, 'DebugDir', fullfile('TBD', 'test', 'figs', 'debug'), @(x) ischar(x) || isstring(x));
            addParameter(p, 'DebugVisible', false, @islogical);
            
            parse(p, varargin{:});

            %%% TBD_PF is a value class, so these land on run()'s own copy of
            %%% obj and never reach the caller's filter object. timestep() sees
            %%% them (it is called on that copy), but show() has to read the
            %%% policy back off R, which is why run() records it there.
            obj.bootstrap = p.Results.Bootstrap;
            obj.RESAMPLE_THRESHOLD = p.Results.ESS;
            obj.likelihood = p.Results.Likelihood;
            obj.debugPlots = p.Results.DebugPlots;
            obj.NoiseSTD = p.Results.NoiseSTD;
            
            %%% Filepaths
            obj.TARGETCONTRIBUTION_FILEPATH = "TBD/supp/TargetContribution.mat";

            if strcmp(obj.likelihood, 'model')

                [obj.M, obj.G] = env.get_mWidar_matrices();
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
            prior   = [nan(4, obj.N); false(1, obj.N)];
            logw_prior = -log(obj.N)*ones(1,obj.N);

            R = struct();
            R.N         = obj.N;
            R.K         = K;
            R.particles = nan(5, obj.N, K);
            %%% Object state space:
            %%% x[1] = px
            %%% x[2] = vx
            %%% x[3] = py
            %%% x[4] = vy
            %%% x[5] = E (0 or 1)
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
            %%% Normalized weights BEFORE each frame's resample decision,
            %%% N x K. Plotting only (show 'AnimateWeights'): R.weights is
            %%% flat 1/N on every resampled frame, so it can't show the
            %%% weight collapse that triggered the resample. Column k is
            %%% paired with the pre-resample cloud, not R.particles(:,:,k).
            R.weightsPre = nan(obj.N, K);

            %%% Debug plots: everything visualize needs that only the filter
            %%% knows. Plotting only. yMin is the y gate hard-coded in
            %%% importance_weights / birth_cells / llr_map.
            if obj.debugPlots
                fp = struct('N', obj.N, 'Pb', obj.Pb, 'Ps', obj.Ps, 'gamma', obj.gamma, ...
                    'NoiseSTD', obj.NoiseSTD, 'TC', obj.TC, 'likelihood', obj.likelihood, ...
                    'yMin', 0.25, ...
                    'v_min', obj.v_min, 'v_max', obj.v_max, 'pEthresh', obj.pEthresh, ...
                    'essThresh', obj.RESAMPLE_THRESHOLD, 'bootstrap', obj.bootstrap);
                dbgCtx = obj.vis.debug_context(scenario, K, fp, ...
                    'Dir', p.Results.DebugDir, 'Name', p.Results.DebugName, ...
                    'Frames', p.Results.DebugFrames, 'Visible', p.Results.DebugVisible);
                dbgSteps = cell(1, K);
            end

            %%% Filter
            for k = 1:K
                [post, w, logw, ess, didResample, llrMap, births, dbg, wPre] = obj.timestep(prior, logw_prior, z{k});
                if obj.debugPlots
                    dbgSteps{k} = obj.vis.debug_step(dbg, k, dbgCtx);
                end
                if ~isempty(llrMap)
                    R.llrMax(k)   = max(llrMap(:));
                    R.llr(:,:,k)  = single(llrMap - R.llrMax(k));
                end
                R.births{k} = births;
                R.weightsPre(:,k) = wPre(:);

                R.particles(:,:,k) = post;
                R.weights(:,k)     = w(:);
                R.ess(k)           = ess;
                R.resampled(k)     = didResample;
                %%% Existence is the weight mass sitting on E = 1, not the
                %%% particle count: between resamples the two disagree.
                R.pE(k)    = sum(w(post(5,:) ~= 0)); %%% TODO: This is always == 1 with a target in the frame. Particles are dying... so it shouldn't be 1 
                R.exist(k) = R.pE(k) > obj.pEthresh;

                obj.debug_print(sprintf("k=%d/%d  pE=%.3f  exist=%d  ESS/N=%.3f  resampled=%d", ...
                    k, K, R.pE(k), R.exist(k), ess / obj.N, didResample));

                prior   = post; % Carry posterior forward as next step's prior
                logw_prior = logw;    % ... and its weights, that is the whole point
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

            if obj.debugPlots
                obj.vis.debug_summary([dbgSteps{:}], dbgCtx);
            end

        end

        %%% Quick look at a run. Draws the vis.plot_TBD dashboard (track over
        %%% the energy map, existence, per-axis position, error, occupied
        %%% cells, and ESS) and optionally plays the
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
        %%%   'AnimateWeights' also play particle weight vs particle index, one
        %%%               frame per timestep, in its own figure: normalized log
        %%%               weights on the left, normalized weights on the right
        %%%               (vis.animate_weights). Uses the weights from BEFORE
        %%%               each resample (R.weightsPre), since R.weights is flat
        %%%               1/N on resampled frames. Independent of 'Animate'.
        %%%               With 'Save' writes <base>_weights.gif. Default false.
        %%%
        %%%   figW is the weight animation figure, [] unless 'AnimateWeights'.
        function [fig, figAnim, figW] = show(obj, R, scenario, varargin)

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
            addParameter(p, 'AnimateWeights', false, @islogical);
            parse(p, varargin{:});
            opt = p.Results;

            figAnim = [];
            figW = [];

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

            if opt.AnimateWeights
                if isfield(R, 'weightsPre') && ~isempty(R.weightsPre)
                    W = R.weightsPre;
                    wLabel = 'pre-resample';
                elseif isfield(R, 'weights') && ~isempty(R.weights)
                    %%% Runs from before R.weightsPre existed
                    W = R.weights;
                    wLabel = 'posterior';
                    obj.debug_print("show: R has no weightsPre, animating the posterior R.weights (flat 1/N on resampled frames)");
                else
                    W = [];
                    obj.debug_print("show: R has no weights, skipping AnimateWeights");
                end
                if ~isempty(W)
                    wPath = '';
                    if ~isempty(opt.Save)
                        wPath = char(opt.Save) + "_weights.gif";
                    end
                    essArg = {};
                    if isfield(R, 'ess'), essArg = {'ESS', R.ess}; end
                    resArg = {};
                    if isfield(R, 'resampled'), resArg = {'Resampled', R.resampled}; end
                    obj.debug_print(sprintf("show: animating %s weights, %d frames at %g fps", ...
                        wLabel, size(W, 2), opt.FPS));
                    figW = obj.vis.animate_weights(W, ...
                        essArg{:}, resArg{:}, ...
                        'Which', wLabel, ...
                        'FPS',   opt.FPS, ...
                        'Title', ttl, ...
                        'Save',  wPath);
                end
            end

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
                    'DataUnits', 'meters', ...
                    'Trail',     opt.Trail, ...
                    'FPS',       opt.FPS, ...
                    'WeightColormap', opt.WeightColormap, ...
                    'WeightScale',    opt.WeightScale, ...
                    'Title',     ttl, ...
                    'CLim',          'frame', ...
                    'ImageLabel',    'raw signal (per-frame limits)', ...
                    second{:}, births{:}, ...
                    'Save',          animPath);
            end

        end
    end

    methods (Hidden)

        function [logW] = importance_weights(obj, Y, z)

            X = Y(1:4);
            E = Y(5);
            logW = 0; 
            if ~E, return; end % skip non existing


            px = X(1);
            py = X(3);
            
            %%% Particles outside the scene get a weight of 0, so log weight is -Inf
            if px > obj.max_x || px < obj.min_x
                logW = -Inf;
                return
            end

            % Going to set weights for particles below y = 0.25 to 0 to avoid sensor array
            if py > obj.max_y || py < 0.25
                logW = -Inf;
                return
            end

            % Particle's pixel index (meters -> index, grid is uniform so this is exact)
            ix = round((px - obj.xgrid(1)) / obj.dx) + 1;
            iy = round((py - obj.ygrid(1)) / obj.dy) + 1;

            if strcmp(obj.likelihood, 'model')
                [logW] = obj.model_likelihood(iy,ix);
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
                    %%% Products here become a sum
                    logW = logW + obj.gauss_likelihood([px py], [xm(i), ym(j)], ...
                                                    [xvec(i), yvec(j)], z);
                end
            end

        end

        %%% Likelihood ratio for one pixel (Ristic 11.20 style).
        %%%   pos  [px py] particle position in meters
        %%%   pix  [x y]   pixel center in meters
        %%%   idx  [col row] pixel index into z
        function l = gauss_likelihood(obj, pos, pix, idx, z)
            px = pos(1);
            py = pos(2);

            ii = pix(1);
            jj = pix(2);

            sig = obj.Sigma * obj.dx; % Sigma is given in pixels
            hh = exp( - ((ii -px)^2 + ((jj -py)^2))/ (2 * sig^2));
            zz = z(idx(2), idx(1)); % rows = y, cols = x
            l = (-(hh*(hh - 2*zz))/(2 * obj.NoiseSTD^2));
        end

        function [l] = model_likelihood(obj, r, c) % doesn't need z since C is computed already

            Cv = obj.C(r,c);
            TCv = obj.TC(r,c);
            l = (Cv - 0.5*TCv) / obj.NoiseSTD^2;   % log domain

        end
        %%% Profile log-LR for EVERY cell at once, from this frame's C: the
        %%% same expression model_likelihood evaluates per particle, so the
        %%% map is exactly what the particles are scored against. Rows below
        %%% y = 0.25 m are NaN, matching the region importance_weights zeroes.
        %%% Only used for plotting (R.llr); the filter itself still calls
        %%% model_likelihood per particle.
        function LL = llr_map(obj)
            LL = ( obj.C - 0.5 .* obj.TC) / obj.NoiseSTD^2;
            LL(obj.ygrid(:) < 0.25, :) = NaN;   % rows = y
        end

        %%% Run through one timestep of TBD_PF.
        %%%
        %%%   [post, w, logw, ESS, didResample, llrMap, births, dbg, wPre] = obj.timestep(prior, logw_prior, z)
        function [post, w, logw, ESS, didResample, llrMap, births, dbg, wPre] = timestep(obj, prior, logw_prior, z)
            llrMap = [];
            dbg = [];
            x_minus = prior(1:4,:);
            E_minus = prior(5,:);

            if nargin < 4
                error('TBD_PF:timestep', ...
                    'timestep now takes the prior weights: timestep(prior, w_prior, z)');
            end
            logw_prior = reshape(logw_prior, 1, []);

            % Regime transition
            E_plus = obj.RT(E_minus);
            x_plus = nan(4,obj.N);
            Y = nan(5,obj.N);

            % compute c_k
            % only if likelihood is model
            if strcmp(obj.likelihood, 'model')
                zt = z.'; % row major ordering
                c = obj.M' * (obj.G * zt(:));
                obj.C = reshape(c, obj.npx, obj.npx).';
                llrMap = obj.llr_map();
            end
            % birth_cells will return the # of valid cells, as well as the indicies 
            % valid cells are any which are in the upper gamma percentile of brightness
            % So for gamma = 0.7, births are uniformly chosen from the gamma brightest
            [birthIdx, nValid] = obj.birth_cells(z);

            % Hold our log weights
            logW = zeros(1, obj.N); 
            for n = 1:obj.N

                % New born particles
                if E_plus(n) && ~E_minus(n)
                    x_plus(:,n) = obj.sample_new(birthIdx); 
                elseif E_plus(n) && E_minus(n)
                    x_plus(:,n) = obj.dynamics(x_minus(:,n));
                end

                % Evaluate importance weights
                Y(:, n) = [x_plus(:,n); E_plus(n)];
                [logW(n)] = obj.importance_weights(Y(:,n),z);

            end

            %%% Update
            %%% logw_prior + logW is the weight update
            [w, logw_post] = obj.normalize_log(logw_prior + logW);
            ESS = obj.get_ESS(w);
            wPre = w;   % plotting only, the resample below overwrites w

            %%% Where this frame's newborns landed, BEFORE the resample: a
            %%% newborn almost never survives one, so the posterior alone
            %%% can't show the birth process. Plotting only.
            births = Y([1 3], E_plus & ~E_minus);   % 2 x nBorn [px; py], meters

            %%% Debug plots only: snapshot the step before the resample
            %%% overwrites w. logw repeats the expression passed to
            %%% normalize_log above.
            if obj.debugPlots
                dbg = struct('z', z, 'C', obj.C, 'llrMap', llrMap, ...
                    'birthIdx', birthIdx, 'nValid', nValid, 'logBirthCorr', logBirthCorr, ...
                    'E_minus', E_minus, 'E_plus', E_plus, ...
                    'x_minus', x_minus, 'x_plus', x_plus, 'Y', Y, ...
                    'logw_prior', logw_prior, 'logL', logL, ...
                    'logw', logw_prior + logL, 'w', w, 'ESS', ESS);
            end

            didResample = obj.bootstrap || (ESS <= obj.RESAMPLE_THRESHOLD * obj.N);
            if didResample
                [post, w] = obj.resample(Y, w);
                logw = -log(obj.N)*ones(1,obj.N);
            else
                post = Y;
                logw = logw_post;
            end

            if obj.debugPlots
                dbg.didResample = didResample;
                dbg.post        = post;
                dbg.w_post      = w;
            end

            if obj.debug
                nBorn = nnz(E_plus & ~E_minus);
                kept  = nnz(E_plus &  E_minus);
                died  = nnz(~E_plus & E_minus);
                w0    = nnz(E_plus & logW == -Inf); % alive but zero weight (OOB or likelihood underflow)
                birthMode = sprintf("top %.0f%% of valid cells (%d)", 100*(1 - obj.gamma), numel(birthIdx));
                obj.debug_print(sprintf("timestep: born=%d kept=%d died=%d alive=%d w0=%d | L max=%.3g ESS/N=%.3f resampled=%d | birth=%s", ...
                    nBorn, kept, died, nnz(E_plus), w0, max(logW), ESS / obj.N, didResample, birthMode));
            end

        end

        %%% Normalize log weights with the log-sum-exp trick. Returns the
        %%% linear weights w_n (sum to 1) and their logs logw_norm.
        function [w_n,logw_norm] = normalize_log(obj, logw)
            logw = reshape(double(logw), 1, []);
            logw(isnan(logw)) = -Inf;
            m = max(logw);
            if ~isfinite(m)
                obj.debug_print("all log weights -Inf, using uniform")
                w_n = ones(size(logw)) / numel(logw);
                logw_norm = -log(obj.N);
                return
            end
            %%% Subtract the max before exp so the largest term is exp(0) = 1:
            %%% no overflow, and the ratios (all that survives normalizing)
            %%% are unchanged.
            w = exp(logw - m);
            w_n = w ./ sum(w);
            logw_norm = log(w_n);
            assert(abs(sum(w_n) - 1) < 1e-6, "Sum of weights != 1, sum(w_n) = %d", sum(w_n))
            % assert(abs(exp(sum(logw_norm)) - 1) < 1e-6, "Sum of log weights != 1, exp(sum(logw_norm)) = %d", sum(logw_norm))
            
        end
        %%% Effective sample size, Kong's 1/sum(w^2) written scale-free so it
        %%% works on normalized or unnormalized weights. Ranges over [1, N].
        function ESS = get_ESS(~, w)
            s2 = sum(w.^2);
            if ~(s2 > 0)
                ESS = NaN; % every weight zero, caller treats this as "no info"
                return
            end
            ESS = 1 / s2; % Min =1, Max = N
        end

        % the brightest (1 - gamma) fraction of the valid pixels in this frame
        % rows below y = 0.25 are excluded always

        function [idx, nValid] = birth_cells(obj,z)
            validRow = obj.ygrid(:) >= 0.25;
            nValid = nnz(validRow)*obj.npx;
            nKeep = max(1, ceil((1-obj.gamma) * nValid));
            if all(z == 0, 'all') % need a check for empty frames, sample uniformly
                validMask = repmat(validRow, 1, size(z, 2));   % 128x128 logical
                validLin  = find(validMask);                   % linear indices of valid pixels
                idx       = validLin(randperm(numel(validLin), nKeep));
            else
                zv = z;
                zv(~validRow,:) = -Inf; % never chosen
                
                [~, idx] = maxk(zv(:), nKeep);
            end
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
            
            %%% Sample velocity uniformly
            vx = obj.v_min + (obj.v_max - obj.v_min) * rand();
            vy = obj.v_min + (obj.v_max - obj.v_min) * rand();
            X = [px; vx; py; vy]; % Random new alive particle
        end

        %%% Push a particle through it's dynamics model
        function X_plus = dynamics(obj, X_minus)
            X_plus = obj.F * X_minus + mvnrnd(zeros(4,1), obj.Q)';
        end


        %%% Systematic resampling. w must already be normalized. The
        %%% returned weights are flat 1/N, which is what lets the SIS recursion
        %%% in timestep() reduce to the bootstrap weights on the next frame.
        function [post, w] = resample(obj, pre, w)
            post = nan(5,obj.N);

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
                if isfield(scenario, 'tvec') && numel(scenario.tvec) == R.K
                    res.t = scenario.tvec;
                end
            elseif iscell(scenario) && ~isempty(scenario)
                res.signals = cat(3, scenario{:});
            end
        end

    end
end