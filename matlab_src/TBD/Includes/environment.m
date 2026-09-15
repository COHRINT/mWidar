%%% environment class will create GT tracks using generator, and then create signals with simulator.
%%% The main function scenario = environment.setup() will return the signal at each timestep, as well as the truth data at each timestep
%%% In a scenario struct for simple access
%%% scenario struct example:
%%% scenario.object_count = n, # of objects present in scenario
%%% scenario.signal = {1 x K} cell array, where each cell contains the signal (128x128) at time k
%%% scenario.truth  = {1 x n} cell array, where each cell is the 4 x K ground truth [px; vx; py; vy] of object i
%%% This should allow things to flow down easily
%%%
%%% environment is a facade: it owns one generator, one simulator and one
%%% visualize, forwards the relevant config to each, and stitches their
%%% outputs together. It does no trajectory or forward-model math itself.
%%% mWidar is a value class, so environment is too: methods that change
%%% config must return obj, and setup() returns the scenario rather than
%%% caching it on the object.

classdef environment < mWidar

    properties
        %%% Flags
        debug
        units   % 'meters' (generator output), forwarded to sim & vis
        seed    % rng seed, [] leaves rng alone

        %%% Composed objects
        gen     % generator
        sim     % simulator
        vis     % visualize

        %%% Scenario / truth config (forwarded to generator)
        ct              % # of objects
        kEnd            % final timestep
        dt              % seconds
        tvec            % 0:dt:kEnd*dt, so K = kEnd + 1 samples
        K               % # of timesteps
        trajectories    % 1 x ct cellstr from {'line','parabola','scurve'}
        rw              % 1 x ct logical
        start           % 1 x ct cell of [x y] in meters, [] = random in-scene
        final           % 1 x ct cell of [x y] in meters, [] = random in-scene

        %%% Existence (per object, in timesteps 1..K). All alive by default.
        kBirth          % 1 x ct
        kDeath          % 1 x ct

        %%% Signal config (forwarded to simulator)
        normalize
        blur
        sigma
    end

    methods

        %%% Constructor
        function obj = environment(varargin)

            obj = obj@mWidar();

            p = inputParser;
            p.FunctionName = 'environment';

            % Flags
            addParameter(p, 'Debug', false, @islogical);
            addParameter(p, 'Units', 'meters', @(x) ismember(x, {'meters', 'Meters', 'm', 'pixels', 'pixel', 'px'}));
            addParameter(p, 'Seed', [], @(x) isempty(x) || (isscalar(x) && x >= 0 && mod(x,1) == 0));

            % Scenario / truth
            addParameter(p, 'objs', 1, @(x) isscalar(x) && x >= 1 && mod(x,1) == 0);
            addParameter(p, 'kEnd', 100, @(x) isscalar(x) && x >= 1 && mod(x,1) == 0);
            addParameter(p, 'dt', 0.1, @(x) isscalar(x) && x > 0);
            addParameter(p, 'trajectories', {'line'}, @(x) iscellstr(x) && all(ismember(x, {'line', 'parabola', 'scurve'})));
            addParameter(p, 'rw', false, @(x) isvector(x) && (islogical(x) || isnumeric(x)));
            addParameter(p, 'start', [], @(x) isempty(x) || isnumeric(x) || iscell(x));
            addParameter(p, 'final', [], @(x) isempty(x) || isnumeric(x) || iscell(x));

            % Existence
            addParameter(p, 'kBirth', [], @(x) isempty(x) || (isvector(x) && all(mod(x,1) == 0)));
            addParameter(p, 'kDeath', [], @(x) isempty(x) || (isvector(x) && all(mod(x,1) == 0)));

            % Signal
            addParameter(p, 'Normalize', true, @islogical);
            addParameter(p, 'Blur', true, @islogical);
            addParameter(p, 'Sigma', 2, @(x) isscalar(x) && x > 0);

            parse(p, varargin{:});
            R = p.Results;

            %%% Flags
            obj.debug = R.Debug;
            obj.units = R.Units;
            obj.seed  = R.Seed;

            %%% Time base
            obj.ct   = R.objs;
            obj.kEnd = R.kEnd;
            obj.dt   = R.dt;
            obj.tvec = 0:obj.dt:(obj.kEnd * obj.dt);
            obj.K    = numel(obj.tvec);

            %%% Per-object config. Scalars are broadcast to ct, anything else
            %%% must already be length ct. Fail here rather than letting
            %%% generator print ABORTING and hand back an empty cell.
            obj.trajectories = obj.broadcast(R.trajectories, 'trajectories');
            obj.rw           = logical(obj.broadcast(R.rw, 'rw'));
            obj.start        = obj.normalize_endpoints(R.start, 'start');
            obj.final        = obj.normalize_endpoints(R.final, 'final');

            %%% Existence, defaults to alive for the whole scenario
            if isempty(R.kBirth), R.kBirth = 1;     end
            if isempty(R.kDeath), R.kDeath = obj.K; end
            obj.kBirth = obj.broadcast(R.kBirth, 'kBirth');
            obj.kDeath = obj.broadcast(R.kDeath, 'kDeath');
            if any(obj.kBirth < 1) || any(obj.kDeath > obj.K) || any(obj.kBirth > obj.kDeath)
                error('environment:existence', ...
                      'kBirth/kDeath must satisfy 1 <= kBirth <= kDeath <= K (K = %d)', obj.K);
            end

            %%% Signal
            obj.normalize = R.Normalize;
            obj.blur      = R.Blur;
            obj.sigma     = R.Sigma;

            %%% Build the sub-objects from a single property -> option table
            gen_args = obj.forward_args('gen');
            sim_args = obj.forward_args('sim');
            vis_args = obj.forward_args('vis');

            obj.gen = generator(gen_args{:});
            obj.sim = simulator(sim_args{:});
            obj.vis = visualize(vis_args{:});

            obj.debug_print(sprintf("constructed: %d obj, K=%d, dt=%.3g, traj={%s}, rw=[%s]", ...
                obj.ct, obj.K, obj.dt, strjoin(obj.trajectories, ','), num2str(double(obj.rw))));
        end

        %%% Returns a 1 x ct cell of 4 x K
        %%% state histories [px; vx; py; vy] in meters.
        function tracks = generate(obj)

            if ~isempty(obj.seed)
                rng(obj.seed);
            end

            % Fill in any endpoints the user left as []
            [p0, p1] = obj.default_endpoints();

            % generator wants plain vectors for one trajectory, cells otherwise
            if obj.ct == 1
                p0 = p0{1};
                p1 = p1{1};
            end

            tracks = obj.gen.generate_trajectories(p0, p1);

            if isempty(tracks)
                error('environment:generate', 'generator aborted, see messages above');
            end

        end

        function signals = simulate(obj, tracks)

            signals = cell(1,obj.K);

            for k = 1:obj.K
                
                % gather pos for every object alive at k
                pos = cell(1, obj.ct);
                for i = 1:obj.ct
                    if k >=  obj.kBirth(i) && k <= obj.kDeath(i)
                        pos{i} = [tracks{i}(1,k) tracks{i}(3,k)];
                    end
                end

                % Call simulator
                if strcmp(obj.units, 'meters')
                    signals{k} = obj.sim.generate_mWidar_image(pos,'meters',true);
                elseif strcmp(obj.units, 'pixels')
                    signals{k} = obj.sim.generate_mWidar_image(pos,'pixels',true);
                end
            end
        end

        function scenario = setup(obj)
            tracks = obj.generate();
            signals = obj.simulate(tracks);
            scenario = obj.build_scenario(tracks, signals);
        end

        %%% Quick look at a scenario. Draws the truth tracks over the max
        %%% intensity projection of the signal, and optionally plays the
        %%% frame-by-frame time history. Everything is delegated to vis.
        %%%
        %%%   fig = env.show(scenario)
        %%%   [fig, figAnim] = env.show(scenario, 'Animate', true, 'Save', 'figs/run1')
        %%%
        %%% Options
        %%%   'Animate'   also play the time history (default false, slow for big K)
        %%%   'Save'      base path with no extension. Writes <base>_tracks.png
        %%%               and, with Animate, <base>_history.gif
        %%%   'FPS'       animation frame rate (default vis.fps)
        %%%   'Trail'     # of past samples drawn in the animation (default 15)
        %%%   'Title'     override the auto title on the track plot
        function [fig, figAnim] = show(obj, scenario, varargin)

            p = inputParser;
            addParameter(p, 'Animate', false, @islogical);
            addParameter(p, 'Save', '', @(x) ischar(x) || isstring(x));
            addParameter(p, 'FPS', obj.vis.fps);
            addParameter(p, 'Trail', 15);
            addParameter(p, 'Title', '');
            parse(p, varargin{:});
            R = p.Results;

            figAnim = [];

            % vis wants signals as an npx x npx x K stack, scenario keeps a cell
            stack = cat(3, scenario.signal{:});

            % Per-target legend entries from the config snapshot
            labels = cell(1, scenario.object_count);
            for i = 1:scenario.object_count
                rw_tag = '';
                if scenario.meta.rw(i), rw_tag = ' + rw'; end
                labels{i} = sprintf('%d: %s%s', i, scenario.meta.trajectories{i}, rw_tag);
            end

            ttl = R.Title;
            if isempty(ttl)
                ttl = sprintf('%d object(s), K = %d, dt = %.3g s', ...
                              scenario.object_count, scenario.K, scenario.dt);
            end

            savePath = '';
            if ~isempty(R.Save)
                savePath = char(R.Save) + "_tracks.png";
            end

            % Tracks over the energy MIP, masked so a target only draws while alive
            fig = obj.vis.trajectories(scenario.truth, ...
                'Background', stack, ...
                'Mask',       scenario.exist, ...
                'Labels',     labels, ...
                'DataUnits',  scenario.units, ...
                'Title',      ttl, ...
                'Save',       savePath);

            if R.Animate
                animPath = '';
                if ~isempty(R.Save)
                    animPath = char(R.Save) + "_history.gif";
                end
                figAnim = obj.vis.animate_time_history(stack, ...
                    'Truth',     scenario.truth, ...
                    'DataUnits', scenario.units, ...
                    'Trail',     R.Trail, ...
                    'FPS',       R.FPS, ...
                    'Title',     ttl, ...
                    'Save',      animPath);
            end

        end

        function save_scenario(~, scenario, path)
            save(path,"scenario")
        end

        function scenario = load_scenario(~, path)
            scenario = load(path).scenario;
        end

    end

    methods(Hidden)

        %%% Name/value cell for one sub-object's constructor. This is the only
        %%% place that knows which environment property maps to which option
        %%% name on which class, so the constructor and configure() agree.
        function args = forward_args(obj, which)
            switch which
                case 'gen'
                    args = {'Debug', obj.debug, ...
                            'Units', obj.units, ...
                            'kEnd', obj.kEnd, ...
                            'dt', obj.dt, ...
                            'objs', obj.ct, ...
                            'trajectories', obj.trajectories, ...
                            'rw', obj.rw};
                case 'sim'
                    args = {'Debug', obj.debug, ...
                            'Objects', obj.ct, ...
                            'Normalize', obj.normalize, ...
                            'Blur', obj.blur, ...
                            'Sigma', obj.sigma, ...
                            'Objects', obj.ct};
                case 'vis'
                    args = {'Debug', obj.debug, ...
                            'Units', obj.units};
                otherwise
                    error('environment:forward_args', 'unknown target "%s"', which);
            end
        end

        %%% Assemble the scenario struct handed downstream. Everything a
        %%% filter or plot needs is here, so a saved scenario is self
        %%% describing without the environment that produced it.
        %%%
        %%%   scenario.object_count   ct
        %%%   scenario.K              # timesteps (kEnd + 1)
        %%%   scenario.dt, .tvec      time base, tvec is 1 x K
        %%%   scenario.units          units of truth ('meters' or 'pixels')
        %%%   scenario.signal         1 x K cell, each npx x npx
        %%%   scenario.truth          1 x ct cell, each 4 x K  [px; vx; py; vy]
        %%%   scenario.exist          ct x K logical, true where object i is alive
        %%%   scenario.cardinality    1 x K, # objects alive at each k
        %%%   scenario.meta           config snapshot (trajectories, rw, endpoints, sim settings, seed)
        function scenario = build_scenario(obj, tracks, signals)

            % Shape checks, so a bad upstream stage fails here with a clear
            % message instead of deep inside a filter.
            if ~iscell(tracks) || numel(tracks) ~= obj.ct
                error('environment:scenario', 'tracks must be a 1 x %d cell', obj.ct);
            end
            if ~iscell(signals) || numel(signals) ~= obj.K
                error('environment:scenario', 'signals must be a 1 x %d cell', obj.K);
            end
            for i = 1:obj.ct
                if ~isequal(size(tracks{i}), [4 obj.K])
                    error('environment:scenario', 'tracks{%d} is %s, expected 4 x %d', ...
                          i, mat2str(size(tracks{i})), obj.K);
                end
            end
            for k = 1:obj.K
                if ~isequal(size(signals{k}), [obj.npx obj.npx])
                    error('environment:scenario', 'signals{%d} is %s, expected %d x %d', ...
                          k, mat2str(size(signals{k})), obj.npx, obj.npx);
                end
            end

            % Existence mask from the per-object birth/death timesteps
            exist_mask = false(obj.ct, obj.K);
            for i = 1:obj.ct
                exist_mask(i, obj.kBirth(i):obj.kDeath(i)) = true;
            end

            scenario = struct();
            scenario.object_count = obj.ct;
            scenario.K            = obj.K;
            scenario.dt           = obj.dt;
            scenario.tvec         = obj.tvec;
            scenario.units        = obj.units;
            scenario.signal       = reshape(signals, 1, []);
            scenario.truth        = reshape(tracks, 1, []);
            scenario.exist        = exist_mask;
            scenario.cardinality  = sum(exist_mask, 1);

            % Actual endpoints used, read back off the tracks so randomly
            % chosen ones are recorded too.
            p0 = cell(1, obj.ct);
            p1 = cell(1, obj.ct);
            for i = 1:obj.ct
                p0{i} = [tracks{i}(1,1)   tracks{i}(3,1)];
                p1{i} = [tracks{i}(1,end) tracks{i}(3,end)];
            end

            scenario.meta = struct( ...
                'trajectories', {obj.trajectories}, ...
                'rw',           obj.rw, ...
                'start',        {p0}, ...
                'final',        {p1}, ...
                'kBirth',       obj.kBirth, ...
                'kDeath',       obj.kDeath, ...
                'normalize',    obj.normalize, ...
                'blur',         obj.blur, ...
                'sigma',        obj.sigma, ...
                'seed',         obj.seed, ...
                'npx',          obj.npx, ...
                'created',      char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')));

            obj.debug_print(sprintf("scenario built: %d obj, K=%d, %d signal frames", ...
                obj.ct, obj.K, numel(scenario.signal)));

        end

        %%% Expand a scalar (or 1-element cell) to 1 x ct, or verify an
        %%% existing vector is already 1 x ct.
        function v = broadcast(obj, v, name)
            if numel(v) == 1
                v = repmat(v(:)', 1, obj.ct);
            elseif numel(v) ~= obj.ct
                error('environment:length', ...
                      '"%s" has %d entries but objs = %d', name, numel(v), obj.ct);
            end
            v = reshape(v, 1, []);
        end

        %%% Accept [] (all random), a 1x2 vector (one object or broadcast), or a
        %%% 1 x ct cell whose entries are 1x2 or []. Always return a 1 x ct cell.
        function c = normalize_endpoints(obj, x, name)
            if isempty(x)
                c = cell(1, obj.ct);
                return
            end
            if isnumeric(x)
                if numel(x) ~= 2
                    error('environment:endpoint', '"%s" must be a [x y] pair', name);
                end
                c = repmat({reshape(x, 1, 2)}, 1, obj.ct);
                return
            end
            % cell
            if numel(x) ~= obj.ct
                error('environment:length', ...
                      '"%s" has %d entries but objs = %d', name, numel(x), obj.ct);
            end
            c = cell(1, obj.ct);
            for i = 1:obj.ct
                xi = x{i};
                if isempty(xi)
                    c{i} = [];
                elseif isnumeric(xi) && numel(xi) == 2
                    c{i} = reshape(xi, 1, 2);
                else
                    error('environment:endpoint', '"%s"{%d} must be a [x y] pair or []', name, i);
                end
            end
        end

        function [] = debug_print(obj, str)
            if obj.debug
                str = "[DEBUG][ENVIRONMENT]" + str + "\n";
                fprintf(str)
            end
        end

        %%% Fill any empty start/final entries with a random in-scene pair.
        %%% Uses the same bands as generator.validate (start bottom-left,
        %%% final top-right) so defaults look like the fallback. Entries the
        %%% user supplied are left alone. Returns 1 x ct cells.
        function [p0, p1] = default_endpoints(obj)
            p0 = obj.start;
            p1 = obj.final;
            for i = 1:obj.ct
                if isempty(p0{i})
                    obj.debug_print(sprintf("object %d: no start given, using random in-scene point", i));
                    p0{i} = [-1.9 + 0.9*rand(),  0.5 + 1.0*rand()];   % x in [-1.9,-1], y in [0.5,1.5]
                end
                if isempty(p1{i})
                    obj.debug_print(sprintf("object %d: no final given, using random in-scene point", i));
                    p1{i} = [ 1.0 + 0.9*rand(),  3.0 + 0.9*rand()];   % x in [1,1.9],  y in [3,3.9]
                end
            end
        end

    end

end
