%%% Parent class for all TBD sub classes

classdef TBD < mWidar

    properties

        %%% Flags
        debug
        units

        %%% Composed objects
        vis % visualize, used by show()

        %%% Sensor Params
        
        NoiseSTD
        Sigma % For Gaussian PSF, blurring parameter in PIXELS (converted to meters in the likelihood)
        dt % dt
        Ip % Do more reasearch on this term

        %%% PF Params
        N % # of particles
        Pb % prob of birth
        Ps % prob of survival
        pEthresh % existence threshold
        PI % Markov transision
        pDist % cells you evaluate accross for liklihood

        %%% Target Dynamics
        A
        F
        Q
        v_max % used for sampling velocity on newly birthed particles
        v_min

        %%% General
        gamma
        seed % rng
        K % # of timesteps
    end

    methods
        function obj = TBD(varargin)
            obj = obj@mWidar();
            
            p = inputParser;
            
            % flags
            addParameter(p, 'Debug', false, @islogical);
            addParameter(p,'Units', 'meters', @(x) ismember(x, {'meters', 'Meters', 'm', 'pixels', 'pixel', 'px'}));
            addParameter(p, 'Seed', 420, @(x) isempty(x) || (isscalar(x) && x >= 0 && mod(x,1) == 0));

            % sensor params
            addParameter(p, 'STD', 0.5, @(x) isscalar(x) && x > 0);
            addParameter(p, 'Sigma', 0.5, @(x) isscalar(x) && x > 0);
            addParameter(p, 'dt', 0.1, @(x) isscalar(x) && x > 0);
            addParameter(p, 'Ip', 0.75, @(x) isscalar(x) && x > 0);

            % PF Params
            addParameter(p, 'N', 5000, @(x) isscalar(x) && x >= 1 && mod(x,1) == 0);
            addParameter(p, 'Pb', 0.03, @(x) x >= 0 && x <= 1);
            addParameter(p, 'Ps', 0.98, @(x) x >= 0 && x <= 1);
            addParameter(p, 'pEthresh', 0.5, @(x) x >= 0 && x <= 1);
            addParameter(p, 'pDist', 5, @(x) isscalar(x) && x >= 0 && mod(x,1) == 0);
            
            % Target dynamics
            addParameter(p, 'Q', [0.01 0 0 0;0 0.5 0 0;0 0 0.01 0;0 0 0 0.5], @(x) isequal(size(x), [4 4]));
            addParameter(p, 'v_max', 10, @(x) isscalar(x));
            addParameter(p, 'v_min', -10, @(x) isscalar(x));

            % general
            addParameter(p, 'gamma', 0.75, @(x) isscalar(x) && x >= 0);
            addParameter(p, 'K', 100, @(x) isscalar(x) && x >= 1 && mod(x,1) == 0);

            parse(p, varargin{:});
            R = p.Results;

            %%% Flags
            obj.debug = R.Debug;
            obj.units = R.Units;
            obj.seed  = R.Seed;

            %%% Plotting, same idea as environment.vis
            obj.vis = visualize('Debug', obj.debug, 'Units', obj.units);

            %%% Sensor params
            obj.NoiseSTD = R.STD;
            obj.Sigma    = R.Sigma;
            obj.dt       = R.dt;
            obj.Ip       = R.Ip;

            %%% PF params
            obj.N        = R.N;
            obj.Pb       = R.Pb;
            obj.Ps       = R.Ps;
            obj.pEthresh = R.pEthresh;
            obj.pDist    = R.pDist;
            % Existence Markov chain, rows = from {dead, alive}, cols = to {dead, alive}
            obj.PI = [1-obj.Pb, obj.Pb;
                      1-obj.Ps, obj.Ps];

            %%% Target dynamics, constant velocity in [px; vx; py; vy]
            obj.A = [0 1 0 0;
                     0 0 0 0;
                     0 0 0 1;
                     0 0 0 0];
            obj.F = expm(obj.A * obj.dt);
            obj.Q = R.Q;
            if R.v_min > R.v_max
                error('TBD:badVelocityBounds', 'v_min (%g) must be <= v_max (%g)', R.v_min, R.v_max);
            end
            obj.v_max = R.v_max;
            obj.v_min = R.v_min;

            %%% General
            obj.gamma = R.gamma;
            obj.K     = R.K;

            if ~isempty(obj.seed)
                rng(obj.seed);
            end

            obj.debug_print(sprintf("constructed: N=%d, K=%d, dt=%.3g, Pb=%.3g, Ps=%.3g, units=%s", ...
                obj.N, obj.K, obj.dt, obj.Pb, obj.Ps, obj.units));
        end

    end
    
    methods (Abstract)
        %%% All these should be in child classes
        w = importance_weights(obj, Y, z)
        l = guass_likelihood(obj, pos, pix, z)
        X = sample_new(obj, z)
        post = timestep(obj, prior, z)
        X_plus = dynamics(obj, X_minus)
        w_n = normalize(obj, w)

    end
    methods(Hidden)
        function [] = debug_print(obj, str)
            if obj.debug
                str = "[DEBUG][" + upper(class(obj)) + "]" + str + "\n";
                fprintf(str)
            end
        end

        function rp = RT(obj,rm)
            rp = false(1, obj.N);
            C = [zeros(2,1), cumsum(obj.PI,2)];
            u = rand(obj.N,1);
            for n = 1:obj.N
                i = rm(n) + 1;
                rp(n) = find(u(n) <= C(i,2:end), 1) - 1;
            end
        end
    end

end
