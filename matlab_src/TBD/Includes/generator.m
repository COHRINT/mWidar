
%%%%%%%% Generator %%%%%%%%%%
%%% Responsible for generating valid GT tracks
%%% Different options for target paths: constant accel line, parabola, analytic s-curve
%%% Option to enable rw to all target paths, will make traj non-analytic/stochastic


classdef generator < mWidar

    properties

        %%% Flags
        debug
        %%% Default to meters, can set to pixels
        units

        %%% Object count, defaults 1
        ct
        %%% Trajectory, list of trajectories you want to generate. 
        % TODO: If length of list is more than ct, it will pick up to obj ct. If its less than ct, it will repeat trajs
        trajectories

        %%% rw, enable or disable rw, default false
        rw

        %%% Final time step, defaults to 100
        kEnd
  
        %%% delta time in seconds, defaults to 0.1
        dt
        tvec

        %%% System dynamics for linear systems: 
        %%% Will use constant acceleration model to make harder tracks
        %%% x = [px vx ax py vy ay]
        A
        F

    end

    methods

        %%% Constructor
        function obj =  generator(varargin)
    
            p = inputParser;
            addParameter(p, 'Debug', false, @islogical);
            addParameter(p,'Units', 'meters', @(x) ismember(x, {'meters', 'Meters', 'm', 'pixels', 'pixel', 'px'}));
            addParameter(p,'kEnd', 100, @(x) isscalar(x) && mod(x,1) == 0 ); % ensure its whole #
            addParameter(p,'dt', 0.1);
            addParameter(p,'objs',1, @(x) isscalar(x) && mod(x,1) == 0 );
            addParameter(p,'trajectories',{'line'}, @(x) iscellstr(x) && all(ismember(x, {'line', 'parabola', 'scurve'})));
            addParameter(p,'rw', false, @(x) isvector(x) && (islogical(x) || isnumeric(x))); %%% Should be a vector of logicals


            parse(p, varargin{:});

            obj.debug = p.Results.Debug;
            obj.units = p.Results.Units;
            obj.kEnd = p.Results.kEnd;
            obj.dt = p.Results.dt;
            obj.ct = p.Results.objs;
            obj.trajectories = p.Results.trajectories;
            obj.rw = p.Results.rw;
            obj.tvec = 0:obj.dt:(obj.kEnd * obj.dt);
            %%% C.A model
            
            obj.A = [0 1 0 0 0 0;
                     0 0 1 0 0 0;
                     0 0 0 0 0 0;
                     0 0 0 1 0 0;
                     0 0 0 0 1 0;
                     0 0 0 0 0 0];

            obj.F = expm(obj.A * obj.dt);

        end

        %%% Start and final should be cell arrays if n_traj != 1, ow normal vector is fine
        function [X] = generate_trajectories(obj,start, final);
            
            n_traj = numel(obj.trajectories);

            %%% Check, ensure length of start, final, and obj.rw == n_traj
            %%% TODO: make this better
            X = {};

            if numel(obj.rw) ~= n_traj
                fprintf("ABORTING GENERATOR: length of rw not consistent w/ # of trajectories provided \n")
                return;
            end

            if iscell(start) ~= iscell(final)
                fprintf("ABORTING GENERATOR: start and final must both be cell arrays or both be vectors \n")
                return;
            end

            if iscell(start)
                if numel(start) ~= n_traj || numel(final) ~= n_traj
                    fprintf("ABORTING GENERATOR: length of start or final not consistent w/ # of trajectories provided \n")
                    return;
                end
            elseif n_traj ~= 1
                fprintf("ABORTING GENERATOR: start and final must be cell arrays when # of trajectories > 1 \n")
                return;
            end
            
  
            X = cell(1,n_traj);


            for i = 1:n_traj
                traj = obj.trajectories{i};
                rw_enabled = obj.rw(i);

                traj_start = [0,0]; % placeholder
                if iscell(start)
                    traj_start = start{i};
                else 
                    traj_start = start;
                end

                traj_end = [0,0]; % placeholder
                if iscell(final)
                    traj_end = final{i};
                else 
                    traj_end = final;
                end
                switch traj
                    case 'line'
                        if rw_enabled
                            X{i} = obj.generate_line_rw(traj_start, traj_end);
                        else
                            X{i} = obj.generate_line(traj_start, traj_end);
                        end
                    case 'parabola'
                        if rw_enabled
                            X{i} = obj.generate_parabola_rw(traj_start, traj_end);
                        else
                            X{i} = obj.generate_parabola(traj_start, traj_end);
                        end
                    case 'scurve'
                        if rw_enabled
                            X{i} = obj.generate_scurve_rw(traj_start, traj_end);
                        else
                            X{i} = obj.generate_scurve(traj_start, traj_end);
                        end
                end
            
            end



        end


    end

    methods(Hidden)

        %%% All path generating functions take in start and end pos, and construct path between those.

        %%% Generate_Line
        function [X] = generate_line(obj, start, final)

            %%% Check bounds, if out of bounds generate random start/end pos
            [x_start, x_end, y_start, y_end] = obj.validate(start, final);

            T = obj.tvec(end);

            delta_pos = [x_end - x_start; y_end - y_start];

            % Use a scalar progress variable with constant acceleration so the
            % trajectory stays on the straight segment between start and end.
            a_progress = -0.01 + 0.02 .* rand();
            v0_progress = (1 - 0.5 * a_progress * T^2) / T;

            progress      = v0_progress .* obj.tvec + 0.5 .* a_progress .* obj.tvec.^2;
            progress_dot  = v0_progress + a_progress .* obj.tvec;

            x_traj = x_start + delta_pos(1) .* progress;
            y_traj = y_start + delta_pos(2) .* progress;

            vx_traj = delta_pos(1) .* progress_dot;
            vy_traj = delta_pos(2) .* progress_dot;

            x_traj = max(-2, min(2, x_traj));
            y_traj = max( 0, min(4, y_traj));

            X = [x_traj; vx_traj; y_traj; vy_traj];


        end

        %%% Generate_Parabola
        function [X] = generate_parabola(obj, start, final)
            %%% Check bounds, if out of bounds generate random start/end pos
            [x_start, x_end, y_start, y_end] = obj.validate(start, final);

            T = obj.tvec(end);

            delta_pos = [x_end - x_start; y_end - y_start];
            chord = norm(delta_pos);

            % Progress advances at a constant rate along the chord, and the
            % parabolic bulge is applied along the chord normal. That keeps the
            % acceleration constant, so the path matches the C.A. model exactly.
            s = obj.tvec ./ T;

            if chord < eps
                % Degenerate segment, bulge straight up so the path is still a curve
                n = [0; 1];
                chord = 1;
            else
                n = [-delta_pos(2); delta_pos(1)] ./ chord;
            end

            % Apex offset, random magnitude and random side of the chord
            arc_dir = sign(rand() - 0.5);
            if arc_dir == 0
                arc_dir = 1;
            end
            h = arc_dir * (0.2 + 0.3 .* rand()) * chord;

            % Shrink the apex until the whole arc sits inside the scene. h -> 0
            % is the straight chord, which is in bounds, so this terminates.
            for i = 1:25
                bulge = 4 .* h .* s .* (1 - s);
                x_traj = x_start + delta_pos(1) .* s + n(1) .* bulge;
                y_traj = y_start + delta_pos(2) .* s + n(2) .* bulge;

                if all(x_traj > -2 & x_traj < 2) && all(y_traj > 0 & y_traj < 4)
                    break
                end
                h = 0.7 * h;
            end

            % d/dt of the above, ds/dt = 1/T and d(bulge)/ds = 4h(1 - 2s)
            bulge_dot = 4 .* h .* (1 - 2 .* s) ./ T;

            vx_traj = delta_pos(1) ./ T + n(1) .* bulge_dot;
            vy_traj = delta_pos(2) ./ T + n(2) .* bulge_dot;

            X = [x_traj; vx_traj; y_traj; vy_traj];

        end

        %%% Generate_SCurve
        function [X] = generate_scurve(obj, start, final)

            %%% Check bounds, if out of bounds generate random start/end pos
            [x_start, x_end, y_start, y_end] = obj.validate(start, final);

            T   = obj.tvec(end);
            tau = obj.tvec / T;  % [0, 1]

            % Straight chord, always inside the scene for in-bounds endpoints
            x_line = x_start + (x_end - x_start)*tau;
            y_line = y_start + (y_end - y_start)*tau;

            % The S wiggle laid on top of the chord, and its d/dtau
            wx = 0.6*sin(2*pi*tau).*(1-tau).^2.*tau.^2;
            wy = 0.4*sin(pi*tau).*sin(2*pi*tau);

            dwx_dtau = 0.6.*(2*pi*cos(2*pi*tau).*(1-tau).^2.*tau.^2 + ...
                        sin(2*pi*tau).*(2*(1-tau).*tau.^2.*(-1) + 2*(1-tau).^2.*tau));
            dwy_dtau = 0.4.*(pi*cos(pi*tau).*sin(2*pi*tau) + ...
                        sin(pi*tau).*2*pi.*cos(2*pi*tau));

            % Shrink the wiggle to fit the scene rather than clamping the
            % result, which would leave the analytic velocity below describing
            % a path the target no longer follows.
            sw = min(obj.fit_scale(x_line, wx, obj.min_x, obj.max_x), ...
                     obj.fit_scale(y_line, wy, obj.min_y, obj.max_y));

            % Position
            x_traj = x_line + sw.*wx;
            y_traj = y_line + sw.*wy;

            % Velocity  (d/dt analytically)
            dtau_dt = 1/T;
            vx_traj = ((x_end-x_start) + sw.*dwx_dtau) .* dtau_dt;
            vy_traj = ((y_end-y_start) + sw.*dwy_dtau) .* dtau_dt;

            X = [x_traj;  vx_traj; y_traj; vy_traj];  % 4 x n_t

        end

        %%% Generate_Line_rw
        function [X] = generate_line_rw(obj,start,final)

            %%% Check bounds, if out of bounds generate random start/end pos
            [x_start, x_end, y_start, y_end] = obj.validate(start, final);

            T = obj.tvec(end);

            % Nominal path is the one generate_line builds: a scalar progress
            % variable with constant acceleration along the chord.
            delta_pos = [x_end - x_start; y_end - y_start];

            a_progress  = -0.01 + 0.02 .* rand();
            v0_progress = (1 - 0.5 * a_progress * T^2) / T;
            progress    = v0_progress .* obj.tvec + 0.5 .* a_progress .* obj.tvec.^2;

            x_base = x_start + delta_pos(1) .* progress;
            y_base = y_start + delta_pos(2) .* progress;

            X = obj.apply_random_walk(x_base, y_base);

        end

        %%% Generate_Parabola_rw
        function [X] = generate_parabola_rw(obj,start,final)

            %%% Check bounds, if out of bounds generate random start/end pos
            [x_start, x_end, y_start, y_end] = obj.validate(start, final);

            tau = obj.tvec ./ obj.tvec(end);

            % Nominal path is the one generate_parabola builds: constant-rate
            % progress along the chord plus a bulge along its normal.
            delta_pos = [x_end - x_start; y_end - y_start];
            chord = norm(delta_pos);

            if chord < eps
                % Degenerate segment, bulge straight up so the path is still a curve
                n = [0; 1];
                chord = 1;
            else
                n = [-delta_pos(2); delta_pos(1)] ./ chord;
            end

            % Apex offset, random magnitude and random side of the chord
            arc_dir = sign(rand() - 0.5);
            if arc_dir == 0
                arc_dir = 1;
            end
            h = arc_dir * (0.2 + 0.3 .* rand()) * chord;

            % Shrink the apex until the whole arc sits inside the scene. h -> 0
            % is the straight chord, which is in bounds, so this terminates.
            for i = 1:25
                bulge = 4 .* h .* tau .* (1 - tau);
                x_base = x_start + delta_pos(1) .* tau + n(1) .* bulge;
                y_base = y_start + delta_pos(2) .* tau + n(2) .* bulge;

                if all(x_base > -2 & x_base < 2) && all(y_base > 0 & y_base < 4)
                    break
                end
                h = 0.7 * h;
            end

            X = obj.apply_random_walk(x_base, y_base);

        end

        %%% Generate_SCurve_rw
        function [X] = generate_scurve_rw(obj,start,final)

            %%% Check bounds, if out of bounds generate random start/end pos
            [x_start, x_end, y_start, y_end] = obj.validate(start, final);

            tau = obj.tvec ./ obj.tvec(end);

            % Nominal path is the one generate_scurve builds, wiggle fitted to
            % the scene so the base handed to the random walk is already inside.
            x_line = x_start + (x_end - x_start).*tau;
            y_line = y_start + (y_end - y_start).*tau;

            wx = 0.6.*sin(2.*pi.*tau).*(1-tau).^2.*tau.^2;
            wy = 0.4.*sin(pi.*tau).*sin(2.*pi.*tau);

            sw = min(obj.fit_scale(x_line, wx, obj.min_x, obj.max_x), ...
                     obj.fit_scale(y_line, wy, obj.min_y, obj.max_y));

            x_base = x_line + sw.*wx;
            y_base = y_line + sw.*wy;

            X = obj.apply_random_walk(x_base, y_base);

        end

        %%% Perturb a nominal path with a random-walk acceleration bias.
        %%%
        %%% The perturbation is scaled to fit inside the scene rather than
        %%% clamped at the edge. Clamping used to pin a target against a wall
        %%% for long stretches, which flattened the reported velocity to zero
        %%% and then stepped it discontinuously on release. No motion model can
        %%% follow that, so the deviation is shrunk to fit instead.
        function [X] = apply_random_walk(obj, x_base, y_base)

            n_t = numel(obj.tvec);
            tau = obj.tvec ./ obj.tvec(end);

            % Every ~5 timesteps, let the acceleration bias take a random walk.
            change_interval = 5;
            accel_step_sigma = 0.04;
            accel_bias_limit = 0.1;

            accel_bias = zeros(2, n_t);
            current_bias = [0; 0];
            for i = 2:n_t
                if mod(i-1, change_interval) == 0
                    current_bias = current_bias + accel_step_sigma .* randn(2,1);
                    current_bias = max(-accel_bias_limit, min(accel_bias_limit, current_bias));
                end
                accel_bias(:, i) = current_bias;
            end

            % Integrate the bias to get a smooth deviation from the nominal path.
            vel_offset = zeros(2, n_t);
            pos_offset = zeros(2, n_t);
            for i = 2:n_t
                pos_offset(:, i) = pos_offset(:, i-1) + vel_offset(:, i-1).*obj.dt + 0.5.*accel_bias(:, i-1).*obj.dt.^2;
                vel_offset(:, i) = vel_offset(:, i-1) + accel_bias(:, i-1).*obj.dt;
            end

            % Keep the perturbation small near the start/end so the result still
            % resembles the nominal path and holds the requested endpoints.
            envelope = sin(pi.*tau).^2;
            dx = envelope .* pos_offset(1, :);
            dy = envelope .* pos_offset(2, :);

            % One uniform scale for both axes, so the shape of the deviation is
            % preserved and only its size changes.
            s = min(obj.fit_scale(x_base, dx, obj.min_x, obj.max_x), ...
                    obj.fit_scale(y_base, dy, obj.min_y, obj.max_y));

            x_traj = x_base + s .* dx;
            y_traj = y_base + s .* dy;

            % Derivatives of the path actually returned, so the reported state
            % stays self consistent. No clamp here, s already guarantees bounds.
            vx_traj = gradient(x_traj, obj.dt);
            vy_traj = gradient(y_traj, obj.dt);

            X = [x_traj;  vx_traj; y_traj; vy_traj];  % 4 x n_t

        end

        %%% Largest s in [0, 1] with lo <= base + s*delta <= hi at every sample.
        %%% The band is inset by a small margin so a path approaches the scene
        %%% edge rather than touching it, but it is never tightened past where
        %%% the nominal path already sits, so a feasible s always exists.
        function s = fit_scale(~, base, delta, lo, hi)

            margin = 0.02 * (hi - lo);
            lo = min(lo + margin, min(base));
            hi = max(hi - margin, max(base));

            s = 1;

            rising = delta > 0;
            if any(rising)
                s = min(s, min((hi - base(rising)) ./ delta(rising)));
            end

            falling = delta < 0;
            if any(falling)
                s = min(s, min((lo - base(falling)) ./ delta(falling)));
            end

            s = max(0, s);

        end

        function [x_start, x_end, y_start, y_end] = validate(obj, start, final)
            if obj.checkbound(start) && obj.checkbound(final)
                x_start = start(1); y_start = start(2);
                x_end = final(1); y_end = final(2);
            else
                obj.debug_print("Start/End position provided out of bounds, falling back to random start/end pos")
                x_start = -1.9 + (-1-(-1.9)).*rand(); x_end = 1 + (1.9-(1)).*rand();
                y_start = 0.5 + (1.5-(0.5)).*rand(); y_end = 3 + (3.9-(3)).*rand();
            end
        end        

        function [] = debug_print(obj, str)
            if obj.debug
                str = "[DEBUG][GENERATOR]" + str + "\n";
                fprintf(str)
            end
        end
    end


end