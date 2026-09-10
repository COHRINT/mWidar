%%% visualize.m
%%% Holds methods for visualizing TBD/mWidar stuff
%%% Totally not written by claude, all me
%%%
%%% Conventions used throughout this class
%%%   - Signal images are npx x npx arrays indexed signal(row,col) = signal(y,x),
%%%     matching simulator.m which writes S(Gy,Gx). Every image is drawn with
%%%     imagesc(xvec, yvec, signal) + axis xy so row -> y and col -> x.
%%%   - State vectors follow the demo layout [px; vx; py; vy]; the rows holding
%%%     position are configurable via the 'StateIdx' property (default [1 3]).
%%%     A 2-row state is interpreted directly as [px; py].
%%%   - Ground truth / estimated tracks are 4 x K, 4 x K x T (T targets) or a
%%%     1 x T cell of 4 x K. Everything is normalized internally to a cell.
%%%   - Particle histories are 5 x N x K ([px;vx;py;vy;E]), 5 x N x K x T, or a
%%%     cell of 5 x N x K.
%%%   - 'Units' controls the axes (default 'meters'); 'DataUnits' declares what
%%%     the passed data is already in and defaults to 'Units' (no conversion).
%%%     Pass 'DataUnits','pixels' when handing over raw filter state.
%%%
%%% All methods are stateless: scene geometry + styling live on the object,
%%% data is always passed in. Every method returns the figure handle and
%%% accepts 'Save', <path> to write the figure out.

classdef visualize < mWidar

    properties
        %%% Flags
        debug

        %%% Default axis units for every plot: "meters" or "pixels"
        units

        %%% Rows of the state vector holding [x; y]
        stateIdx

        %%% Styling
        cmap % colormap for signal images
        divmap % diverging colormap for difference images
        truthColor
        estColor
        particleColor
        lw % line width
        ms % marker size
        fs % font size

        %%% Export
        dpi % resolution for exportgraphics
        fps % default animation frame rate
    end

    methods

        %%% Visualize constructor
        function obj = visualize(varargin)

            obj = obj@mWidar();

            p = inputParser;
            addParameter(p, 'Debug', false, @islogical);
            addParameter(p, 'Units', 'meters');
            addParameter(p, 'StateIdx', [1 3]);
            addParameter(p, 'Colormap', 'gray');
            addParameter(p, 'FontSize', 11);
            addParameter(p, 'LineWidth', 1.5);
            addParameter(p, 'MarkerSize', 6);
            addParameter(p, 'DPI', 200);
            addParameter(p, 'FPS', 15);
            parse(p, varargin{:})

            obj.debug = p.Results.Debug;
            obj.units = obj.check_units(p.Results.Units);
            obj.stateIdx = p.Results.StateIdx;
            obj.cmap = p.Results.Colormap;
            obj.divmap = 'parula';
            obj.fs = p.Results.FontSize;
            obj.lw = p.Results.LineWidth;
            obj.ms = p.Results.MarkerSize;
            obj.dpi = p.Results.DPI;
            obj.fps = p.Results.FPS;

            obj.truthColor = [0 0 0];
            obj.estColor = [0.85 0.10 0.10];
            obj.particleColor = [0.20 0.55 0.95];

        end

        %% ------------------------------------------------------------------
        %% Scene / signal plots
        %% ------------------------------------------------------------------

        %{
            Plot object trajectory

            fig = v.trajectories(tracks, ...)

            tracks    4 x K, 4 x K x T, or cell of 4 x K (may be [] if only
                      truth is given via 'Truth')
            Options
              'Truth'       ground truth tracks, same formats as tracks
              'Background'  npx x npx image, or an npx x npx x K stack (a max
                            intensity projection over k is drawn) to lay the
                            tracks over the measured energy
              'ColorByTime' scatter the estimate colored by frame index
              'Mask'        1 x K (or T x K) logical, only plot where true
                            (e.g. declared = pE > thresh)
              'Labels'      cellstr of per-target names for the legend
              'Units'/'DataUnits'/'Axes'/'Title'/'Save'
        %}
        function fig = trajectories(obj, tracks, varargin)

            p = obj.common_parser();
            addParameter(p, 'Truth', []);
            addParameter(p, 'Background', []);
            addParameter(p, 'ColorByTime', false, @islogical);
            addParameter(p, 'Mask', []);
            addParameter(p, 'Labels', {});
            parse(p, varargin{:})
            R = p.Results;

            [fig, ax] = obj.get_axes(R, 'Trajectories');
            du = obj.resolve_data_units(R);

            %%% Optional energy background
            if ~isempty(R.Background)
                bg = R.Background;
                if ndims(bg) == 3
                    bg = max(bg, [], 3); % max intensity projection over time
                end
                obj.draw_image(ax, bg, R.Units, R.Colormap, []);
                hold(ax, 'on');
            end

            est = obj.to_cell(tracks);
            tru = obj.to_cell(R.Truth);
            mask = obj.to_mask(R.Mask, max(numel(est), numel(tru)));
            cols = obj.palette(max(numel(est), 1));

            h = gobjects(0);
            lbl = {};

            %%% Ground truth first so estimates draw on top
            for t = 1:numel(tru)
                [x, y] = obj.track_xy(tru{t}, du, R.Units);
                hh = plot(ax, x, y, '-', 'Color', obj.truthColor, ...
                    'LineWidth', obj.lw);
                hold(ax, 'on');
                plot(ax, x(find(~isnan(x), 1, 'first')), ...
                    y(find(~isnan(y), 1, 'first')), 's', ...
                    'Color', obj.truthColor, 'MarkerFaceColor', obj.truthColor, ...
                    'MarkerSize', obj.ms, 'HandleVisibility', 'off');
                if t == 1
                    h(end+1) = hh; %#ok<AGROW>
                    lbl{end+1} = 'Truth'; %#ok<AGROW>
                else
                    set(hh, 'HandleVisibility', 'off');
                end
            end

            %%% Estimates
            for t = 1:numel(est)
                [x, y] = obj.track_xy(est{t}, du, R.Units);
                m = obj.mask_for(mask, t, numel(x));
                x(~m) = NaN;
                y(~m) = NaN;

                if R.ColorByTime
                    kk = 1:numel(x);
                    good = ~isnan(x) & ~isnan(y);
                    hh = scatter(ax, x(good), y(good), 22, kk(good), 'filled');
                    hold(ax, 'on');
                    cb = colorbar(ax);
                    cb.Label.String = 'k';
                else
                    hh = plot(ax, x, y, '.-', 'Color', cols(t,:), ...
                        'LineWidth', obj.lw, 'MarkerSize', obj.ms + 4);
                    hold(ax, 'on');
                end
                h(end+1) = hh; %#ok<AGROW>
                if t <= numel(R.Labels)
                    lbl{end+1} = R.Labels{t}; %#ok<AGROW>
                elseif numel(est) == 1
                    lbl{end+1} = 'Estimate'; %#ok<AGROW>
                else
                    lbl{end+1} = sprintf('Estimate %d', t); %#ok<AGROW>
                end
            end

            obj.style_scene(ax, R.Units);
            if ~isempty(h)
                legend(ax, h, lbl, 'Location', 'best');
            end
            obj.set_title(ax, R.Title, 'Target Trajectories');
            obj.save_figure(fig, R.Save);

        end

        %{
            Plot just a single mWidar image frame

            fig = v.signal_frame(signal, ...)

            Options
              'Truth'/'Est'   4 x 1 states (or 4 x T) to mark on the frame
              'Particles'     5 x N particle set for frame k, overlaid as dots
              'Weights'       1 x N weights, used to size/shade the particles
              'CLim'          color limits, [] for auto
              'Colorbar'      logical, default true
              'Units'/'DataUnits'/'Axes'/'Title'/'Save'
        %}
        function fig = signal_frame(obj, signal, varargin)

            p = obj.common_parser();
            addParameter(p, 'Truth', []);
            addParameter(p, 'Est', []);
            addParameter(p, 'Particles', []);
            addParameter(p, 'Weights', []);
            addParameter(p, 'CLim', []);
            addParameter(p, 'Colorbar', true, @islogical);
            parse(p, varargin{:})
            R = p.Results;

            if isempty(signal)
                obj.debug_print("EMPTY SIGNAL PASSED TO signal_frame");
            end

            [fig, ax] = obj.get_axes(R, 'mWidar Frame');
            du = obj.resolve_data_units(R);

            obj.draw_image(ax, signal, R.Units, R.Colormap, R.CLim);
            hold(ax, 'on');

            if R.Colorbar
                cb = colorbar(ax);
                cb.Label.String = 'intensity';
            end

            obj.overlay_particles(ax, R.Particles, R.Weights, du, R.Units);
            obj.overlay_states(ax, R.Truth, du, R.Units, obj.truthColor, 'o', 'Truth');
            obj.overlay_states(ax, R.Est, du, R.Units, obj.estColor, 'x', 'Estimate');

            obj.style_scene(ax, R.Units);
            obj.maybe_legend(ax);
            obj.set_title(ax, R.Title, 'mWidar Signal Frame');
            obj.save_figure(fig, R.Save);

        end

        %{
            Grid of frames pulled out of a signal stack. Quick way to eyeball a
            whole run without waiting on an animation.

            fig = v.signal_montage(signals, ...)

            Options
              'Frames'  explicit frame indices; overrides 'Count'
              'Count'   number of evenly spaced frames (default 9)
              'Truth'   truth tracks, marked on the frame they belong to
              'Est'     estimated tracks, same
              'CLim'    shared color limits ('auto' per-frame if [])
        %}
        function fig = signal_montage(obj, signals, varargin)

            p = obj.common_parser();
            addParameter(p, 'Frames', []);
            addParameter(p, 'Count', 9);
            addParameter(p, 'Truth', []);
            addParameter(p, 'Est', []);
            addParameter(p, 'CLim', []);
            parse(p, varargin{:})
            R = p.Results;

            K = size(signals, 3);
            if isempty(R.Frames)
                n = min(R.Count, K);
                idx = unique(round(linspace(1, K, n)));
            else
                idx = R.Frames(:)';
            end

            du = obj.resolve_data_units(R);
            tru = obj.to_cell(R.Truth);
            est = obj.to_cell(R.Est);

            fig = obj.new_figure('Signal Montage', [1000 800]);
            nc = ceil(sqrt(numel(idx)));
            nr = ceil(numel(idx) / nc);
            tl = tiledlayout(fig, nr, nc, 'TileSpacing', 'compact', 'Padding', 'compact');

            for ii = 1:numel(idx)
                k = idx(ii);
                ax = nexttile(tl);
                obj.draw_image(ax, signals(:,:,k), R.Units, R.Colormap, R.CLim);
                hold(ax, 'on');
                obj.mark_tracks_at_k(ax, tru, k, du, R.Units, obj.truthColor, 'o');
                obj.mark_tracks_at_k(ax, est, k, du, R.Units, obj.estColor, 'x');
                obj.style_scene(ax, R.Units);
                title(ax, sprintf('k = %d', k), 'FontSize', obj.fs);
                if ii ~= 1
                    xlabel(ax, ''); ylabel(ax, '');
                end
            end

            if ~isempty(R.Title)
                title(tl, R.Title, 'FontSize', obj.fs + 2);
            end
            obj.save_figure(fig, R.Save);

        end

        %{
            Side by side comparison of two frames plus their difference.
            Built for validating the simulator against measured data, or one
            forward-model setting against another.

            fig = v.compare_signals(A, B, ...)

            Options
              'Names'  1 x 2 cellstr of panel titles
        %}
        function fig = compare_signals(obj, A, B, varargin)

            p = obj.common_parser();
            addParameter(p, 'Names', {'A', 'B'});
            parse(p, varargin{:})
            R = p.Results;

            assert(isequal(size(A), size(B)), 'compare_signals: size mismatch');

            D = A - B;
            lim = max(abs(D(:)));
            if lim == 0 || ~isfinite(lim)
                lim = 1;
            end
            shared = [min([A(:); B(:)]), max([A(:); B(:)])];
            if diff(shared) == 0
                shared = shared + [-1 1];
            end

            rmseVal = sqrt(mean(D(:).^2));
            cc = corrcoef(A(:), B(:));
            if numel(cc) > 1
                ccVal = cc(1,2);
            else
                ccVal = NaN;
            end

            fig = obj.new_figure('Signal Comparison', [1300 460]);
            tl = tiledlayout(fig, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

            ax = nexttile(tl);
            obj.draw_image(ax, A, R.Units, R.Colormap, shared);
            obj.style_scene(ax, R.Units);
            title(ax, R.Names{1}, 'FontSize', obj.fs);

            ax = nexttile(tl);
            obj.draw_image(ax, B, R.Units, R.Colormap, shared);
            obj.style_scene(ax, R.Units);
            title(ax, R.Names{2}, 'FontSize', obj.fs);
            colorbar(ax);

            ax = nexttile(tl);
            obj.draw_image(ax, D, R.Units, obj.divmap, [-lim lim]);
            obj.style_scene(ax, R.Units);
            title(ax, sprintf('%s - %s   (RMSE %.3g, \\rho %.3f)', ...
                R.Names{1}, R.Names{2}, rmseVal, ccVal), 'FontSize', obj.fs);
            colorbar(ax);

            if ~isempty(R.Title)
                title(tl, R.Title, 'FontSize', obj.fs + 2);
            end
            obj.save_figure(fig, R.Save);

        end

        %{
            Animate entire time history with signal in the back and object
            moving through it.

            fig = v.animate_time_history(signals, ...)

            Options
              'Truth'      truth tracks (4 x K x T or cell)
              'Est'        estimated tracks, same formats
              'Particles'  5 x N x K particle history (or cell / 5 x N x K x T)
              'Weights'    N x K weights
              'pE'         T x K existence probability, shown in the title
              'Trail'      # of past estimate samples to keep drawn (default 15,
                           Inf for the full track)
              'FPS'        playback / export frame rate
              'Save'       output path; extension picks the format
              'Format'     'gif' | 'mp4' | 'png' (frame dump), inferred from Save
              'CLim'       fixed color limits so brightness does not flicker
              'Pause'      extra pause per frame during live playback
        %}
        function fig = animate_time_history(obj, signals, varargin)

            p = obj.common_parser();
            addParameter(p, 'Truth', []);
            addParameter(p, 'Est', []);
            addParameter(p, 'Particles', []);
            addParameter(p, 'Weights', []);
            addParameter(p, 'pE', []);
            addParameter(p, 'Trail', 15);
            addParameter(p, 'FPS', obj.fps);
            addParameter(p, 'Format', '');
            addParameter(p, 'CLim', []);
            addParameter(p, 'Pause', 0);
            parse(p, varargin{:})
            R = p.Results;

            K = size(signals, 3);
            du = obj.resolve_data_units(R);
            tru = obj.to_cell(R.Truth);
            est = obj.to_cell(R.Est);
            Yc = obj.to_particle_cell(R.Particles);
            pE = obj.to_rowmat(R.pE);

            %%% Lock the color scale across the run unless told otherwise, so
            %%% frame-to-frame brightness changes mean something.
            clim_ = R.CLim;
            if isempty(clim_)
                clim_ = [min(signals(:)), max(signals(:))];
                if diff(clim_) == 0
                    clim_ = clim_ + [0 1];
                end
            end

            [fig, ax] = obj.get_axes(R, 'Time History');
            [fmt, outFile] = obj.resolve_format(R.Save, R.Format);
            vw = [];
            if fmt == "mp4"
                vw = VideoWriter(outFile, 'MPEG-4');
                vw.FrameRate = R.FPS;
                open(vw);
            end

            for k = 1:K

                cla(ax);
                obj.draw_image(ax, signals(:,:,k), R.Units, R.Colormap, clim_);
                hold(ax, 'on');

                %%% Particle cloud for this frame
                for t = 1:numel(Yc)
                    Yk = Yc{t}(:,:,k);
                    wk = [];
                    if ~isempty(R.Weights)
                        wk = R.Weights(:,k);
                    end
                    obj.overlay_particles(ax, Yk, wk, du, R.Units);
                end

                %%% Trails + current markers
                obj.draw_trails(ax, tru, k, R.Trail, du, R.Units, obj.truthColor, '-');
                obj.draw_trails(ax, est, k, R.Trail, du, R.Units, obj.estColor, '--');
                obj.mark_tracks_at_k(ax, tru, k, du, R.Units, obj.truthColor, 'o');
                obj.mark_tracks_at_k(ax, est, k, du, R.Units, obj.estColor, 'x');

                obj.style_scene(ax, R.Units);

                ttl = sprintf('k = %d / %d', k, K);
                if ~isempty(pE)
                    ttl = [ttl, sprintf('   P(E) = %s', ...
                        strjoin(compose('%.2f', pE(:,k)'), ', '))]; %#ok<AGROW>
                end
                if ~isempty(R.Title)
                    ttl = [R.Title, '   |   ', ttl];
                end
                title(ax, ttl, 'FontSize', obj.fs);

                drawnow limitrate;

                switch fmt
                    case "gif"
                        frame = getframe(fig);
                        [im8, map] = rgb2ind(frame2im(frame), 256);
                        if k == 1
                            imwrite(im8, map, outFile, 'gif', ...
                                'LoopCount', Inf, 'DelayTime', 1/R.FPS);
                        else
                            imwrite(im8, map, outFile, 'gif', ...
                                'WriteMode', 'append', 'DelayTime', 1/R.FPS);
                        end
                    case "mp4"
                        writeVideo(vw, getframe(fig));
                    case "png"
                        [d, n, e] = fileparts(outFile);
                        exportgraphics(fig, fullfile(d, sprintf('%s_%03d%s', n, k, e)), ...
                            'Resolution', obj.dpi);
                    otherwise
                        if R.Pause > 0
                            pause(R.Pause);
                        end
                end

            end

            if ~isempty(vw)
                close(vw);
            end

        end

        %% ------------------------------------------------------------------
        %% TBD specific plots
        %% ------------------------------------------------------------------

        %{
            The one-stop TBD results dashboard: existence, track in the plane,
            per-axis position vs time, position error, and particle health.

            fig = v.plot_TBD(res, ...)

            res is a struct; see visualize.results_template() for the fields.
            Anything absent is simply skipped, so partial results still plot.

            Options
              'pEthresh'  declaration threshold (default 0.5 or res.pEthresh)
              'Units'/'DataUnits'/'Title'/'Save'
        %}
        function fig = plot_TBD(obj, res, varargin)

            p = obj.common_parser();
            addParameter(p, 'pEthresh', []);
            parse(p, varargin{:})
            R = p.Results;

            res = obj.fill_results(res);
            du = obj.resolve_data_units(R);

            thresh = R.pEthresh;
            if isempty(thresh)
                thresh = res.pEthresh;
            end

            %%% Derive estimates from the particle history when the caller did
            %%% not hand over a point estimate.
            [est, pE] = obj.resolve_estimates(res);
            tru = obj.to_cell(res.truth);
            kvec = obj.time_vector(res, obj.n_steps(res, est, tru));
            [tlab, ~] = obj.time_label(res);

            %%%   [ track | existence ]
            %%%   [  p_x  |    p_y    ]
            %%%   [ error | particles ]
            %%% Wide on purpose: the scene panel is axis-equal, so it leaves
            %%% horizontal room in its tile that the legend drops into.
            fig = obj.new_figure('TBD Results', [1400 950]);
            tl = tiledlayout(fig, 3, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

            %%% Track in the plane over the energy map -------------------------
            ax = nexttile(tl);
            bg = [];
            if ~isempty(res.signals)
                bg = max(res.signals, [], 3);
            end
            if ~isempty(bg)
                obj.draw_image(ax, bg, R.Units, R.Colormap, []);
                hold(ax, 'on');
            end
            cols = obj.palette(max(numel(est), 1));
            %%% All truth is drawn black, so only the first one earns a legend
            %%% entry - numbering identical black lines tells you nothing.
            for t = 1:numel(tru)
                [x, y] = obj.track_xy(tru{t}, du, R.Units);
                hh = plot(ax, x, y, '-', 'Color', obj.truthColor, 'LineWidth', obj.lw, ...
                    'DisplayName', 'Truth');
                if t > 1
                    set(hh, 'HandleVisibility', 'off');
                end
                hold(ax, 'on');
            end
            for t = 1:numel(est)
                [x, y] = obj.track_xy(est{t}, du, R.Units);
                if ~isempty(pE) && t <= size(pE,1)
                    declared = pE(t,:) > thresh;
                    n = min(numel(x), numel(declared));
                    x(1:n) = obj.nan_where(x(1:n), ~declared(1:n));
                    y(1:n) = obj.nan_where(y(1:n), ~declared(1:n));
                end
                plot(ax, x, y, '.-', 'Color', cols(t,:), 'LineWidth', obj.lw, ...
                    'MarkerSize', obj.ms + 4, ...
                    'DisplayName', obj.name_for('Estimate', t, numel(est)));
                hold(ax, 'on');
            end
            obj.style_scene(ax, R.Units);
            %%% Legend goes outside: the scene is square and a 'best' legend
            %%% lands on top of the image every time.
            obj.maybe_legend(ax, 'eastoutside');
            title(ax, 'Track (declared only)', 'FontSize', obj.fs);

            %%% Existence -----------------------------------------------------
            ax = nexttile(tl);
            obj.existence_axes(ax, kvec, pE, obj.to_rowmat(res.Etruth), thresh, tlab);

            %%% Per-axis position ---------------------------------------------
            axNames = {'p_x', 'p_y'};
            for d = 1:2
                ax = nexttile(tl);
                for t = 1:numel(tru)
                    v = obj.axis_series(tru{t}, d, du, R.Units);
                    plot(ax, kvec(1:min(end,numel(v))), v(1:min(end,numel(kvec))), '-', ...
                        'Color', obj.truthColor, 'LineWidth', obj.lw);
                    hold(ax, 'on');
                end
                for t = 1:numel(est)
                    v = obj.axis_series(est{t}, d, du, R.Units);
                    plot(ax, kvec(1:min(end,numel(v))), v(1:min(end,numel(kvec))), '.', ...
                        'Color', cols(t,:), 'MarkerSize', obj.ms + 4);
                    hold(ax, 'on');
                end
                grid(ax, 'on');
                xlabel(ax, tlab, 'FontSize', obj.fs);
                ylabel(ax, [axNames{d} ' [' char(obj.unit_suffix(R.Units)) ']'], ...
                    'FontSize', obj.fs);
                title(ax, [axNames{d} ' vs time'], 'FontSize', obj.fs);
            end

            %%% Position error --------------------------------------------------
            ax = nexttile(tl);
            if ~isempty(tru) && ~isempty(est)
                err = obj.compute_error(tru, est, du, R.Units);
                for t = 1:size(err,1)
                    plot(ax, kvec(1:min(end,size(err,2))), err(t,1:min(end,numel(kvec))), ...
                        '-', 'Color', cols(min(t,size(cols,1)),:), 'LineWidth', obj.lw);
                    hold(ax, 'on');
                end
                good = isfinite(err);
                if any(good(:))
                    yline(ax, sqrt(mean(err(good).^2)), 'k:', 'LineWidth', 1, ...
                        'Label', 'RMSE');
                end
            else
                text(ax, 0.5, 0.5, 'no truth/estimate pair', ...
                    'Units', 'normalized', 'HorizontalAlignment', 'center');
            end
            grid(ax, 'on');
            xlabel(ax, tlab, 'FontSize', obj.fs);
            ylabel(ax, ['position error [' char(obj.unit_suffix(R.Units)) ']'], ...
                'FontSize', obj.fs);
            title(ax, 'Position Error', 'FontSize', obj.fs);

            %%% Particle health ---------------------------------------------------
            ax = nexttile(tl);
            if ~isempty(res.weights)
                ess = obj.ess(res.weights);
                plot(ax, kvec(1:min(end,numel(ess))), ...
                    ess(1:min(end,numel(kvec))) ./ size(res.weights,1), ...
                    '-', 'Color', obj.particleColor, 'LineWidth', obj.lw);
                ylabel(ax, 'ESS / N', 'FontSize', obj.fs);
                ylim(ax, [0 1]);
                title(ax, 'Particle Health', 'FontSize', obj.fs);
            elseif ~isempty(res.particles)
                Yc = obj.to_particle_cell(res.particles);
                nAlive = squeeze(sum(Yc{1}(5,:,:) ~= 0, 2))';
                plot(ax, kvec(1:min(end,numel(nAlive))), ...
                    nAlive(1:min(end,numel(kvec))), '-', ...
                    'Color', obj.particleColor, 'LineWidth', obj.lw);
                ylabel(ax, '# particles with E = 1', 'FontSize', obj.fs);
                title(ax, 'Existing Particles', 'FontSize', obj.fs);
            else
                text(ax, 0.5, 0.5, 'no particle data', ...
                    'Units', 'normalized', 'HorizontalAlignment', 'center');
            end
            grid(ax, 'on');
            xlabel(ax, tlab, 'FontSize', obj.fs);

            if ~isempty(R.Title)
                title(tl, R.Title, 'FontSize', obj.fs + 2);
            end
            obj.save_figure(fig, R.Save);

        end

        %{
            Existence probability vs time against truth. Split out of plot_TBD
            so it can be used on its own when tuning Pb / Ps / the threshold.

            fig = v.existence(pE, ...)
              pE  T x K estimated existence probability
            Options
              'Etruth'    T x K (or 1 x K) true existence flags
              'pEthresh'  declaration threshold (default 0.5)
              'Time'      1 x K time vector, defaults to 1:K
        %}
        function fig = existence(obj, pE, varargin)

            p = obj.common_parser();
            addParameter(p, 'Etruth', []);
            addParameter(p, 'pEthresh', 0.5);
            addParameter(p, 'Time', []);
            parse(p, varargin{:})
            R = p.Results;

            pE = obj.to_rowmat(pE);
            kvec = R.Time;
            if isempty(kvec)
                kvec = 1:size(pE,2);
            end

            [fig, ax] = obj.get_axes(R, 'Target Existence');
            obj.existence_axes(ax, kvec, pE, obj.to_rowmat(R.Etruth), R.pEthresh, 'k');
            obj.set_title(ax, R.Title, 'Target Existence');
            obj.save_figure(fig, R.Save);

        end

        %{
            Estimated vs true target count over time. The headline number for
            multi-target TBD: are births and deaths being caught at all?

            fig = v.cardinality(estCard, ...)
              estCard  1 x K estimated number of targets, or a T x K matrix of
                       existence probabilities (summed to give the expected count)
            Options
              'Truth'  1 x K true count, or truth tracks (counted as the number
                       of non-NaN tracks at each k)
              'Time'
        %}
        function fig = cardinality(obj, estCard, varargin)

            p = obj.common_parser();
            addParameter(p, 'Truth', []);
            addParameter(p, 'Time', []);
            parse(p, varargin{:})
            R = p.Results;

            estCard = obj.to_rowmat(estCard);
            if size(estCard,1) > 1
                estCard = sum(estCard, 1); % expected cardinality from per-target pE
            end

            trueCard = [];
            if ~isempty(R.Truth)
                if isnumeric(R.Truth) && isvector(R.Truth)
                    trueCard = R.Truth(:)';
                else
                    tru = obj.to_cell(R.Truth);
                    trueCard = zeros(1, size(tru{1}, 2));
                    for t = 1:numel(tru)
                        trueCard = trueCard + ~isnan(tru{t}(obj.state_row(tru{t},1), :));
                    end
                end
            end

            kvec = R.Time;
            if isempty(kvec)
                kvec = 1:numel(estCard);
            end

            [fig, ax] = obj.get_axes(R, 'Cardinality');
            if ~isempty(trueCard)
                stairs(ax, kvec(1:min(end,numel(trueCard))), ...
                    trueCard(1:min(end,numel(kvec))), '-', ...
                    'Color', obj.truthColor, 'LineWidth', obj.lw, ...
                    'DisplayName', 'True count');
                hold(ax, 'on');
            end
            plot(ax, kvec(1:min(end,numel(estCard))), estCard(1:min(end,numel(kvec))), ...
                '-', 'Color', obj.estColor, 'LineWidth', obj.lw, ...
                'DisplayName', 'Estimated count');
            grid(ax, 'on');
            xlabel(ax, 'k', 'FontSize', obj.fs);
            ylabel(ax, '# targets', 'FontSize', obj.fs);
            obj.maybe_legend(ax);
            obj.set_title(ax, R.Title, 'Target Cardinality');
            obj.save_figure(fig, R.Save);

        end

        %{
            Snapshot of the particle cloud for one frame: positions colored by
            weight, plus the velocity cloud and the weight histogram. This is
            the plot to reach for when the filter "loses" a target.

            fig = v.particle_cloud(Yk, ...)
              Yk  5 x N particle set for a single frame
            Options
              'Weights'  1 x N weights
              'Signal'   npx x npx frame to draw underneath
              'Truth'    4 x 1 (or 4 x T) truth state for that frame
              'Est'      4 x 1 point estimate
              'ShowDead' also draw the E = 0 particles, in grey
        %}
        function fig = particle_cloud(obj, Yk, varargin)

            p = obj.common_parser();
            addParameter(p, 'Weights', []);
            addParameter(p, 'Signal', []);
            addParameter(p, 'Truth', []);
            addParameter(p, 'Est', []);
            addParameter(p, 'ShowDead', false, @islogical);
            parse(p, varargin{:})
            R = p.Results;

            du = obj.resolve_data_units(R);
            [ix, iy] = obj.pos_rows(Yk);
            N = size(Yk, 2);

            if size(Yk,1) >= 5
                alive = Yk(5,:) ~= 0;
            else
                alive = true(1, N);
            end

            w = R.Weights;
            if isempty(w)
                w = ones(1, N) / N;
            end
            w = reshape(w, 1, []);

            fig = obj.new_figure('Particle Cloud', [1300 460]);
            tl = tiledlayout(fig, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

            %%% Positions ------------------------------------------------------
            ax = nexttile(tl);
            if ~isempty(R.Signal)
                obj.draw_image(ax, R.Signal, R.Units, R.Colormap, []);
                hold(ax, 'on');
            end
            if R.ShowDead && any(~alive)
                [xd, yd] = obj.convert_xy(Yk(ix,~alive), Yk(iy,~alive), du, R.Units);
                scatter(ax, xd, yd, 6, [0.6 0.6 0.6], 'filled', ...
                    'MarkerFaceAlpha', 0.25, 'DisplayName', 'E = 0');
                hold(ax, 'on');
            end
            if any(alive)
                [xa, ya] = obj.convert_xy(Yk(ix,alive), Yk(iy,alive), du, R.Units);
                wa = w(alive);
                sz = 8 + 60 * obj.unit_scale(wa);
                scatter(ax, xa, ya, sz, wa, 'filled', 'MarkerFaceAlpha', 0.6, ...
                    'DisplayName', 'E = 1');
                hold(ax, 'on');
                cb = colorbar(ax);
                cb.Label.String = 'weight';
            end
            obj.overlay_states(ax, R.Truth, du, R.Units, obj.truthColor, 'o', 'Truth');
            obj.overlay_states(ax, R.Est, du, R.Units, obj.estColor, 'x', 'Estimate');
            obj.style_scene(ax, R.Units);
            obj.maybe_legend(ax);
            title(ax, sprintf('Positions (%d / %d existing)', nnz(alive), N), ...
                'FontSize', obj.fs);

            %%% Velocities -----------------------------------------------------
            ax = nexttile(tl);
            if size(Yk,1) >= 4 && numel(obj.stateIdx) == 2 && obj.stateIdx(1) == 1
                vx = Yk(2, alive);
                vy = Yk(4, alive);
                scatter(ax, vx, vy, 10, obj.particleColor, 'filled', ...
                    'MarkerFaceAlpha', 0.4);
                grid(ax, 'on');
                axis(ax, 'equal');
                xlabel(ax, 'v_x', 'FontSize', obj.fs);
                ylabel(ax, 'v_y', 'FontSize', obj.fs);
                title(ax, 'Velocity Cloud', 'FontSize', obj.fs);
            else
                text(ax, 0.5, 0.5, 'no velocity states', 'Units', 'normalized', ...
                    'HorizontalAlignment', 'center');
                axis(ax, 'off');
            end

            %%% Weights --------------------------------------------------------
            ax = nexttile(tl);
            wa = w(w > 0);
            if isempty(wa)
                text(ax, 0.5, 0.5, 'all weights zero', 'Units', 'normalized', ...
                    'HorizontalAlignment', 'center');
                axis(ax, 'off');
            else
                histogram(ax, log10(wa), 40, 'FaceColor', obj.particleColor);
                grid(ax, 'on');
                xlabel(ax, 'log_{10} weight', 'FontSize', obj.fs);
                ylabel(ax, 'count', 'FontSize', obj.fs);
                title(ax, sprintf('Weights  (ESS/N = %.3f)', obj.ess(w(:)) / N), ...
                    'FontSize', obj.fs);
            end

            if ~isempty(R.Title)
                title(tl, R.Title, 'FontSize', obj.fs + 2);
            end
            obj.save_figure(fig, R.Save);

        end

        %{
            Particle density (2-D weighted histogram over the pixel grid) next
            to the measurement it came from. Shows whether the cloud is actually
            sitting on the energy, and exposes multi-modality that a mean
            estimate hides.

            fig = v.particle_density(Yk, ...)
            Options
              'Weights'  1 x N weights
              'Signal'   npx x npx frame to compare against
              'Bins'     grid coarsening factor (default 2 -> 64 x 64 bins)
              'Truth'
        %}
        function fig = particle_density(obj, Yk, varargin)

            p = obj.common_parser();
            addParameter(p, 'Weights', []);
            addParameter(p, 'Signal', []);
            addParameter(p, 'Bins', 2);
            addParameter(p, 'Truth', []);
            parse(p, varargin{:})
            R = p.Results;

            du = obj.resolve_data_units(R);
            [ix, iy] = obj.pos_rows(Yk);
            N = size(Yk, 2);
            if size(Yk,1) >= 5
                alive = Yk(5,:) ~= 0;
            else
                alive = true(1, N);
            end

            w = R.Weights;
            if isempty(w)
                w = ones(1, N);
            end
            w = reshape(w, 1, []);

            %%% Bin in pixel space so the density lines up with the image grid
            [xp, yp] = obj.convert_xy(Yk(ix,alive), Yk(iy,alive), du, 'pixels');
            nb = max(1, round(obj.npx / R.Bins));
            edges = linspace(0.5, obj.npx + 0.5, nb + 1);
            D = zeros(nb, nb);
            xb = discretize(xp, edges);
            yb = discretize(yp, edges);
            wa = w(alive);
            valid = ~isnan(xb) & ~isnan(yb);
            for n = find(valid)
                D(yb(n), xb(n)) = D(yb(n), xb(n)) + wa(n); % row = y, col = x
            end
            if max(D(:)) > 0
                D = D / max(D(:));
            end

            fig = obj.new_figure('Particle Density', [1000 460]);
            nTiles = 1 + ~isempty(R.Signal);
            tl = tiledlayout(fig, 1, nTiles, 'TileSpacing', 'compact', 'Padding', 'compact');

            if ~isempty(R.Signal)
                ax = nexttile(tl);
                obj.draw_image(ax, R.Signal, R.Units, R.Colormap, []);
                hold(ax, 'on');
                obj.overlay_states(ax, R.Truth, du, R.Units, obj.truthColor, 'o', 'Truth');
                obj.style_scene(ax, R.Units);
                colorbar(ax);
                title(ax, 'Measurement', 'FontSize', obj.fs);
            end

            ax = nexttile(tl);
            %%% Draw the coarse density on the same physical extent
            if strcmpi(R.Units, 'meters')
                xv = linspace(obj.xgrid(1), obj.xgrid(end), nb);
                yv = linspace(obj.ygrid(1), obj.ygrid(end), nb);
            else
                xv = linspace(1, obj.npx, nb);
                yv = linspace(1, obj.npx, nb);
            end
            imagesc(ax, xv, yv, D);
            axis(ax, 'xy');
            colormap(ax, 'hot');
            hold(ax, 'on');
            obj.overlay_states(ax, R.Truth, du, R.Units, [0.2 0.8 1.0], 'o', 'Truth');
            obj.style_scene(ax, R.Units);
            colorbar(ax);
            title(ax, 'Weighted Particle Density', 'FontSize', obj.fs);

            if ~isempty(R.Title)
                title(tl, R.Title, 'FontSize', obj.fs + 2);
            end
            obj.save_figure(fig, R.Save);

        end

        %% ------------------------------------------------------------------
        %% Performance / validation plots
        %% ------------------------------------------------------------------

        %{
            Per-axis and total position error vs time.

            [fig, err] = v.position_error(truth, est, ...)
              err  T x K euclidean position error, also returned for reuse in
                   rmse_mc across Monte Carlo runs
            Options
              'Mask'  1 x K (or T x K) logical, blank the error where the track
                      is not declared
              'Time'
        %}
        function [fig, err] = position_error(obj, truth, est, varargin)

            p = obj.common_parser();
            addParameter(p, 'Mask', []);
            addParameter(p, 'Time', []);
            parse(p, varargin{:})
            R = p.Results;

            du = obj.resolve_data_units(R);
            tru = obj.to_cell(truth);
            es = obj.to_cell(est);
            T = min(numel(tru), numel(es));
            assert(T > 0, 'position_error: need at least one truth/estimate pair');

            K = min(size(tru{1},2), size(es{1},2));
            err = nan(T, K);
            ex = nan(T, K);
            ey = nan(T, K);
            mask = obj.to_mask(R.Mask, T);

            for t = 1:T
                [xt_, yt_] = obj.track_xy(tru{t}, du, R.Units);
                [xe_, ye_] = obj.track_xy(es{t}, du, R.Units);
                n = min([K, numel(xt_), numel(xe_)]);
                ex(t,1:n) = xe_(1:n) - xt_(1:n);
                ey(t,1:n) = ye_(1:n) - yt_(1:n);
                m = obj.mask_for(mask, t, K);
                ex(t, ~m) = NaN;
                ey(t, ~m) = NaN;
                err(t,:) = hypot(ex(t,:), ey(t,:));
            end

            kvec = R.Time;
            if isempty(kvec)
                kvec = 1:K;
            end
            cols = obj.palette(T);
            usuf = char(obj.unit_suffix(R.Units));

            fig = obj.new_figure('Position Error', [800 800]);
            tl = tiledlayout(fig, 3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

            series = {ex, ey, err};
            names = {['x error [' usuf ']'], ['y error [' usuf ']'], ...
                     ['|error| [' usuf ']']};
            for s = 1:3
                ax = nexttile(tl);
                for t = 1:T
                    plot(ax, kvec, series{s}(t,:), '-', 'Color', cols(t,:), ...
                        'LineWidth', obj.lw, 'DisplayName', sprintf('Target %d', t));
                    hold(ax, 'on');
                end
                if s < 3
                    yline(ax, 0, 'k:');
                else
                    g = isfinite(err);
                    if any(g(:))
                        yline(ax, sqrt(mean(err(g).^2)), 'k--', 'LineWidth', 1, ...
                            'Label', sprintf('RMSE = %.3g', sqrt(mean(err(g).^2))));
                    end
                end
                grid(ax, 'on');
                ylabel(ax, names{s}, 'FontSize', obj.fs);
                if s == 3
                    xlabel(ax, 'k', 'FontSize', obj.fs);
                end
                if s == 1 && T > 1
                    legend(ax, 'Location', 'best');
                end
            end

            if ~isempty(R.Title)
                title(tl, R.Title, 'FontSize', obj.fs + 2);
            end
            obj.save_figure(fig, R.Save);

        end

        %{
            Monte Carlo RMSE with a spread band. Feed it the per-run error
            traces from position_error and it gives the plot you want in a
            results section.

            fig = v.rmse_mc(errStack, ...)
              errStack  K x M matrix (K frames, M runs), or a 1 x M cell of
                        1 x K error traces
            Options
              'Band'   'std' | 'quantile' | 'none' (default 'quantile')
              'Time'
              'Label'  legend entry, for overlaying several configurations by
                       passing the same 'Axes'
        %}
        function fig = rmse_mc(obj, errStack, varargin)

            p = obj.common_parser();
            addParameter(p, 'Band', 'quantile');
            addParameter(p, 'Time', []);
            addParameter(p, 'Label', 'RMSE');
            addParameter(p, 'Color', obj.estColor);
            parse(p, varargin{:})
            R = p.Results;

            if iscell(errStack)
                M = numel(errStack);
                K = max(cellfun(@numel, errStack));
                E = nan(K, M);
                for m = 1:M
                    e = errStack{m}(:);
                    E(1:numel(e), m) = e;
                end
            else
                E = errStack;
                if size(E,1) == 1
                    E = E(:); % single run passed as a row
                end
            end

            K = size(E,1);
            M = size(E,2);
            kvec = R.Time;
            if isempty(kvec)
                kvec = 1:K;
            end
            kvec = kvec(:)';

            %%% RMSE across runs at each k, ignoring frames where the target
            %%% was absent / not declared (NaN)
            rmseK = sqrt(mean(E.^2, 2, 'omitnan'))';
            valid = ~isnan(rmseK);

            [fig, ax] = obj.get_axes(R, 'Monte Carlo RMSE');

            switch lower(R.Band)
                case 'std'
                    s = std(E, 0, 2, 'omitnan')';
                    lo = max(0, rmseK - s);
                    hi = rmseK + s;
                case 'quantile'
                    lo = obj.row_pctile(E, 0.25);
                    hi = obj.row_pctile(E, 0.75);
                otherwise
                    lo = [];
                    hi = [];
            end

            if ~isempty(lo)
                b = valid & isfinite(lo) & isfinite(hi);
                fill(ax, [kvec(b), fliplr(kvec(b))], [lo(b), fliplr(hi(b))], ...
                    R.Color, 'FaceAlpha', 0.15, 'EdgeColor', 'none', ...
                    'HandleVisibility', 'off');
                hold(ax, 'on');
            end
            plot(ax, kvec, rmseK, '-', 'Color', R.Color, 'LineWidth', obj.lw, ...
                'DisplayName', sprintf('%s (M = %d)', R.Label, M));
            hold(ax, 'on');
            grid(ax, 'on');
            xlabel(ax, 'k', 'FontSize', obj.fs);
            ylabel(ax, ['RMSE [' char(obj.unit_suffix(R.Units)) ']'], 'FontSize', obj.fs);
            obj.maybe_legend(ax);
            obj.set_title(ax, R.Title, 'Monte Carlo Position RMSE');
            obj.save_figure(fig, R.Save);

        end

        %{
            OSPA distance vs time, split into its localization and cardinality
            components. The right validation metric once births/deaths and
            multiple targets are in play, because it penalizes both position
            error and the wrong number of tracks on one scale.

            [fig, d] = v.ospa(truth, est, ...)
              truth/est  cell or 3-D stacks of tracks; NaN marks "not present"
            Options
              'c'      cutoff, in the plotting units (default 1/4 of the scene)
              'p'      order (default 2)
              'Mask'   declaration mask applied to the estimates
              'Time'
        %}
        function [fig, d] = ospa(obj, truth, est, varargin)

            p = obj.common_parser();
            addParameter(p, 'c', []);
            addParameter(p, 'p', 2);
            addParameter(p, 'Mask', []);
            addParameter(p, 'Time', []);
            parse(p, varargin{:})
            R = p.Results;

            du = obj.resolve_data_units(R);
            tru = obj.to_cell(truth);
            es = obj.to_cell(est);

            c = R.c;
            if isempty(c)
                if strcmpi(R.Units, 'meters')
                    c = obj.Lscene / 4;
                else
                    c = obj.npx / 4;
                end
            end

            K = 0;
            for t = 1:numel(tru), K = max(K, size(tru{t},2)); end
            for t = 1:numel(es),  K = max(K, size(es{t},2));  end

            mask = obj.to_mask(R.Mask, numel(es));

            d = struct('total', nan(1,K), 'loc', nan(1,K), 'card', nan(1,K));

            for k = 1:K
                X = obj.states_at_k(tru, k, du, R.Units, []);
                Y = obj.states_at_k(es, k, du, R.Units, mask);
                [d.total(k), d.loc(k), d.card(k)] = obj.ospa_dist(X, Y, c, R.p);
            end

            kvec = R.Time;
            if isempty(kvec)
                kvec = 1:K;
            end

            [fig, ax] = obj.get_axes(R, 'OSPA');
            plot(ax, kvec, d.total, '-', 'Color', obj.estColor, ...
                'LineWidth', obj.lw, 'DisplayName', 'OSPA');
            hold(ax, 'on');
            plot(ax, kvec, d.loc, '--', 'Color', obj.particleColor, ...
                'LineWidth', 1, 'DisplayName', 'localization');
            plot(ax, kvec, d.card, ':', 'Color', [0.4 0.4 0.4], ...
                'LineWidth', 1.2, 'DisplayName', 'cardinality');
            grid(ax, 'on');
            ylim(ax, [0 c * 1.05]);
            xlabel(ax, 'k', 'FontSize', obj.fs);
            ylabel(ax, ['OSPA [' char(obj.unit_suffix(R.Units)) ']'], 'FontSize', obj.fs);
            legend(ax, 'Location', 'best');
            obj.set_title(ax, R.Title, sprintf('OSPA  (c = %.3g, p = %d)', c, R.p));
            obj.save_figure(fig, R.Save);

        end

        %{
            Particle filter health over the whole run: effective sample size,
            largest normalized weight, and the weight distribution at a few
            frames. Flat-lining ESS or a max weight near 1 means the cloud has
            collapsed and any track output is luck.

            fig = v.pf_diagnostics(W, ...)
              W  N x K weight history (post-normalization, pre-resample)
            Options
              'Particles'  5 x N x K history, adds unique-particle count
              'Frames'     frames to histogram (default 3 evenly spaced)
              'Time'
        %}
        function fig = pf_diagnostics(obj, W, varargin)

            p = obj.common_parser();
            addParameter(p, 'Particles', []);
            addParameter(p, 'Frames', []);
            addParameter(p, 'Time', []);
            parse(p, varargin{:})
            R = p.Results;

            [N, K] = size(W);
            kvec = R.Time;
            if isempty(kvec)
                kvec = 1:K;
            end

            essK = obj.ess(W) ./ N;
            maxW = max(W, [], 1);

            frames = R.Frames;
            if isempty(frames)
                frames = unique(round(linspace(1, K, min(3, K))));
            end

            fig = obj.new_figure('PF Diagnostics', [1100 700]);
            tl = tiledlayout(fig, 2, max(numel(frames), 2), ...
                'TileSpacing', 'compact', 'Padding', 'compact');

            %%% ESS + max weight across the top row
            ax = nexttile(tl, 1, [1 max(numel(frames),2)]);
            yyaxis(ax, 'left');
            plot(ax, kvec, essK, '-', 'LineWidth', obj.lw);
            ylabel(ax, 'ESS / N', 'FontSize', obj.fs);
            ylim(ax, [0 1]);
            yyaxis(ax, 'right');
            plot(ax, kvec, maxW, '-', 'LineWidth', 1);
            ylabel(ax, 'max weight', 'FontSize', obj.fs);
            grid(ax, 'on');
            xlabel(ax, 'k', 'FontSize', obj.fs);
            title(ax, 'Weight Degeneracy', 'FontSize', obj.fs);

            %%% Weight histograms along the bottom
            for ii = 1:numel(frames)
                ax = nexttile(tl);
                k = frames(ii);
                w = W(:,k);
                w = w(w > 0);
                if isempty(w)
                    text(ax, 0.5, 0.5, sprintf('k = %d: all zero', k), ...
                        'Units', 'normalized', 'HorizontalAlignment', 'center');
                    axis(ax, 'off');
                else
                    histogram(ax, log10(w), 40, 'FaceColor', obj.particleColor);
                    grid(ax, 'on');
                    xlabel(ax, 'log_{10} w', 'FontSize', obj.fs);
                    title(ax, sprintf('k = %d, ESS/N = %.3f', k, essK(k)), ...
                        'FontSize', obj.fs);
                end
            end

            if ~isempty(R.Particles)
                Yc = obj.to_particle_cell(R.Particles);
                uniq = zeros(1, size(Yc{1},3));
                for k = 1:numel(uniq)
                    uniq(k) = size(unique(Yc{1}(:,:,k)', 'rows'), 1);
                end
                obj.debug_print(sprintf("UNIQUE PARTICLES: min %d, max %d of %d", ...
                    min(uniq), max(uniq), N));
            end

            if ~isempty(R.Title)
                title(tl, R.Title, 'FontSize', obj.fs + 2);
            end
            obj.save_figure(fig, R.Save);

        end

    end

    methods(Static)

        %{
            Empty results struct documenting what plot_TBD accepts. Fill in
            whatever the run produced and leave the rest empty.
        %}
        function res = results_template()
            res = struct( ...
                'signals',   [], ... % npx x npx x K measurement stack
                'truth',     [], ... % 4 x K x T (or cell) ground truth states
                'Etruth',    [], ... % T x K true existence flags
                'est',       [], ... % 4 x K x T (or cell) point estimates
                'pE',        [], ... % T x K estimated existence probability
                'particles', [], ... % 5 x N x K (or cell / 5 x N x K x T)
                'weights',   [], ... % N x K normalized weights
                't',         [], ... % 1 x K time stamps (seconds); [] -> use k
                'pEthresh',  0.5);
        end

    end

    methods(Hidden)

        %% ------------------------------------------------------------------
        %% Option parsing
        %% ------------------------------------------------------------------

        %%% Options every plotting method understands
        function p = common_parser(obj)
            p = inputParser;
            p.KeepUnmatched = false;
            addParameter(p, 'Units', obj.units);
            addParameter(p, 'DataUnits', '');
            addParameter(p, 'Axes', []);
            addParameter(p, 'Title', '');
            addParameter(p, 'Save', '');
            addParameter(p, 'Colormap', obj.cmap);
        end

        %%% Units the incoming data is expressed in; defaults to the axis units
        %%% so that nothing is converted unless the caller says so.
        function du = resolve_data_units(obj, R)
            if isempty(R.DataUnits)
                du = R.Units;
            else
                du = obj.check_units(R.DataUnits);
            end
        end

        function u = check_units(~, u)
            u = lower(char(string(u)));
            assert(any(strcmp(u, {'meters', 'pixels'})), ...
                'Units must be ''meters'' or ''pixels''');
        end

        %% ------------------------------------------------------------------
        %% Figure / axes plumbing
        %% ------------------------------------------------------------------

        %%% New figure. sz is an optional [w h] in pixels; multi-panel figures
        %%% need the extra room or the tick labels collide.
        function fig = new_figure(~, name, sz)
            fig = figure('Name', name, 'Color', 'w');
            if nargin >= 3 && ~isempty(sz)
                pos = get(fig, 'Position');
                set(fig, 'Position', [pos(1), max(50, pos(2) - (sz(2) - pos(4))), sz(1), sz(2)]);
            end
        end

        %%% Either draw into the axes the caller supplied, or make a new figure
        function [fig, ax] = get_axes(obj, R, name)
            if ~isempty(R.Axes)
                ax = R.Axes;
                fig = ancestor(ax, 'figure');
            else
                fig = obj.new_figure(name);
                ax = axes(fig); %#ok<LAXES>
            end
        end

        %%% Draw a signal with the correct orientation and extent.
        %%% signal(row,col) = signal(y,x), matching simulator.m
        function draw_image(obj, ax, signal, units, cmapName, clim_)
            if isempty(signal)
                return
            end
            if strcmpi(units, 'meters')
                xv = obj.xgrid;
                yv = obj.ygrid;
            else
                xv = 1:size(signal,2);
                yv = 1:size(signal,1);
            end
            imagesc(ax, xv, yv, signal);
            axis(ax, 'xy');
            colormap(ax, cmapName);
            if ~isempty(clim_)
                caxis(ax, clim_); %#ok<CAXIS>
            end
        end

        %%% Common scene formatting: equal aspect, full scene extent, labels
        function style_scene(obj, ax, units)
            axis(ax, 'equal');
            if strcmpi(units, 'meters')
                xlim(ax, [obj.xgrid(1), obj.xgrid(end)]);
                ylim(ax, [obj.ygrid(1), obj.ygrid(end)]);
                xlabel(ax, 'x [m]', 'FontSize', obj.fs);
                ylabel(ax, 'y [m]', 'FontSize', obj.fs);
            else
                xlim(ax, [1, obj.npx]);
                ylim(ax, [1, obj.npx]);
                xlabel(ax, 'x [px]', 'FontSize', obj.fs);
                ylabel(ax, 'y [px]', 'FontSize', obj.fs);
            end
            set(ax, 'FontSize', obj.fs);
            grid(ax, 'on');
        end

        function s = unit_suffix(~, units)
            if strcmpi(units, 'meters')
                s = "m";
            else
                s = "px";
            end
        end

        function set_title(obj, ax, given, fallback)
            if isempty(given)
                title(ax, fallback, 'FontSize', obj.fs);
            else
                title(ax, given, 'FontSize', obj.fs);
            end
        end

        %%% Only draw a legend if something asked to be in it
        function maybe_legend(~, ax, loc)
            if nargin < 3 || isempty(loc)
                loc = 'best';
            end
            h = findobj(ax, '-property', 'DisplayName');
            if isempty(h)
                return
            end
            names = get(h, 'DisplayName');
            if ischar(names)
                names = {names};
            end
            if any(~cellfun(@isempty, names))
                legend(ax, 'Location', loc);
            end
        end

        function cols = palette(~, n)
            base = [0.85 0.10 0.10;
                    0.10 0.45 0.85;
                    0.15 0.65 0.30;
                    0.90 0.60 0.05;
                    0.55 0.25 0.75;
                    0.20 0.70 0.70];
            reps = ceil(n / size(base,1));
            cols = repmat(base, reps, 1);
            cols = cols(1:max(n,1), :);
        end

        function nm = name_for(~, base, t, total)
            if total <= 1
                nm = base;
            else
                nm = sprintf('%s %d', base, t);
            end
        end

        %%% Work out the export format from the path / explicit override
        function [fmt, outFile] = resolve_format(~, savePath, fmtIn)
            outFile = char(savePath);
            if isempty(outFile)
                fmt = "";
                return
            end
            [d, ~, ext] = fileparts(outFile);
            if ~isempty(d) && ~isfolder(d)
                mkdir(d);
            end
            if ~isempty(fmtIn)
                fmt = lower(string(fmtIn));
            else
                switch lower(ext)
                    case '.gif', fmt = "gif";
                    case {'.mp4', '.m4v'}, fmt = "mp4";
                    case {'.png', '.jpg', '.jpeg'}, fmt = "png";
                    otherwise, fmt = "gif";
                end
            end
            if isempty(ext)
                outFile = [outFile '.' char(fmt)];
            end
        end

        function save_figure(obj, fig, savePath)
            if isempty(savePath)
                return
            end
            savePath = char(savePath);
            [d, ~, ext] = fileparts(savePath);
            if ~isempty(d) && ~isfolder(d)
                mkdir(d);
            end
            if isempty(ext)
                savePath = [savePath '.png'];
            end
            exportgraphics(fig, savePath, 'Resolution', obj.dpi);
            obj.debug_print("SAVED FIGURE TO " + string(savePath));
        end

        %% ------------------------------------------------------------------
        %% Data normalization
        %% ------------------------------------------------------------------

        %%% Normalize tracks to a 1 x T cell of D x K
        function C = to_cell(~, D)
            if isempty(D)
                C = {};
                return
            end
            if iscell(D)
                C = reshape(D, 1, []);
                return
            end
            if ndims(D) == 3
                T = size(D,3);
                C = cell(1,T);
                for t = 1:T
                    C{t} = D(:,:,t);
                end
            else
                C = {D};
            end
        end

        %%% Normalize particle histories to a 1 x T cell of 5 x N x K
        function C = to_particle_cell(~, D)
            if isempty(D)
                C = {};
                return
            end
            if iscell(D)
                C = reshape(D, 1, []);
                return
            end
            if ndims(D) == 4
                T = size(D,4);
                C = cell(1,T);
                for t = 1:T
                    C{t} = D(:,:,:,t);
                end
            else
                C = {D};
            end
        end

        %%% Force a T x K matrix out of a vector / matrix / cell
        function M = to_rowmat(~, D)
            if isempty(D)
                M = [];
                return
            end
            if iscell(D)
                M = cell2mat(reshape(D, [], 1));
                return
            end
            if isvector(D)
                M = reshape(double(D), 1, []);
            else
                M = double(D);
            end
        end

        function M = to_mask(obj, D, T)
            if isempty(D)
                M = [];
                return
            end
            M = logical(obj.to_rowmat(D));
            if size(M,1) == 1 && T > 1
                M = repmat(M, T, 1);
            end
        end

        function m = mask_for(~, mask, t, K)
            if isempty(mask)
                m = true(1, K);
                return
            end
            row = mask(min(t, size(mask,1)), :);
            m = false(1, K);
            n = min(K, numel(row));
            m(1:n) = row(1:n);
        end

        %%% Rows of a state array holding [x; y]
        function [ix, iy] = pos_rows(obj, S)
            if size(S,1) == 2
                ix = 1;
                iy = 2;
            else
                ix = obj.stateIdx(1);
                iy = obj.stateIdx(2);
            end
        end

        function r = state_row(obj, S, which)
            [ix, iy] = obj.pos_rows(S);
            if which == 1
                r = ix;
            else
                r = iy;
            end
        end

        %%% Pull the (x,y) series out of a D x K track, in the requested units
        function [x, y] = track_xy(obj, S, fromUnits, toUnits)
            [ix, iy] = obj.pos_rows(S);
            [x, y] = obj.convert_xy(S(ix,:), S(iy,:), fromUnits, toUnits);
        end

        function v = axis_series(obj, S, d, fromUnits, toUnits)
            [x, y] = obj.track_xy(S, fromUnits, toUnits);
            if d == 1
                v = x;
            else
                v = y;
            end
        end

        %%% Pixel index <-> meters. Pixel 1 sits at xgrid(1).
        function [X, Y] = convert_xy(obj, X, Y, from, to)
            from = lower(char(string(from)));
            to = lower(char(string(to)));
            if strcmp(from, to)
                return
            end
            if strcmp(from, 'pixels')
                X = obj.xgrid(1) + (X - 1) * obj.dx;
                Y = obj.ygrid(1) + (Y - 1) * obj.dy;
            else
                X = (X - obj.xgrid(1)) / obj.dx + 1;
                Y = (Y - obj.ygrid(1)) / obj.dy + 1;
            end
        end

        function v = nan_where(~, v, cond)
            v(cond) = NaN;
        end

        %% ------------------------------------------------------------------
        %% Overlay helpers
        %% ------------------------------------------------------------------

        %%% Scatter the existing particles onto ax, shaded by weight if given
        function overlay_particles(obj, ax, Yk, w, du, units)
            if isempty(Yk)
                return
            end
            [ix, iy] = obj.pos_rows(Yk);
            N = size(Yk,2);
            if size(Yk,1) >= 5
                alive = Yk(5,:) ~= 0;
            else
                alive = true(1, N);
            end
            if ~any(alive)
                return
            end
            [x, y] = obj.convert_xy(Yk(ix,alive), Yk(iy,alive), du, units);
            if isempty(w)
                plot(ax, x, y, '.', 'Color', obj.particleColor, ...
                    'MarkerSize', 4, 'DisplayName', 'Particles');
            else
                wa = reshape(w, 1, []);
                wa = wa(alive);
                sz = 6 + 50 * obj.unit_scale(wa);
                scatter(ax, x, y, sz, wa, 'filled', 'MarkerFaceAlpha', 0.5, ...
                    'DisplayName', 'Particles');
            end
        end

        function overlay_states(obj, ax, S, du, units, color, marker, name)
            if isempty(S)
                return
            end
            [ix, iy] = obj.pos_rows(S);
            [x, y] = obj.convert_xy(S(ix,:), S(iy,:), du, units);
            plot(ax, x, y, marker, 'Color', color, 'MarkerSize', obj.ms + 4, ...
                'LineWidth', 1.5, 'LineStyle', 'none', 'DisplayName', name);
        end

        %%% Mark where each track sits at frame k
        function mark_tracks_at_k(obj, ax, tracks, k, du, units, color, marker)
            for t = 1:numel(tracks)
                S = tracks{t};
                if k > size(S,2)
                    continue
                end
                [ix, iy] = obj.pos_rows(S);
                if isnan(S(ix,k)) || isnan(S(iy,k))
                    continue
                end
                [x, y] = obj.convert_xy(S(ix,k), S(iy,k), du, units);
                plot(ax, x, y, marker, 'Color', color, 'MarkerSize', obj.ms + 4, ...
                    'LineWidth', 1.5, 'LineStyle', 'none');
            end
        end

        %%% Draw the last `trail` samples of each track up to frame k
        function draw_trails(obj, ax, tracks, k, trail, du, units, color, style)
            if isempty(tracks)
                return
            end
            if isinf(trail)
                k0 = 1;
            else
                k0 = max(1, k - trail);
            end
            for t = 1:numel(tracks)
                S = tracks{t};
                kk = min(k, size(S,2));
                if kk < k0
                    continue
                end
                [ix, iy] = obj.pos_rows(S);
                [x, y] = obj.convert_xy(S(ix,k0:kk), S(iy,k0:kk), du, units);
                plot(ax, x, y, style, 'Color', color, 'LineWidth', obj.lw);
            end
        end

        %%% Scale a vector into [0,1]; flat input maps to 0.5
        function u = unit_scale(~, v)
            lo = min(v);
            hi = max(v);
            if ~isfinite(lo) || ~isfinite(hi) || hi <= lo
                u = 0.5 * ones(size(v));
            else
                u = (v - lo) / (hi - lo);
            end
        end

        %%% Percentile without the Statistics Toolbox (linear interpolation,
        %%% NaNs dropped), applied down each row of a K x M matrix
        function q = row_pctile(~, E, pct)
            K = size(E,1);
            q = nan(1,K);
            for k = 1:K
                v = sort(E(k, ~isnan(E(k,:))));
                n = numel(v);
                if n == 0
                    continue
                elseif n == 1
                    q(k) = v;
                    continue
                end
                pos = 1 + pct * (n - 1);
                lo = floor(pos);
                hi = ceil(pos);
                q(k) = v(lo) + (pos - lo) * (v(hi) - v(lo));
            end
        end

        %% ------------------------------------------------------------------
        %% Metric helpers
        %% ------------------------------------------------------------------

        %%% Effective sample size per column of a weight matrix
        function e = ess(~, W)
            if isvector(W)
                W = W(:);
            end
            s = sum(W, 1);
            s(s == 0) = 1;
            Wn = W ./ s;
            e = 1 ./ sum(Wn.^2, 1);
        end

        %%% 2 x n matrix of the targets present at frame k, in plotting units
        function P = states_at_k(obj, tracks, k, du, units, mask)
            P = zeros(2,0);
            for t = 1:numel(tracks)
                S = tracks{t};
                if k > size(S,2)
                    continue
                end
                if ~isempty(mask)
                    row = mask(min(t, size(mask,1)), :);
                    if k > numel(row) || ~row(k)
                        continue
                    end
                end
                [ix, iy] = obj.pos_rows(S);
                if isnan(S(ix,k)) || isnan(S(iy,k))
                    continue
                end
                [x, y] = obj.convert_xy(S(ix,k), S(iy,k), du, units);
                P(:,end+1) = [x; y]; %#ok<AGROW>
            end
        end

        %%% OSPA distance between two point sets, split into localization and
        %%% cardinality parts (Schuhmacher et al.)
        function [d, dLoc, dCard] = ospa_dist(~, X, Y, c, pOrd)
            m = size(X,2);
            n = size(Y,2);

            if m == 0 && n == 0
                d = 0; dLoc = 0; dCard = 0;
                return
            end
            if m == 0 || n == 0
                d = c; dLoc = 0; dCard = c;
                return
            end

            %%% Cost matrix, cut off at c
            D = zeros(m,n);
            for i = 1:m
                D(i,:) = min(c, sqrt(sum((X(:,i) - Y).^2, 1)));
            end
            Dp = D.^pOrd;

            %%% Optimal assignment over the smaller cardinality
            if exist('matchpairs', 'file')
                unmatched = c^pOrd; % never worth leaving a pair unmatched below cutoff
                M = matchpairs(Dp, unmatched);
                cost = 0;
                for r = 1:size(M,1)
                    cost = cost + Dp(M(r,1), M(r,2));
                end
                %%% Any pair matchpairs declined to match (only possible on a
                %%% tie at the cutoff) still costs the cutoff
                cost = cost + (min(m,n) - size(M,1)) * unmatched;
            else
                %%% Greedy fallback if matchpairs is unavailable
                A = Dp;
                cost = 0;
                for r = 1:min(m,n)
                    [v, li] = min(A(:));
                    [i, j] = ind2sub(size(A), li);
                    cost = cost + v;
                    A(i,:) = Inf;
                    A(:,j) = Inf;
                end
            end

            N = max(m,n);
            dLoc = (cost / N)^(1/pOrd);
            dCard = ((c^pOrd) * abs(m - n) / N)^(1/pOrd);
            d = ((cost + (c^pOrd) * abs(m - n)) / N)^(1/pOrd);
        end

        %% ------------------------------------------------------------------
        %% plot_TBD helpers
        %% ------------------------------------------------------------------

        %%% Merge a partial results struct onto the template
        function res = fill_results(~, res)
            tmpl = visualize.results_template();
            f = fieldnames(tmpl);
            for i = 1:numel(f)
                if ~isfield(res, f{i}) || isempty(res.(f{i}))
                    if ~isfield(res, f{i})
                        res.(f{i}) = tmpl.(f{i});
                    elseif isempty(res.(f{i})) && ~isempty(tmpl.(f{i}))
                        res.(f{i}) = tmpl.(f{i});
                    end
                end
            end
        end

        %%% Use the supplied point estimate if there is one, otherwise take the
        %%% MMSE over the existing particles
        function [est, pE] = resolve_estimates(obj, res)
            est = obj.to_cell(res.est);
            pE = obj.to_rowmat(res.pE);

            if ~isempty(est) && ~isempty(pE)
                return
            end

            Yc = obj.to_particle_cell(res.particles);
            if isempty(Yc)
                return
            end

            for t = 1:numel(Yc)
                e = obj.particle_estimate(Yc{t}, res.weights);
                if isempty(res.est)
                    S = nan(4, numel(e.x));
                    [ix, iy] = obj.pos_rows(zeros(4,1));
                    S(ix,:) = e.x;
                    S(iy,:) = e.y;
                    est{t} = S; %#ok<AGROW>
                end
                if isempty(res.pE)
                    pE(t,:) = e.pE; %#ok<AGROW>
                end
            end
        end

        %%% MMSE position estimate + existence probability from a particle set
        function e = particle_estimate(obj, Y, W)
            K = size(Y,3);
            N = size(Y,2);
            [ix, iy] = obj.pos_rows(Y);
            hasE = size(Y,1) >= 5;

            e.pE = nan(1,K);
            e.x = nan(1,K);
            e.y = nan(1,K);
            e.varx = nan(1,K);
            e.vary = nan(1,K);

            for k = 1:K
                if hasE
                    alive = Y(5,:,k) ~= 0;
                else
                    alive = true(1,N);
                end

                if isempty(W)
                    w = ones(1,N) / N;
                else
                    w = reshape(W(:,k), 1, []);
                    s = sum(w);
                    if s > 0
                        w = w / s;
                    else
                        w = ones(1,N) / N;
                    end
                end

                e.pE(k) = sum(w(alive));

                if ~any(alive)
                    continue
                end

                wa = w(alive);
                sa = sum(wa);
                if sa <= 0
                    wa = ones(1,nnz(alive)) / nnz(alive);
                else
                    wa = wa / sa;
                end

                xs = Y(ix,alive,k);
                ys = Y(iy,alive,k);
                e.x(k) = sum(wa .* xs);
                e.y(k) = sum(wa .* ys);
                e.varx(k) = sum(wa .* (xs - e.x(k)).^2);
                e.vary(k) = sum(wa .* (ys - e.y(k)).^2);
            end
        end

        %%% Shared existence panel, used by plot_TBD and existence()
        function existence_axes(obj, ax, kvec, pE, Etruth, thresh, tlab)
            cols = obj.palette(max(size(pE,1), 1));
            if ~isempty(Etruth)
                for t = 1:size(Etruth,1)
                    hh = stairs(ax, kvec(1:min(end,size(Etruth,2))), ...
                        double(Etruth(t,1:min(end,numel(kvec)))), '--', ...
                        'Color', obj.truthColor, 'LineWidth', 1.2, ...
                        'DisplayName', 'True existence');
                    if t > 1
                        set(hh, 'HandleVisibility', 'off');
                    end
                    hold(ax, 'on');
                end
            end
            for t = 1:size(pE,1)
                plot(ax, kvec(1:min(end,size(pE,2))), pE(t,1:min(end,numel(kvec))), ...
                    '-', 'Color', cols(t,:), 'LineWidth', obj.lw, ...
                    'DisplayName', obj.name_for('P(E_k)', t, size(pE,1)));
                hold(ax, 'on');
            end
            yline(ax, thresh, 'r:', 'LineWidth', 1, ...
                'DisplayName', 'Declare threshold');
            ylim(ax, [-0.05 1.05]);
            grid(ax, 'on');
            xlabel(ax, tlab, 'FontSize', obj.fs);
            ylabel(ax, 'P(exists)', 'FontSize', obj.fs);
            obj.maybe_legend(ax);
            title(ax, 'Target Existence', 'FontSize', obj.fs);
        end

        function K = n_steps(~, res, est, tru)
            K = 0;
            if ~isempty(res.signals)
                K = max(K, size(res.signals,3));
            end
            for t = 1:numel(est)
                K = max(K, size(est{t},2));
            end
            for t = 1:numel(tru)
                K = max(K, size(tru{t},2));
            end
            if ~isempty(res.weights)
                K = max(K, size(res.weights,2));
            end
        end

        function kvec = time_vector(~, res, K)
            if ~isempty(res.t)
                kvec = reshape(res.t, 1, []);
            else
                kvec = 1:K;
            end
        end

        function [lab, isTime] = time_label(~, res)
            isTime = ~isempty(res.t);
            if isTime
                lab = 'time [s]';
            else
                lab = 'k';
            end
        end

        %%% Euclidean position error, T x K
        function err = compute_error(obj, tru, est, du, units)
            T = min(numel(tru), numel(est));
            K = 0;
            for t = 1:T
                K = max(K, min(size(tru{t},2), size(est{t},2)));
            end
            err = nan(T, K);
            for t = 1:T
                [xt_, yt_] = obj.track_xy(tru{t}, du, units);
                [xe_, ye_] = obj.track_xy(est{t}, du, units);
                n = min([K, numel(xt_), numel(xe_)]);
                err(t,1:n) = hypot(xe_(1:n) - xt_(1:n), ye_(1:n) - yt_(1:n));
            end
        end

        %% ------------------------------------------------------------------
        %% Misc
        %% ------------------------------------------------------------------

        %%% If debug is on, will print str, ow it will do nothing.
        %%% Note the %s: str often carries Windows paths, and passing those as
        %%% a format string eats the backslash escapes.
        function [] = debug_print(obj, str)
            if obj.debug
                fprintf('%s\n', "[DEBUG][VISUALIZE]" + string(str))
            end
        end

    end

end
