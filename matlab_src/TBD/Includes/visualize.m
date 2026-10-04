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
%%%   - Particle histories are S x N x K with existence E in the LAST row
%%%     ([px;vx;py;vy;I;E], or the older [px;vx;py;vy;E]), S x N x K x T, or a
%%%     cell of S x N x K.
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
            addParameter(p, 'Colormap', 'turbo');
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
            % Magenta: turbo (the default signal colormap) runs blue -> cyan
            % -> green -> yellow -> red and never produces magenta/pink, so
            % particles stay visible against the signal at any intensity.
            obj.particleColor = [1.00 0.00 0.90];

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
              'SNR'         scalar or 1 x K signal SNR [dB]; appended to the
                            title, but only when a background is actually drawn
              'Units'/'DataUnits'/'Axes'/'Title'/'Save'
        %}
        function fig = trajectories(obj, tracks, varargin)

            p = obj.common_parser();
            addParameter(p, 'Truth', []);
            addParameter(p, 'Background', []);
            addParameter(p, 'ColorByTime', false, @islogical);
            addParameter(p, 'Mask', []);
            addParameter(p, 'Labels', {});
            addParameter(p, 'SNR', []);
            parse(p, varargin{:})
            R = p.Results;

            [fig, ax] = obj.get_axes(R, 'Trajectories');
            du = obj.resolve_data_units(R);

            %%% Optional energy background
            snr = [];
            if ~isempty(R.Background)
                bg = R.Background;
                if ndims(bg) == 3
                    bg = max(bg, [], 3); % max intensity projection over time
                end
                obj.draw_image(ax, bg, R.Units, R.Colormap, []);
                hold(ax, 'on');
                %%% Only label SNR when there is a signal on the axes to label
                snr = R.SNR;
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
            obj.set_title(ax, obj.signal_title(R.Title, '', snr), ...
                obj.signal_title('', 'Target Trajectories', snr));
            obj.save_figure(fig, R.Save);

        end

        %{
            Plot just a single mWidar image frame

            fig = v.signal_frame(signal, ...)

            Options
              'Truth'/'Est'   4 x 1 states (or 4 x T) to mark on the frame
              'Particles'     S x N particle set for frame k, overlaid as dots
              'Weights'       1 x N weights, used to size/shade the particles
              'CLim'          color limits, [] for auto
              'Colorbar'      logical, default true
              'SNR'           frame SNR [dB], appended to the title
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
            addParameter(p, 'SNR', []);
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
            obj.set_title(ax, obj.signal_title(R.Title, '', R.SNR), ...
                obj.signal_title('', 'mWidar Signal Frame', R.SNR));
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
              'SNR'     1 x K signal SNR [dB]; each tile is labelled with the
                        SNR of the frame it shows
        %}
        function fig = signal_montage(obj, signals, varargin)

            p = obj.common_parser();
            addParameter(p, 'Frames', []);
            addParameter(p, 'Count', 9);
            addParameter(p, 'Truth', []);
            addParameter(p, 'Est', []);
            addParameter(p, 'CLim', []);
            addParameter(p, 'SNR', []);
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
                %%% Second line rather than a separator: the tiles are small,
                %%% and a one-line title with the SNR on it gets clipped.
                tag = obj.snr_tag(obj.snr_at(R.SNR, k));
                if isempty(tag)
                    title(ax, sprintf('k = %d', k), 'FontSize', obj.fs);
                else
                    title(ax, {sprintf('k = %d', k), tag}, 'FontSize', obj.fs);
                end
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
              'SNR'    1 x 2 SNR [dB] for A and B, appended to their titles
                       (a scalar is taken to apply to both)
        %}
        function fig = compare_signals(obj, A, B, varargin)

            p = obj.common_parser();
            addParameter(p, 'Names', {'A', 'B'});
            addParameter(p, 'SNR', []);
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
            title(ax, obj.signal_title('', R.Names{1}, obj.snr_at(R.SNR, 1)), ...
                'FontSize', obj.fs);

            ax = nexttile(tl);
            obj.draw_image(ax, B, R.Units, R.Colormap, shared);
            obj.style_scene(ax, R.Units);
            title(ax, obj.signal_title('', R.Names{2}, obj.snr_at(R.SNR, 2)), ...
                'FontSize', obj.fs);
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
              'Particles'  S x N x K particle history (or cell / S x N x K x T)
              'Weights'    N x K weights. Shades and sizes the particles, and
                           adds a weight colorbar next to the scene.
              'WeightColormap'  colormap for that shading (default 'gray');
                           '' falls back to the flat particle color
              'WeightScale'     'log' (default) or 'linear'
              'WeightCLim'      fix the weight color limits, in the scaled
                           domain (so log10 units when 'WeightScale' is 'log')
              'pE'         T x K existence probability, shown in the title
              'SNR'        1 x K signal SNR [dB]; the title is relabelled with
                           the current frame's SNR on every frame
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
            addParameter(p, 'SNR', []);
            addParameter(p, 'Trail', 15);
            addParameter(p, 'FPS', obj.fps);
            addParameter(p, 'Format', '');
            addParameter(p, 'CLim', []);
            addParameter(p, 'Pause', 0);
            addParameter(p, 'WeightColormap', 'gray');
            addParameter(p, 'WeightScale', 'log');
            addParameter(p, 'WeightCLim', []);
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

            %%% Weight shading. The scene axes already owns a colormap (the
            %%% signal image) and an axes only gets one, so the particles are
            %%% given explicit RGB and the weight colorbar hangs off a hidden
            %%% axes of its own. Limits are computed once over the whole run:
            %%% per-frame limits would make every frame look identical and hide
            %%% exactly the collapse the shading is there to show.
            wopt = [];
            if ~isempty(R.Weights) && ~isempty(R.WeightColormap)
                wopt = obj.weight_opts(R.Weights, R.WeightColormap, ...
                                       R.WeightScale, R.WeightCLim);
                obj.weight_colorbar(fig, ax, wopt);
            end

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
                    if ~isempty(R.Weights) && k <= size(R.Weights, 2)
                        wk = R.Weights(:,k);
                    end
                    obj.overlay_particles(ax, Yk, wk, du, R.Units, wopt);
                end

                %%% Trails + current markers
                obj.draw_trails(ax, tru, k, R.Trail, du, R.Units, obj.truthColor, '-');
                obj.draw_trails(ax, est, k, R.Trail, du, R.Units, obj.estColor, '--');
                obj.mark_tracks_at_k(ax, tru, k, du, R.Units, obj.truthColor, 'o');
                obj.mark_tracks_at_k(ax, est, k, du, R.Units, obj.estColor, 'x');

                obj.style_scene(ax, R.Units);

                ttl = sprintf('k = %d / %d', k, K);
                %%% Per frame, not the whole run: SNR changes with the target's
                %%% position, so the number has to track the frame on screen.
                tag = obj.snr_tag(obj.snr_at(R.SNR, k));
                if ~isempty(tag)
                    ttl = [ttl, '   ', tag]; %#ok<AGROW>
                end
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
            per-axis position vs time, position error, particle health, and
            measurement SNR vs time when res.snr is filled in.

            fig = v.plot_TBD(res, ...)

            res is a struct; see visualize.results_template() for the fields.
            Anything absent is simply skipped, so partial results still plot.

            Options
              'pEthresh'   declaration threshold (default 0.5 or res.pEthresh)
              'ESSThresh'  ESS/N the filter resamples at, drawn on the ESS row
              'Units'/'DataUnits'/'Title'/'Save'
        %}
        function fig = plot_TBD(obj, res, varargin)

            p = obj.common_parser();
            addParameter(p, 'pEthresh', []);
            addParameter(p, 'ESSThresh', []);
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
            %%%   [       ESS        ]   <- only when the weights were recorded
            %%%   [       SNR        ]   <- only when res.snr is given
            %%% Wide on purpose: the scene panel is axis-equal, so it leaves
            %%% horizontal room in its tile that the legend drops into.
            %%% ESS and SNR span the full width: they share the x axis with the
            %%% rows above them and read as the run's timeline.
            hasSNR = obj.has_snr(res.snr);
            essR = obj.ess_ratio(res);
            hasESS = ~isempty(essR);
            nRows = 3 + hasESS + hasSNR;
            figH = 950 + 220 * hasESS + 220 * hasSNR;
            fig = obj.new_figure('TBD Results', [1400 figH]);
            tl = tiledlayout(fig, nRows, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

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
            %%% The background is a projection over every frame, so snr_tag
            %%% summarises the run rather than quoting one frame.
            snrBG = [];
            if ~isempty(bg)
                snrBG = res.snr;
            end
            title(ax, obj.signal_title('', 'Track (declared only)', snrBG), ...
                'FontSize', obj.fs);

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
            %%% Head count, not ESS: ESS gets its own full-width row below when
            %%% the weights were recorded, and the two answer different
            %%% questions (how many particles think the target exists, vs how
            %%% many of them are still carrying weight).
            ax = nexttile(tl);
            if ~isempty(res.particles)
                Yc = obj.to_particle_cell(res.particles);
                nAlive = squeeze(sum(Yc{1}(end,:,:) ~= 0, 2))';
                plot(ax, kvec(1:min(end,numel(nAlive))), ...
                    nAlive(1:min(end,numel(kvec))), '-', ...
                    'Color', obj.particleColor, 'LineWidth', obj.lw);
                ylabel(ax, '\# particles with E = 1', 'FontSize', obj.fs);
                title(ax, 'Existing Particles', 'FontSize', obj.fs);
            elseif hasESS
                obj.ess_axes(ax, kvec, essR, R.ESSThresh, res.resampled, tlab);
            else
                text(ax, 0.5, 0.5, 'no particle data', ...
                    'Units', 'normalized', 'HorizontalAlignment', 'center');
            end
            grid(ax, 'on');
            xlabel(ax, tlab, 'FontSize', obj.fs);

            %%% Effective sample size ---------------------------------------------
            if hasESS && ~isempty(res.particles)
                ax = nexttile(tl, [1 2]);
                obj.ess_axes(ax, kvec, essR, R.ESSThresh, res.resampled, tlab);
            end

            %%% Measurement SNR ---------------------------------------------------
            if hasSNR
                ax = nexttile(tl, [1 2]);
                obj.snr_axes(ax, kvec, res.snr, tlab);
            end

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
            Effective sample size vs time, with the resampling threshold and the
            frames the filter actually resampled on. This is the plot that says
            whether the ESS-triggered resampler is earning its keep: a curve
            that never approaches the threshold means the weights are healthy
            and the resamples were being wasted, a curve pinned to the floor
            means the likelihood is too peaked for N particles.

            fig = v.ess_vs_time(ess, ...)
              ess  1 x K effective sample size, absolute (1..N) or already a
                   ratio in [0, 1]

            Options
              'N'          particle count, used to turn an absolute ESS into a
                           ratio; required unless ess is already a ratio
              'Threshold'  ESS/N the filter resamples at, drawn as a line
              'Resampled'  1 x K logical, marks the frames that resampled
              'Time'       1 x K time vector, defaults to 1:K
              'Title'/'Save'/'Axes'
        %}
        function fig = ess_vs_time(obj, ess, varargin)

            p = obj.common_parser();
            addParameter(p, 'N', []);
            addParameter(p, 'Threshold', []);
            addParameter(p, 'Resampled', []);
            addParameter(p, 'Time', []);
            parse(p, varargin{:})
            R = p.Results;

            essR = reshape(double(ess), 1, []);
            if ~isempty(R.N)
                essR = essR ./ R.N;
            elseif max(essR, [], 'omitnan') > 1
                error('visualize:ess_vs_time', ...
                    'ess looks absolute (max %.3g > 1); pass ''N'' so it can be normalized', ...
                    max(essR, [], 'omitnan'));
            end

            kvec = R.Time;
            if isempty(kvec)
                kvec = 1:numel(essR);
            end

            [fig, ax] = obj.get_axes(R, 'Effective Sample Size');
            obj.ess_axes(ax, kvec, essR, R.Threshold, R.Resampled, 'k');
            if ~isempty(R.Title)
                title(ax, R.Title, 'FontSize', obj.fs);
            end
            obj.save_figure(fig, R.Save);

        end

        %{
            Measurement SNR vs time. The companion to existence(): when a track
            drops out, this says whether the signal was there to be found.

            fig = v.snr_vs_time(snr, ...)
              snr  1 x K peak SNR in dB, as simulator.get_SNR reports it
                   (scenario.snr carries it downstream)
            Options
              'Time'    1 x K time vector, defaults to 1:K
              'XLabel'  axis label for 'Time' (default 'k')
              'Axes'/'Title'/'Save'
        %}
        function fig = snr_vs_time(obj, snr, varargin)

            p = obj.common_parser();
            addParameter(p, 'Time', []);
            addParameter(p, 'XLabel', 'k');
            parse(p, varargin{:})
            R = p.Results;

            [fig, ax] = obj.get_axes(R, 'Measurement SNR');
            obj.snr_axes(ax, R.Time, snr, R.XLabel);
            obj.set_title(ax, R.Title, 'Measurement SNR');
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
              Yk  S x N particle set for a single frame
            Options
              'Weights'  1 x N weights
              'Signal'   npx x npx frame to draw underneath
              'Truth'    4 x 1 (or 4 x T) truth state for that frame
              'Est'      4 x 1 point estimate
              'ShowDead' also draw the E = 0 particles, in grey
              'SNR'      SNR [dB] of 'Signal', appended to the positions title
        %}
        function fig = particle_cloud(obj, Yk, varargin)

            p = obj.common_parser();
            addParameter(p, 'Weights', []);
            addParameter(p, 'Signal', []);
            addParameter(p, 'Truth', []);
            addParameter(p, 'Est', []);
            addParameter(p, 'ShowDead', false, @islogical);
            addParameter(p, 'SNR', []);
            parse(p, varargin{:})
            R = p.Results;

            du = obj.resolve_data_units(R);
            [ix, iy] = obj.pos_rows(Yk);
            N = size(Yk, 2);

            if size(Yk,1) >= 5
                alive = Yk(end,:) ~= 0;
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
            %%% Only label SNR when the signal is actually drawn underneath
            snr = [];
            if ~isempty(R.Signal)
                snr = R.SNR;
            end
            title(ax, obj.signal_title('', ...
                sprintf('Positions (%d / %d existing)', nnz(alive), N), snr), ...
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
              'SNR'      SNR [dB] of 'Signal', appended to its panel title
        %}
        function fig = particle_density(obj, Yk, varargin)

            p = obj.common_parser();
            addParameter(p, 'Weights', []);
            addParameter(p, 'Signal', []);
            addParameter(p, 'Bins', 2);
            addParameter(p, 'Truth', []);
            addParameter(p, 'SNR', []);
            parse(p, varargin{:})
            R = p.Results;

            du = obj.resolve_data_units(R);
            [ix, iy] = obj.pos_rows(Yk);
            N = size(Yk, 2);
            if size(Yk,1) >= 5
                alive = Yk(end,:) ~= 0;
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
                title(ax, obj.signal_title('', 'Measurement', R.SNR), ...
                    'FontSize', obj.fs);
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
              'Particles'  S x N x K history, adds unique-particle count
              'Frames'     frames to histogram (default 3 evenly spaced)
              'Time'
        %}
        function fig = pf_diagnostics(obj, W, varargin)

            p = obj.common_parser();
            addParameter(p, 'Particles', []);
            addParameter(p, 'Frames', []);
            addParameter(p, 'Time', []);
            addParameter(p, 'ESS', []);
            parse(p, varargin{:})
            R = p.Results;

            [N, K] = size(W);
            kvec = R.Time;
            if isempty(kvec)
                kvec = 1:K;
            end

            %%% W is the posterior weight, so on a frame that resampled it is
            %%% flat and its ESS reads N. Pass the filter's own pre-resample
            %%% series ('ESS', R.ess) to see what actually drove the decision.
            if isempty(R.ESS)
                essK = obj.ess(W) ./ N;
            else
                essK = reshape(double(R.ESS), 1, []) ./ N;
                essK = essK(1:min(numel(essK), K));
            end
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
                'particles', [], ... % S x N x K (or cell / S x N x K x T)
                'weights',   [], ... % N x K normalized weights
                'ess',       [], ... % 1 x K effective sample size (absolute, 1..N)
                'resampled', [], ... % 1 x K logical, frames the filter resampled on
                'snr',       [], ... % 1 x K measurement peak SNR [dB]
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

        %% ------------------------------------------------------------------
        %% SNR labelling
        %% ------------------------------------------------------------------

        %%% Text for the SNR of a frame, as simulator.get_SNR reports it.
        %%% A vector is summarised by the mean of its finite entries, which is
        %%% what the montage / MIP style plots want. Returns '' when there is
        %%% nothing meaningful to say, so a caller can append it blindly.
        function tag = snr_tag(~, snr)
            tag = '';
            if isempty(snr)
                return
            end
            v = double(snr(:))';

            if ~isscalar(v)
                g = v(isfinite(v));
                if isempty(g)
                    return
                end
                tag = sprintf('SNR = %.1f dB (mean)', mean(g));
                return
            end

            one = v(1);
            if isnan(one)
                return
            elseif isinf(one)
                if one > 0
                    tag = 'SNR = Inf (noiseless)';
                else
                    tag = 'SNR = -Inf (no target)';
                end
            else
                tag = sprintf('SNR = %.1f dB', one);
            end
        end

        %%% Is there an SNR worth drawing a panel for? A noiseless run records
        %%% Inf for every frame, which is true but not a plot.
        function tf = has_snr(~, snr)
            tf = ~isempty(snr) && any(isfinite(double(snr(:))));
        end

        %%% SNR for frame k out of a 1 x K vector. A scalar is taken to apply to
        %%% every frame; anything short of k reports NaN rather than erroring,
        %%% matching how the rest of the class clamps to the shorter series.
        function v = snr_at(~, snr, k)
            v = NaN;
            if isempty(snr)
                return
            end
            s = double(snr(:))';
            if isscalar(s)
                v = s;
            elseif k >= 1 && k <= numel(s)
                v = s(k);
            end
        end

        %%% Title for a panel that draws a signal: the caller's title, or the
        %%% fallback, with the SNR appended. Every signal plot labels SNR the
        %%% same way because they all come through here.
        %%%
        %%% An empty base stays empty, tag or no tag. Callers pair this with
        %%% set_title as
        %%%   set_title(ax, signal_title(R.Title, '', snr), ...
        %%%                 signal_title('', fallback, snr))
        %%% and that only picks the fallback if the first argument is empty, so
        %%% returning a bare tag here would lose the fallback text.
        function t = signal_title(obj, given, fallback, snr)
            t = char(string(given));
            if isempty(t)
                t = char(string(fallback));
            end
            if isempty(t)
                return
            end
            tag = obj.snr_tag(snr);
            if ~isempty(tag)
                t = [t '   |   ' tag];
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

        %%% Normalize particle histories to a 1 x T cell of S x N x K
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
        %%% Draw one frame's particle cloud. wopt (optional) is the struct
        %%% weight_opts() builds; with it the particles are shaded by weight
        %%% using explicit RGB, without it they are only sized by weight.
        function overlay_particles(obj, ax, Yk, w, du, units, wopt)
            if nargin < 7
                wopt = [];
            end
            if isempty(Yk)
                return
            end
            [ix, iy] = obj.pos_rows(Yk);
            N = size(Yk,2);
            if size(Yk,1) >= 5
                alive = Yk(end,:) ~= 0;
            else
                alive = true(1, N);
            end
            if ~any(alive)
                return
            end
            [x, y] = obj.convert_xy(Yk(ix,alive), Yk(iy,alive), du, units);
            if isempty(w)
                plot(ax, x, y, '.', 'Color', obj.particleColor, ...
                    'MarkerSize', 8, 'DisplayName', 'Particles');
                return
            end

            wa = reshape(w, 1, []);
            wa = wa(alive);

            if isempty(wopt)
                sz = 10 + 35 * obj.unit_scale(wa);
                scatter(ax, x, y, sz, obj.particleColor, 'filled', ...
                    'MarkerFaceAlpha', 0.6, 'DisplayName', 'Particles');
                return
            end

            %%% One colormap per axes, and the signal image already claimed it,
            %%% so hand scatter an n x 3 of RGB instead of scalar CData.
            u = obj.weight_unit(wa, wopt);
            C = obj.weight_rgb(u, wopt);
            sz = 6 + 40 * u;
            scatter(ax, x, y, sz, C, 'filled', 'MarkerFaceAlpha', 0.85, ...
                'DisplayName', 'Particles');
        end

        %%% Settle the weight color scale once for a whole run.
        %%%   W      N x K (or 1 x N) weights
        %%%   cmap   colormap name, or an m x 3 table
        %%%   scale  'log' | 'linear'
        %%%   lim    caller-supplied limits in the scaled domain, or []
        function wopt = weight_opts(obj, W, cmap, scale, lim)
            scale = lower(char(string(scale)));
            assert(any(strcmp(scale, {'log', 'linear'})), ...
                'WeightScale must be ''log'' or ''linear''');

            wopt = struct('cmap', cmap, 'scale', scale, 'clim', [], ...
                          'range', [0.18 1.0]);

            if ~isempty(lim)
                wopt.clim = reshape(double(lim), 1, 2);
                return
            end

            v = double(W(:));
            v = v(isfinite(v) & v > 0);
            if isempty(v)
                wopt.clim = [0 1];
                return
            end
            if strcmp(scale, 'log')
                v = log10(v);
            end

            hi = max(v);
            %%% 2nd percentile, not the min: a handful of particles sitting on
            %%% a numerically-zero likelihood would otherwise stretch the scale
            %%% over a hundred decades and flatten everything else to one shade.
            lo = obj.row_pctile(reshape(v, 1, []), 0.02);
            if strcmp(scale, 'log')
                lo = max(lo, hi - 6); % six decades is plenty of dynamic range
            end
            if ~(hi > lo)
                lo = hi - 1;
            end
            wopt.clim = [lo hi];
        end

        %%% Weights -> [0, 1] against the run-wide limits in wopt
        function u = weight_unit(~, w, wopt)
            v = reshape(double(w), 1, []);
            if strcmp(wopt.scale, 'log')
                v(v <= 0) = NaN;
                v = log10(v);
            end
            u = (v - wopt.clim(1)) / max(wopt.clim(2) - wopt.clim(1), eps);
            u(~isfinite(u)) = 0; % zero / NaN weight reads as "lightest"
            u = min(max(u, 0), 1);
        end

        %%% n x 3 RGB for unit-scaled weights. wopt.range trims the dark end of
        %%% the colormap off: pure black particles vanish into the low end of
        %%% turbo, which is exactly where an un-weighted particle tends to sit.
        function C = weight_rgb(obj, u, wopt)
            M = obj.colormap_table(wopt.cmap);
            m = size(M, 1);
            r = wopt.range;
            lo = max(1, round(r(1) * m));
            hi = max(lo, round(r(2) * m));
            M = M(lo:hi, :);
            idx = 1 + round(reshape(u, [], 1) * (size(M,1) - 1));
            idx = min(max(idx, 1), size(M,1));
            C = M(idx, :);
        end

        function M = colormap_table(~, cmap)
            if isnumeric(cmap)
                M = cmap;
            else
                M = feval(char(string(cmap)), 256);
            end
        end

        %%% Colorbar for the particle weights. The scene axes' colormap belongs
        %%% to the signal image, so the bar is driven by an invisible axes
        %%% parked on top of it and positioned by hand.
        function cb = weight_colorbar(obj, fig, ax, wopt)
            pos = get(ax, 'Position');
            set(ax, 'Position', [pos(1), pos(2), pos(3) * 0.86, pos(4)]);

            axc = axes(fig, 'Position', get(ax, 'Position'), ...
                'Visible', 'off', 'Color', 'none', 'HandleVisibility', 'off'); %#ok<LAXES>
            colormap(axc, obj.weight_rgb(linspace(0,1,256), wopt));
            caxis(axc, wopt.clim); %#ok<CAXIS>

            cb = colorbar(axc);
            cb.Position = [pos(1) + pos(3) * 0.90, pos(2) + 0.10 * pos(4), ...
                           0.022, 0.78 * pos(4)];
            if strcmp(wopt.scale, 'log')
                cb.Label.String = 'log_{10} particle weight';
            else
                cb.Label.String = 'particle weight';
            end
            cb.Label.FontSize = obj.fs;
            cb.FontSize = obj.fs - 1;
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
                    alive = Y(end,:,k) ~= 0;
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

        %%% Shared SNR vs time panel, used by plot_TBD and snr_vs_time.
        %%% Frames whose SNR is not finite (noise off, or no target in the
        %%% scene) are drawn as gaps: plotting them would drag the y limits out
        %%% to +-Inf and hide the range that matters.
        %%% ESS/N for a results struct, or [] when the run recorded neither an
        %%% ESS series nor the weights it would be computed from.
        function essR = ess_ratio(obj, res)
            essR = [];

            N = 0;
            if ~isempty(res.particles)
                Yc = obj.to_particle_cell(res.particles);
                N = size(Yc{1}, 2);
            elseif ~isempty(res.weights)
                N = size(res.weights, 1);
            end

            if ~isempty(res.ess)
                essR = reshape(double(res.ess), 1, []);
                if N > 0
                    essR = essR / N;
                end
            elseif ~isempty(res.weights)
                %%% Fall back to the stored weights. Note this is the ESS of
                %%% the posterior: on a frame that resampled the weights are
                %%% flat again and it reads 1, which is why run() records the
                %%% pre-resample value separately.
                essR = obj.ess(res.weights) ./ size(res.weights, 1);
            end
        end

        %%% Shared ESS panel, used by plot_TBD and ess_vs_time.
        %%%   essR       1 x K ESS/N
        %%%   thresh     ESS/N the filter resamples at, or [] to omit the line
        %%%   resampled  1 x K logical, or [] to omit the markers
        function ess_axes(obj, ax, kvec, essR, thresh, resampled, tlab)

            essR = reshape(double(essR), 1, []);
            if isempty(kvec)
                kvec = 1:numel(essR);
            end
            n = min(numel(kvec), numel(essR));
            kvec = kvec(1:n);
            essR = essR(1:n);

            plot(ax, kvec, essR, '-', 'Color', obj.particleColor, ...
                'LineWidth', obj.lw, 'DisplayName', 'ESS / N');
            hold(ax, 'on');

            if ~isempty(thresh)
                yline(ax, thresh, 'k--', 'LineWidth', 1, ...
                    'Label', sprintf('resample at %.2f', thresh), ...
                    'LabelHorizontalAlignment', 'left', ...
                    'LabelVerticalAlignment', 'bottom', ...
                    'HandleVisibility', 'off');
            end

            %%% Marked on the curve rather than as vertical lines: on a run that
            %%% resamples most frames a rug of xlines is solid black.
            nres = 0;
            if ~isempty(resampled)
                m = logical(reshape(resampled, 1, []));
                m = m(1:min(numel(m), n));
                nres = nnz(m);
                if nres > 0
                    plot(ax, kvec(m), essR(m), 'o', 'Color', obj.estColor, ...
                        'MarkerSize', obj.ms - 1, 'LineStyle', 'none', ...
                        'DisplayName', 'resampled');
                end
            end

            grid(ax, 'on');
            ylim(ax, [0 1]);
            xlabel(ax, tlab, 'FontSize', obj.fs);
            ylabel(ax, 'ESS / N', 'FontSize', obj.fs);

            good = isfinite(essR);
            ttl = 'Effective Sample Size';
            if any(good)
                ttl = sprintf('%s  (median %.3f', ttl, median(essR(good)));
                if ~isempty(resampled)
                    ttl = sprintf('%s, resampled %d/%d frames', ttl, nres, n);
                end
                ttl = [ttl ')'];
            end
            title(ax, ttl, 'FontSize', obj.fs);
            obj.maybe_legend(ax);
        end

        function snr_axes(obj, ax, kvec, snr, tlab)

            v = obj.to_rowmat(snr);
            if isempty(v)
                text(ax, 0.5, 0.5, 'no SNR data', 'Units', 'normalized', ...
                    'HorizontalAlignment', 'center');
                axis(ax, 'off');
                return
            end
            v = v(1,:);

            if isempty(kvec)
                kvec = 1:numel(v);
            end
            n = min(numel(kvec), numel(v));
            kvec = kvec(1:n);
            v = v(1:n);

            bad = ~isfinite(v);
            if any(bad)
                obj.debug_print(sprintf("SNR: %d / %d frames not finite, drawn as gaps", ...
                    nnz(bad), n));
                v(bad) = NaN;
            end

            plot(ax, kvec, v, '-', 'Color', obj.estColor, 'LineWidth', obj.lw, ...
                'DisplayName', 'SNR_k');
            hold(ax, 'on');

            good = ~isnan(v);
            if any(good)
                yline(ax, mean(v(good)), 'k:', 'LineWidth', 1, ...
                    'Label', sprintf('mean %.1f dB', mean(v(good))));
            else
                text(ax, 0.5, 0.5, 'no finite SNR', 'Units', 'normalized', ...
                    'HorizontalAlignment', 'center');
            end

            grid(ax, 'on');
            xlabel(ax, tlab, 'FontSize', obj.fs);
            ylabel(ax, 'peak SNR [dB]', 'FontSize', obj.fs);
            title(ax, 'Measurement SNR', 'FontSize', obj.fs);
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
            if ~isempty(res.ess)
                K = max(K, numel(res.ess));
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
