%%% Comparison plots for run_compare. Every plot overlays all filters on the
%%% same axes, one fixed color + marker per filter.
%%%
%%% Each figure is for ONE TBD-PF particle count: both TBD-PF variants at
%%% that N, plus the RBPF at its own N. With several N_TBD values you get one
%%% set of figures per value (files suffixed _N<n>). RBPF is the only filter
%%% that may appear at more than one N in a figure (lighter shade = smaller N).
%%%
%%%   figs = plot_compare(T, TS, 'OSPACutoff', 0.5, 'SaveDir', dir, 'ShowRunFigs', true)
%%%
%%% Inputs
%%%   T, TS    from compare_metrics
%%% Options
%%%   'OSPACutoff'   c used for OSPA, only for labels/limits (default 0.5)
%%%   'SaveDir'      folder for PNGs, '' = don't save (default '')
%%%   'ShowRunFigs'  keep the per-run figures open (default: true when there
%%%                  are <= 8 of them in total). Hidden ones are saved then closed.
%%%
%%% Figures (figs(i) is the set for the i-th TBD-PF N)
%%%   summary   RMSE, mean OSPA, runtime per trajectory (bars = mean over
%%%             trials, error bars = std, dots = individual trials)
%%%   run       per (trajectory, trial): position error and OSPA vs frame
%%%   mc        per trajectory: mean +- std over trials of error and OSPA vs frame
%%% plus figs(1).vsN: RMSE, OSPA, runtime vs particle count over ALL runs,
%%% only when some filter was run at more than one N.
%%% Gray background = frames with no target (per-run), or the fraction of
%%% trials with no target (MC mean).

function figs = plot_compare(T, TS, varargin)

    p = inputParser;
    addParameter(p, 'OSPACutoff', 0.5);
    addParameter(p, 'SaveDir', '');
    addParameter(p, 'ShowRunFigs', []);
    parse(p, varargin{:});
    c = p.Results.OSPACutoff;
    saveDir = char(p.Results.SaveDir);
    if ~isempty(saveDir) && ~exist(saveDir, 'dir'), mkdir(saveDir); end

    isRBPF = T.filter == "KF_RBPF";
    tbdNs = unique(T.N(~isRBPF))';
    if isempty(tbdNs), tbdNs = NaN; end        % RBPF only: a single set

    runs = unique(T(:, {'traj', 'trial', 'seed'}), 'stable');
    show = p.Results.ShowRunFigs;
    if isempty(show), show = height(runs) * numel(tbdNs) <= 8; end

    figs = struct('N', {}, 'summary', {}, 'run', {}, 'mc', {}, 'vsN', {});
    for n = tbdNs
        sel = isRBPF | T.N == n;
        Tn = T(sel, :);
        TSn = TS(sel);
        tag = ''; sfx = '';
        if ~isnan(n), tag = sprintf('TBD-PF N = %d', n); sfx = sprintf('_N%d', n); end

        S = series_table(Tn);
        trajs = unique(Tn.traj, 'stable');
        nTrials = numel(unique(Tn.trial));

        f = struct('N', n, 'summary', [], 'run', [], 'mc', [], 'vsN', []);
        f.summary = plot_summary(Tn, S, trajs, nTrials, c, tag);
        save_fig(f.summary, saveDir, ['summary' sfx]);

        f.run = gobjects(1, height(runs));
        for i = 1:height(runs)
            selR = Tn.traj == runs.traj(i) & Tn.trial == runs.trial(i);
            f.run(i) = plot_run(Tn, TSn, S, selR, runs(i, :), c, show, tag);
            save_fig(f.run(i), saveDir, sprintf('run_%s_trial%02d%s', runs.traj(i), runs.trial(i), sfx));
            if ~show, close(f.run(i)); end
        end

        f.mc = gobjects(1, numel(trajs));
        for i = 1:numel(trajs)
            f.mc(i) = plot_mc(Tn, TSn, S, trajs(i), nTrials, c, tag);
            save_fig(f.mc(i), saveDir, sprintf('mc_%s%s', trajs(i), sfx));
        end
        figs(end+1) = f; %#ok<AGROW>
    end

    % Particle-count study across all runs
    Sall = series_table(T);
    if any(groupcounts(Sall.filter) > 1)
        figs(1).vsN = plot_vsN(T, Sall, c);
        save_fig(figs(1).vsN, saveDir, 'vsN');
    end

end

%%% "<title> (TBD-PF N = 5000)", or just the title when tag is empty
function s = with_tag(s, tag)
    if ~isempty(tag), s = sprintf('%s  (%s)', s, tag); end
end


%% ----------------------------------------------------------------------
%% Figures
%% ----------------------------------------------------------------------

function fig = plot_summary(T, S, trajs, nTrials, c, tag)

    metrics = {'rmse',    'Position RMSE [m]';
               'ospa',    sprintf('Mean OSPA [m] (c = %.2g m)', c);
               'runtime', 'Runtime [s]'};

    fig = figure('Name', 'run_compare summary', 'Color', 'w', 'Position', [80 80 1350 450]);
    tl = tiledlayout(fig, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(tl, with_tag(sprintf('Filter comparison, %d trial(s) per trajectory', nTrials), tag));

    nT = numel(trajs);
    nS = height(S);
    for mi = 1:size(metrics, 1)
        ax = nexttile(tl);
        hold(ax, 'on');
        mu = nan(nT, nS);
        sd = nan(nT, nS);
        for t = 1:nT
            for s = 1:nS
                y = T.(metrics{mi, 1})(sel_series(T, S, s) & T.traj == trajs(t));
                mu(t, s) = mean(y, 'omitnan');
                sd(t, s) = std(y, 'omitnan');
            end
        end

        % bar() reads a single row as one group per column only with >= 2 x
        % values, so pad a lone trajectory with an empty group
        xb = 1:nT; mub = mu;
        if nT == 1, xb = [1 2]; mub = [mu; nan(1, nS)]; end
        b = bar(ax, xb, mub, 'grouped', 'BarWidth', 0.8, 'EdgeColor', 'w', 'LineWidth', 1);
        if nT == 1, xlim(ax, [0.5 1.5]); end
        for s = 1:nS
            b(s).FaceColor = S.color(s, :);
            b(s).DisplayName = S.label(s);
            xs = b(s).XEndPoints(1:nT);
            if nTrials > 1
                errorbar(ax, xs, mu(:, s)', sd(:, s)', 'k', 'LineStyle', 'none', ...
                    'LineWidth', 1, 'CapSize', 4, 'HandleVisibility', 'off');
                for t = 1:nT
                    y = T.(metrics{mi, 1})(sel_series(T, S, s) & T.traj == trajs(t));
                    jit = (rand(numel(y), 1) - 0.5) * 0.6 * b(s).BarWidth / nS;
                    scatter(ax, xs(t) + jit, y, 14, [0.25 0.25 0.25], 'filled', ...
                        'MarkerFaceAlpha', 0.6, 'HandleVisibility', 'off');
                end
            end
        end
        hold(ax, 'off');
        style_axes(ax);
        ax.XTick = 1:nT;
        ax.XTickLabel = cellstr(trajs);
        ax.XGrid = 'off';
        title(ax, metrics{mi, 2});
        if strcmp(metrics{mi, 1}, 'ospa'), ylim(ax, [0 c]); else, ylim(ax, [0 max(ylim(ax))]); end
        if mi == 1
            top_legend(ax);
        end
    end

end


function fig = plot_run(T, TS, S, sel, run, c, show, tag)

    vis = 'on'; if ~show, vis = 'off'; end
    fig = figure('Name', sprintf('%s trial %d', run.traj, run.trial), 'Color', 'w', ...
                 'Position', [100 100 1000 620], 'Visible', vis);
    tl = tiledlayout(fig, 2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(tl, with_tag(sprintf('%s, trial %d (seed %d)', run.traj, run.trial, run.seed), tag), 'Interpreter', 'none');

    idx = find(sel)';
    present = TS(idx(1)).present;
    K = numel(present);
    k = 1:K;

    fields = {'err', 'Position error [m]'; 'ospa', sprintf('OSPA [m] (c = %.2g m)', c)};
    for f = 1:2
        ax = nexttile(tl);
        hold(ax, 'on');
        for j = idx
            s = series_of(T(j, :), S);
            y = TS(j).(fields{f, 1});
            plot(ax, k, y, '-', 'Color', S.color(s, :), 'LineWidth', 2, ...
                'Marker', S.marker(s), 'MarkerIndices', marker_idx(y, s), 'MarkerSize', 7, ...
                'MarkerFaceColor', S.color(s, :), 'MarkerEdgeColor', 'w', ...
                'DisplayName', S.label(s));
        end
        hold(ax, 'off');
        style_axes(ax);
        xlim(ax, [0.5 K + 0.5]);
        if f == 2, ylim(ax, [0 1.05 * c]); else, ylim(ax, [0 max(ylim(ax))]); end
        shade_absent(ax, double(~present));
        ylabel(ax, fields{f, 2});
        if f == 1
            top_legend(ax);
        else
            xlabel(ax, 'Frame k');
        end
    end

end


function fig = plot_mc(T, TS, S, traj, nTrials, c, tag)

    fig = figure('Name', sprintf('%s MC mean', traj), 'Color', 'w', 'Position', [120 120 1000 620]);
    tl = tiledlayout(fig, 2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(tl, with_tag(sprintf('%s: mean \\pm std over %d trial(s)', traj, nTrials), tag), 'Interpreter', 'tex');

    selT = find(T.traj == traj)';
    K = numel(TS(selT(1)).present);
    k = 1:K;

    % fraction of trials with no target at each frame (one scenario per trial)
    [~, firstOfTrial] = unique(T.trial(selT), 'stable');
    absent = zeros(1, K);
    for j = selT(firstOfTrial)
        absent = absent + ~TS(j).present;
    end
    absent = absent / numel(firstOfTrial);

    fields = {'err', 'Position error [m]'; 'ospa', sprintf('OSPA [m] (c = %.2g m)', c)};
    for f = 1:2
        ax = nexttile(tl);
        hold(ax, 'on');
        for s = 1:height(S)
            js = selT(sel_series(T(selT, :), S, s));
            if isempty(js), continue; end
            Y = vertcat(TS(js).(fields{f, 1}));
            mu = mean(Y, 1, 'omitnan');
            sd = std(Y, 0, 1, 'omitnan');
            if numel(js) > 1
                band(ax, k, mu - sd, mu + sd, S.color(s, :));
            end
            plot(ax, k, mu, '-', 'Color', S.color(s, :), 'LineWidth', 2, ...
                'Marker', S.marker(s), 'MarkerIndices', marker_idx(mu, s), 'MarkerSize', 7, ...
                'MarkerFaceColor', S.color(s, :), 'MarkerEdgeColor', 'w', ...
                'DisplayName', S.label(s));
        end
        hold(ax, 'off');
        style_axes(ax);
        xlim(ax, [0.5 K + 0.5]);
        if f == 2, ylim(ax, [0 1.05 * c]); else, ylim(ax, [0 max(ylim(ax))]); end
        shade_absent(ax, absent);
        ylabel(ax, fields{f, 2});
        if f == 1
            top_legend(ax);
        else
            xlabel(ax, 'Frame k');
        end
    end

end


function fig = plot_vsN(T, S, c)

    metrics = {'rmse',    'Position RMSE [m]';
               'ospa',    sprintf('Mean OSPA [m] (c = %.2g m)', c);
               'runtime', 'Runtime [s]'};
    filters = unique(S.filter, 'stable');

    fig = figure('Name', 'run_compare vs N', 'Color', 'w', 'Position', [140 140 1350 450]);
    tl = tiledlayout(fig, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(tl, 'Metrics vs particle count (mean \pm std over all runs)');

    for mi = 1:size(metrics, 1)
        ax = nexttile(tl);
        hold(ax, 'on');
        for fi = 1:numel(filters)
            rowsS = find(S.filter == filters(fi))';
            Ns = S.N(rowsS);
            mu = nan(size(Ns));
            sd = nan(size(Ns));
            for i = 1:numel(Ns)
                y = T.(metrics{mi, 1})(T.filter == filters(fi) & T.N == Ns(i));
                mu(i) = mean(y, 'omitnan');
                sd(i) = std(y, 'omitnan');
            end
            base = S.base(rowsS(end), :);
            errorbar(ax, Ns, mu, sd, '-', 'Color', base, 'LineWidth', 2, 'CapSize', 4, ...
                'Marker', S.marker(rowsS(1)), 'MarkerSize', 8, 'MarkerFaceColor', base, ...
                'MarkerEdgeColor', 'w', 'DisplayName', S.name(rowsS(1)));
        end
        hold(ax, 'off');
        style_axes(ax);
        ax.XScale = 'log';
        xlabel(ax, 'N particles');
        title(ax, metrics{mi, 2});
        if strcmp(metrics{mi, 1}, 'ospa'), ylim(ax, [0 c]); else, ylim(ax, [0 max(ylim(ax))]); end
        if mi == 1
            top_legend(ax);
        end
    end

end


%% ----------------------------------------------------------------------
%% Series: one per (filter, N), fixed color per filter
%% ----------------------------------------------------------------------

function S = series_table(T)

    % Fixed filter -> color/marker/name, so a filter keeps its color when
    % others are dropped. Palette validated for CVD separation.
    style = struct( ...
        'KF_RBPF',      struct('color', hex('#2a78d6'), 'marker', 'o', 'name', "KF-RBPF"), ...
        'TBD_PF_gauss', struct('color', hex('#eb6834'), 'marker', 's', 'name', "TBD-PF gauss"), ...
        'TBD_PF_model', struct('color', hex('#1baf7a'), 'marker', '^', 'name', "TBD-PF model"));
    order = ["KF_RBPF", "TBD_PF_gauss", "TBD_PF_model"];

    S = unique(T(:, {'filter', 'N'}));
    [~, o] = sortrows([arrayfun(@(f) find(order == f, 1), S.filter), S.N]);
    S = S(o, :);

    n = height(S);
    S.name   = strings(n, 1);
    S.label  = strings(n, 1);
    S.marker = strings(n, 1);
    S.base   = zeros(n, 3);
    S.color  = zeros(n, 3);
    for i = 1:n
        st = style.(S.filter(i));
        sameF = find(S.filter == S.filter(i));
        nF = numel(sameF);
        r = find(sameF == i);                   % rank by N within the filter (ascending)
        fade = 0;
        if nF > 1, fade = 0.55 * (nF - r) / (nF - 1); end   % largest N = full color
        S.name(i)   = st.name;
        S.marker(i) = st.marker;
        S.base(i, :)  = st.color;
        S.color(i, :) = st.color + fade * (1 - st.color);
        S.label(i)  = sprintf('%s (N=%d)', st.name, S.N(i));
    end

end

function tf = sel_series(T, S, s)
    tf = T.filter == S.filter(s) & T.N == S.N(s);
end

function s = series_of(row, S)
    s = find(S.filter == row.filter & S.N == row.N, 1);
end


%% ----------------------------------------------------------------------
%% Helpers
%% ----------------------------------------------------------------------

%%% One legend for the whole figure, above the tiles, so it is never
%%% clipped by a narrow tile
function top_legend(ax)
    lgd = legend(ax, 'Orientation', 'horizontal', 'Box', 'off');
    lgd.Layout.Tile = 'north';
end

function style_axes(ax)
    ax.Color = 'w';
    ax.Box = 'off';
    ax.TickDir = 'out';
    ax.XColor = [0.35 0.35 0.35];
    ax.YColor = [0.35 0.35 0.35];
    ax.GridColor = [0.85 0.85 0.85];
    ax.GridAlpha = 1;
    ax.YGrid = 'on';
    ax.FontSize = 10;
    ax.TitleFontWeight = 'normal';
end

%%% Markers every ~10 frames on finite points, offset per series so
%%% overlapping lines still show every marker
function mi = marker_idx(y, s)
    mi = (1 + 3 * (s - 1)):10:numel(y);
    mi = mi(isfinite(y(mi)));
end

%%% Gray background where the target is absent, alpha scaled by w(k) in [0, 1]
function shade_absent(ax, w)
    yl = ylim(ax);
    hold(ax, 'on');
    for k = find(w > 0)
        h = patch(ax, [k - 0.5, k + 0.5, k + 0.5, k - 0.5], [yl(1) yl(1) yl(2) yl(2)], ...
            [0.5 0.5 0.5], 'EdgeColor', 'none', 'FaceAlpha', 0.18 * w(k), ...
            'HandleVisibility', 'off');
        uistack(h, 'bottom');
    end
    hold(ax, 'off');
    ylim(ax, yl);
end

%%% Shaded band between lo and hi, split at non-finite gaps
function band(ax, k, lo, hi, color)
    ok = isfinite(lo) & isfinite(hi);
    d = diff([0 ok 0]);
    starts = find(d == 1);
    stops  = find(d == -1) - 1;
    for i = 1:numel(starts)
        r = starts(i):stops(i);
        fill(ax, [k(r), fliplr(k(r))], [max(lo(r), 0), fliplr(hi(r))], color, ...
            'EdgeColor', 'none', 'FaceAlpha', 0.15, 'HandleVisibility', 'off');
    end
end

function save_fig(fig, saveDir, name)
    if isempty(saveDir) || isempty(fig) || ~isgraphics(fig), return; end
    exportgraphics(fig, fullfile(saveDir, [name '.png']), 'Resolution', 150);
end

function rgb = hex(h)
    h = char(h);
    rgb = double([hex2dec(h(2:3)), hex2dec(h(4:5)), hex2dec(h(6:7))]) / 255;
end
