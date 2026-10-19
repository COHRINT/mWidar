%%% Expand a test definition into a struct array of fully resolved trials.
%%%
%%%   trials = scenarios x seeds x sweep points x variants
%%%
%%% Every missing axis contributes a single no-op entry. Loop order puts the
%%% variant innermost, so consecutive trials share a generated scenario and
%%% run_test only has to keep the latest one (sweeping an env.* parameter
%%% just regenerates).
%%%
%%% Override precedence, later wins:
%%%   default_cfg -> t.base -> scenario -> variant -> sweep -> MC seed
%%% then resolve_cfg fills the derived fields.
%%%
%%% Each trial:
%%%   .cfg       resolved cfg, ready for the classes
%%%   .scenario  scenario name     .variant  variant name
%%%   .sweep     "Pb=0.1,N=500"    .seed     env seed
%%%   .group     scenario|variant|sweep, the key MC seeds are averaged over
%%%   .label     group + seed, unique per trial
%%%   .file      label made file-name safe

function trials = expand_trials(t)

    scen = pick_all(t.scenarios, scenarios(), "scenario");
    vars = pick_all(t.variants,  variants(),  "variant");
    pts  = sweep_points(t.sweep, t.sweep_mode);

    if isempty(t.mc)
        seeds = NaN;                                % keep whatever the cfg says
    else
        mc = apply_overrides(struct('n', 1, 'seed0', 1), t.mc, "test.mc");
        seeds = mc.seed0 : mc.seed0 + mc.n - 1;
    end

    trials = struct('cfg', {}, 'scenario', {}, 'variant', {}, 'sweep', {}, ...
                    'seed', {}, 'group', {}, 'label', {}, 'file', {});

    base = apply_overrides(default_cfg(), t.base, "test.base");

    for i = 1:numel(scen)
        for m = 1:numel(seeds)
            for k = 1:numel(pts)
                for j = 1:numel(vars)
                    cfg = apply_overrides(base, scen(i).ov, "scenario(" + scen(i).name + ")");
                    cfg = apply_overrides(cfg,  vars(j).ov, "variant(" + vars(j).name + ")");
                    for p = 1:numel(pts(k).path)
                        cfg = set_path(cfg, pts(k).path{p}, pts(k).value{p});
                    end
                    if ~isnan(seeds(m)), cfg.env.Seed = seeds(m); end
                    cfg = resolve_cfg(cfg);

                    tr.cfg      = cfg;
                    tr.scenario = scen(i).name;
                    tr.variant  = vars(j).name;
                    tr.sweep    = pts(k).label;
                    tr.seed     = cfg.env.Seed;
                    tr.group    = join_nonempty([tr.scenario, tr.variant, tr.sweep]);
                    tr.label    = join_nonempty([tr.group, "s" + tr.seed]);
                    if tr.group == "", tr.group = "default"; end
                    tr.file     = regexprep(tr.label, '[^\w=.\-]+', '_');
                    trials(end+1) = tr; %#ok<AGROW>
                end
            end
        end
    end

end

%% Resolve an axis: names from the library, or inline {name, overrides} pairs
function out = pick_all(items, lib, what)

    out = struct('name', {}, 'ov', {});
    if isempty(items)
        out(1).name = "";
        out(1).ov   = struct();
        return
    end
    if isstring(items) || ischar(items), items = cellstr(items); end

    i = 1;
    while i <= numel(items)
        it = items{i};
        if (isstring(it) || ischar(it)) && i < numel(items) && isstruct(items{i+1})
            out(end+1) = struct('name', string(it), 'ov', items{i+1}); %#ok<AGROW>
            i = i + 2;
        elseif isstring(it) || ischar(it)
            assert(isfield(lib, it), "Unknown %s '%s'. Library has: %s", ...
                what, it, strjoin(string(fieldnames(lib))', ', '));
            out(end+1) = struct('name', string(it), 'ov', lib.(it)); %#ok<AGROW>
            i = i + 1;
        else
            error("expand_trials:badAxis", "%s entries must be names, or a name followed by an overrides struct", what);
        end
    end

end

%% Sweep -> list of points, each a set of (path, value) pairs plus a label
function pts = sweep_points(sw, mode)

    if isempty(sw)
        pts = struct('path', {{}}, 'value', {{}}, 'label', "");
        return
    end

    vals = cell(1, numel(sw));
    for p = 1:numel(sw)
        v = sw(p).values;
        if ~iscell(v), v = num2cell(v); end
        vals{p} = v;
    end
    n = cellfun(@numel, vals);

    if mode == "zip"
        assert(all(n == n(1)), "sweep_mode ""zip"" needs every values list the same length, got %s", mat2str(n));
        idx = repmat((1:n(1))', 1, numel(sw));
    else
        ranges = arrayfun(@(x) 1:x, n, 'UniformOutput', false);
        grids = cell(1, numel(sw));
        [grids{:}] = ndgrid(ranges{:});
        idx = cell2mat(cellfun(@(g) g(:), grids, 'UniformOutput', false));
    end

    pts = struct('path', {}, 'value', {}, 'label', {});
    for r = 1:size(idx, 1)
        pt.path  = {sw.path};
        pt.value = arrayfun(@(p) vals{p}{idx(r, p)}, 1:numel(sw), 'UniformOutput', false);
        lbl = strings(1, numel(sw));
        for p = 1:numel(sw)
            leaf = regexp(sw(p).path, '[^.]+$', 'match', 'once');
            lbl(p) = leaf + "=" + val2str(pt.value{p});
        end
        pt.label = strjoin(lbl, ",");
        pts(end+1) = pt; %#ok<AGROW>
    end

end

function s = val2str(v)
    if ischar(v) || isstring(v)
        s = string(v);
    elseif iscell(v)
        s = strjoin(cellfun(@val2str, v), "+");
    elseif isscalar(v)
        s = string(num2str(v, 4));
    else
        s = string(mat2str(v, 4));
    end
end

function s = join_nonempty(parts)
    s = strjoin(parts(parts ~= ""), "|");
end
