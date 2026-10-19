%%% Find a test definition in config/ by name or T# prefix, call it, and
%%% merge it strictly onto default_test.
%%%
%%%   t = load_test('T5')                         % unique match of config/T5_*.m
%%%   t = load_test('T5_PF_All_N5000_v0_s1')      % exact file name
%%%   t = load_test(5)

function t = load_test(name)

    cfgDir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'config');
    fn = find_config(cfgDir, name);

    raw = feval(fn);
    t = default_test();

    %% Top level: strict, but axes/base are taken as given (they are
    %% validated when expanded onto default_cfg).
    for f = string(fieldnames(raw))'
        if ~isfield(t, f)
            error("load_test:unknownField", "%s: unknown test field '%s'. Valid fields: %s", ...
                fn, f, strjoin(string(fieldnames(t))', ', '));
        end
        if f == "output"
            t.output = apply_overrides(t.output, raw.output, "test.output");
        else
            t.(f) = raw.(f);
        end
    end
    if t.name == "", t.name = string(fn); end
    assert(ismember(t.sweep_mode, ["grid", "zip"]), "%s: sweep_mode must be ""grid"" or ""zip""", fn);

end

function fn = find_config(cfgDir, name)

    if isnumeric(name), name = "T" + name; end
    name = string(name);
    if ~startsWith(name, ["T", "t"]) && ~isnan(str2double(name)), name = "T" + name; end
    name = regexprep(name, '^t', 'T');

    files = dir(fullfile(cfgDir, 'T*.m'));
    names = erase(string({files.name}), ".m");
    if any(names == name)
        fn = char(name);
        return
    end
    hit = names(startsWith(names, name + "_"));
    if isempty(hit)
        error("load_test:notFound", "No test '%s' in %s. Available:\n  %s", name, cfgDir, strjoin(names, "\n  "));
    elseif numel(hit) > 1
        error("load_test:ambiguous", "'%s' matches several tests:\n  %s", name, strjoin(hit, "\n  "));
    end
    fn = char(hit);

end
