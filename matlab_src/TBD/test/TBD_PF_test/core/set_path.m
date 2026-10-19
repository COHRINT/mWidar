%%% Strictly set a dotted path in a struct.
%%%
%%%   cfg = set_path(cfg, 'TBD.Pb', 0.2)
%%%
%%% Errors if any part of the path does not already exist.

function s = set_path(s, path, val)

    parts = strsplit(char(path), '.');
    node = s;
    for i = 1:numel(parts)
        if ~isstruct(node) || ~isfield(node, parts{i})
            error("set_path:unknownField", "Unknown path 'cfg.%s'", char(path));
        end
        node = node.(parts{i});
    end
    s = setfield(s, parts{:}, val);

end
