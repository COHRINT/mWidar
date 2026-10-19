%%% Strict recursive merge of a partial struct ov onto s.
%%%
%%%   cfg = apply_overrides(default_cfg(), struct('TBD', struct('Pb', 0.2)))
%%%
%%% Errors on any field of ov that s does not have, so typos are caught.
%%% Scalar sub-structs are merged field by field; anything else (matrices,
%%% cells, strings) is replaced whole.

function s = apply_overrides(s, ov, where)

    if nargin < 3, where = "cfg"; end
    if isempty(ov), return; end
    assert(isstruct(ov) && isscalar(ov), "apply_overrides: overrides for '%s' must be a scalar struct", where);

    for f = string(fieldnames(ov))'
        here = where + "." + f;
        if ~isfield(s, f)
            error("apply_overrides:unknownField", ...
                "Unknown field '%s'. Valid fields: %s", here, strjoin(string(fieldnames(s))', ', '));
        end
        if isstruct(ov.(f)) && isscalar(ov.(f)) && isstruct(s.(f)) && isscalar(s.(f))
            s.(f) = apply_overrides(s.(f), ov.(f), here);
        else
            s.(f) = ov.(f);
        end
    end

end
