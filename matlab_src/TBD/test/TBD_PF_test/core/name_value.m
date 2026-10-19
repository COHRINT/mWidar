%%% Struct -> {'Field1', v1, 'Field2', v2, ...} for name/value calls.
%%%
%%%   args = name_value(cfg.env);
%%%   e = environment(args{:});

function c = name_value(s)

    f = fieldnames(s);
    c = cell(1, 2 * numel(f));
    c(1:2:end) = f;
    c(2:2:end) = struct2cell(s);

end
