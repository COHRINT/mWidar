%%% Filter factory. Maps cfg.filter to a constructed filter object.
%%%
%%% Every filter returned here must provide
%%%   R = f.run(scenario, <cfg.run as name/value>)
%%%   [fig, figAnim] = f.show(R, scenario, <cfg.show as name/value>)
%%%
%%% Adding a filter:
%%%   1. add its parameter block to default_cfg (e.g. cfg.EKF)
%%%   2. add a case below that builds it from that block
%%%   3. add a variant for it in config/variants.m

function f = build_filter(cfg)

    switch cfg.filter
        case "TBD_PF"
            args = name_value(cfg.TBD);
            f = TBD_PF(args{:});
        otherwise
            error("build_filter:unknown", "Unknown filter '%s'", cfg.filter);
    end

end
