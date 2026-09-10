function rp = RT(rm, p)
    %{
        Regime Transition (Table 3.9 Ristic)
    %}
    rp = false(1,p.N);
    C = [zeros(2,1), cumsum(p.PI, 2)];
    u = rand(p.N,1);
    for n = 1:p.N
        i    = rm(n) + 1;
        rp(n) = find(u(n) <= C(i,2:end), 1) - 1;   % -1 maps column back to {0,1}
    end
    
end