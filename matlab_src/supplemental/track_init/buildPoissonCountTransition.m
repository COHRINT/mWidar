function T_Nk_given_prev = buildPoissonCountTransition(N_vals, lambda_arrival, lambda_depart)
% BUILDPOISSONCOUNTTRANSITION  Object-count state transition matrix.
%
%   T = buildPoissonCountTransition(N_vals, lambda_arrival, lambda_depart)
%
% Builds T(i,j) = P(N_k = N_vals(i) | N_{k-1} = N_vals(j)) under
% independent Poisson birth/death dynamics:
%   arrivals   A ~ Poisson(lambda_arrival)
%   departures D ~ Poisson(lambda_depart * N_prev), truncated to [0, N_prev]
%   N_k = N_{k-1} - D + A   (clamped to N_vals range)
%
% Promoted from the local function in recursive_TI.m so external callers
% (ProbabilisticEstimator class) can reuse it.

    n_states = numel(N_vals);
    T_Nk_given_prev = zeros(n_states, n_states);

    for n = 1:n_states
        n_prev = N_vals(n);

        a_max = n_states + 10;
        a_vals = 0:a_max;
        pA = local_poisson_pmf(a_vals, lambda_arrival);
        pA(end) = pA(end) + max(0, 1 - sum(pA));
        pA = pA ./ max(sum(pA), eps);

        if n_prev > 0
            d_vals = 0:n_prev;
            pD = zeros(size(d_vals));
            if n_prev > 1
                pD(1:end-1) = local_poisson_pmf(0:n_prev-1, lambda_depart * n_prev);
            end
            pD(end) = max(0, 1 - sum(pD(1:end-1)));
            pD = pD ./ max(sum(pD), eps);
        else
            d_vals = 0;
            pD = 1;
        end

        for ia = 1:numel(a_vals)
            for id = 1:numel(d_vals)
                n_next = n_prev - d_vals(id) + a_vals(ia);
                n_next = min(max(n_next, N_vals(1)), N_vals(end));
                idx_next = n_next - N_vals(1) + 1;
                T_Nk_given_prev(idx_next, n) = ...
                    T_Nk_given_prev(idx_next, n) + pA(ia) * pD(id);
            end
        end
    end

    T_Nk_given_prev = T_Nk_given_prev ./ max(sum(T_Nk_given_prev, 1), eps);
end


function p = local_poisson_pmf(k, lambda)
    if lambda <= 0
        p = double(k == 0);
        return
    end
    p = exp(-lambda + k .* log(lambda) - gammaln(k + 1));
end
