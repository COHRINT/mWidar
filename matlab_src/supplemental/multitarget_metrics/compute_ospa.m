function [d_ospa, d_loc, d_card] = compute_ospa(est_xy, gt_xy, c, p)
% COMPUTE_OSPA  Optimal Sub-Pattern Assignment distance.
%
%   d = compute_ospa(est_xy, gt_xy, c, p)
%   [d, d_loc, d_card] = compute_ospa(...)
%
% Schuhmacher / Vo OSPA distance between the set of estimated points
% est_xy and the set of ground-truth points gt_xy.
%
%   d^p = (1/n) * [ min_{sigma} sum_{i=1..m} d_c(x_i, y_sigma(i))^p
%                   + (n - m) * c^p ]
% where m <= n, d_c = min(c, ||.||), and the assignment is optimal.
%
% INPUTS:
%   est_xy - [D x N_est] estimated points
%   gt_xy  - [D x N_gt]  ground-truth points
%   c      - cutoff distance (default 2.0)
%   p      - order (default 2)
%
% OUTPUTS:
%   d_ospa - scalar OSPA distance
%   d_loc  - localization component
%   d_card - cardinality component

    if nargin < 3 || isempty(c), c = 2.0; end
    if nargin < 4 || isempty(p), p = 2;   end

    N_est = size(est_xy, 2);
    N_gt  = size(gt_xy,  2);

    if N_est == 0 && N_gt == 0
        d_ospa = 0; d_loc = 0; d_card = 0;
        return
    end

    n = max(N_est, N_gt);
    m = min(N_est, N_gt);

    if m == 0
        d_ospa = c;
        d_loc  = 0;
        d_card = c;
        return
    end

    cost = zeros(N_est, N_gt);
    for i = 1:N_est
        for j = 1:N_gt
            cost(i, j) = min(c, norm(est_xy(:, i) - gt_xy(:, j)))^p;
        end
    end

    if exist('matchpairs', 'builtin') || exist('matchpairs', 'file')
        M = matchpairs(cost, c^p);
        assigned_cost = 0;
        for r = 1:size(M, 1)
            assigned_cost = assigned_cost + cost(M(r,1), M(r,2));
        end
    else
        assigned_cost = greedy_match_cost(cost);
    end

    card_term = (n - m) * c^p;

    d_loc  = (assigned_cost / n)^(1/p);
    d_card = (card_term     / n)^(1/p);
    d_ospa = ((assigned_cost + card_term) / n)^(1/p);
end


function total = greedy_match_cost(cost)
    [N_est, N_gt] = size(cost);
    used_est = false(N_est, 1);
    used_gt  = false(N_gt,  1);
    [~, ord] = sort(cost(:));
    total = 0;
    for idx = 1:numel(ord)
        [i, j] = ind2sub([N_est, N_gt], ord(idx));
        if ~used_est(i) && ~used_gt(j)
            total = total + cost(i, j);
            used_est(i) = true;
            used_gt(j)  = true;
            if all(used_est) || all(used_gt), break; end
        end
    end
end
