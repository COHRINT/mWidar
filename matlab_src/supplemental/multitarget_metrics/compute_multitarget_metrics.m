function metrics = compute_multitarget_metrics(est_xy_seq, gt_xy_seq, varargin)
% COMPUTE_MULTITARGET_METRICS  One-stop wrapper for multi-target eval.
%
%   metrics = compute_multitarget_metrics(est_xy_seq, gt_xy_seq, ...)
%
% Per-frame Hungarian assignment, per-target RMSE (with track relabelling
% via the most-frequent assignment), cardinality stats, OSPA timeline,
% and (optional) data-association accuracy.
%
% INPUTS:
%   est_xy_seq - {1 x K} cell of [2 x N_est_k] estimated positions
%   gt_xy_seq  - {1 x K} cell of [2 x N_gt_k]  ground-truth positions
%
% OPTIONS:
%   'OSPACutoff'   - scalar c (default 2.0)
%   'OSPAOrder'    - scalar p (default 2)
%   'Gate'         - assignment gate (default 2.0)
%   'EstAssoc'     - {1 x K} cell est measurement->track (for DA acc)
%   'GtAssoc'      - {1 x K} cell gt measurement->target (for DA acc)
%
% OUTPUTS (struct fields):
%   .K                    - number of frames
%   .N_est_seq            - [1 x K]
%   .N_gt_seq             - [1 x K]
%   .cardinality          - struct from compute_cardinality_error
%   .ospa                 - [1 x K] OSPA per frame
%   .ospa_loc, .ospa_card - components
%   .assignment_seq       - {1 x K} of [pair x 2] est_idx, gt_idx
%   .per_target_rmse      - [N_gt_max x 1] (NaN if target never seen)
%   .da                   - struct from evaluate_data_association (if requested)

    p = inputParser;
    addParameter(p, 'OSPACutoff', 2.0, @isnumeric);
    addParameter(p, 'OSPAOrder',  2,   @isnumeric);
    addParameter(p, 'Gate',       2.0, @isnumeric);
    addParameter(p, 'EstAssoc',   [],  @(x) isempty(x) || iscell(x));
    addParameter(p, 'GtAssoc',    [],  @(x) isempty(x) || iscell(x));
    parse(p, varargin{:});

    K = numel(est_xy_seq);
    if numel(gt_xy_seq) ~= K
        error('compute_multitarget_metrics: K mismatch (%d vs %d)', ...
            K, numel(gt_xy_seq));
    end

    N_est_seq = zeros(1, K);
    N_gt_seq  = zeros(1, K);
    ospa      = nan(1, K);
    ospa_loc  = nan(1, K);
    ospa_card = nan(1, K);
    assignment_seq = cell(1, K);

    N_gt_max = 0;
    for k = 1:K
        N_gt_max = max(N_gt_max, size(gt_xy_seq{k}, 2));
    end

    % Per-target RMSE accumulators (gt-target-indexed)
    sse  = zeros(N_gt_max, 1);
    cnt  = zeros(N_gt_max, 1);

    for k = 1:K
        N_est_seq(k) = size(est_xy_seq{k}, 2);
        N_gt_seq(k)  = size(gt_xy_seq{k}, 2);

        assignment_seq{k} = assign_targets_hungarian( ...
            est_xy_seq{k}, gt_xy_seq{k}, p.Results.Gate);

        for r = 1:size(assignment_seq{k}, 1)
            i = assignment_seq{k}(r, 1);
            j = assignment_seq{k}(r, 2);
            d = norm(est_xy_seq{k}(:, i) - gt_xy_seq{k}(:, j));
            sse(j) = sse(j) + d^2;
            cnt(j) = cnt(j) + 1;
        end

        [ospa(k), ospa_loc(k), ospa_card(k)] = compute_ospa( ...
            est_xy_seq{k}, gt_xy_seq{k}, ...
            p.Results.OSPACutoff, p.Results.OSPAOrder);
    end

    per_target_rmse = nan(N_gt_max, 1);
    for j = 1:N_gt_max
        if cnt(j) > 0
            per_target_rmse(j) = sqrt(sse(j) / cnt(j));
        end
    end

    metrics.K               = K;
    metrics.N_est_seq       = N_est_seq;
    metrics.N_gt_seq        = N_gt_seq;
    metrics.cardinality     = compute_cardinality_error(N_est_seq, N_gt_seq);
    metrics.ospa            = ospa;
    metrics.ospa_loc        = ospa_loc;
    metrics.ospa_card       = ospa_card;
    metrics.assignment_seq  = assignment_seq;
    metrics.per_target_rmse = per_target_rmse;

    if ~isempty(p.Results.EstAssoc) && ~isempty(p.Results.GtAssoc)
        metrics.da = evaluate_data_association( ...
            p.Results.EstAssoc, p.Results.GtAssoc, ...
            est_xy_seq, gt_xy_seq, p.Results.Gate);
    end
end
