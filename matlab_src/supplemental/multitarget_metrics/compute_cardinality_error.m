function stats = compute_cardinality_error(N_est_seq, N_gt_seq)
% COMPUTE_CARDINALITY_ERROR  Per-frame and aggregate count-estimation error.
%
%   stats = compute_cardinality_error(N_est_seq, N_gt_seq)
%
% INPUTS:
%   N_est_seq - [1 x K] estimated target count per frame
%   N_gt_seq  - [1 x K] ground-truth target count per frame
%
% OUTPUTS:
%   stats.err          - signed per-frame error (N_est - N_gt)
%   stats.abs_err      - absolute per-frame error
%   stats.mae          - mean absolute error
%   stats.rmse         - root-mean-square cardinality error
%   stats.exact_pct    - percentage of frames with exact count match
%   stats.within_one_pct - percentage of frames where |err| <= 1

    N_est_seq = N_est_seq(:).';
    N_gt_seq  = N_gt_seq(:).';
    K = numel(N_est_seq);
    if numel(N_gt_seq) ~= K
        error('compute_cardinality_error: length mismatch (%d vs %d)', ...
            K, numel(N_gt_seq));
    end

    err = N_est_seq - N_gt_seq;
    abs_err = abs(err);

    valid = ~isnan(N_est_seq) & ~isnan(N_gt_seq);
    if any(valid)
        stats.mae           = mean(abs_err(valid));
        stats.rmse          = sqrt(mean(err(valid).^2));
        stats.exact_pct     = 100 * mean(err(valid) == 0);
        stats.within_one_pct = 100 * mean(abs_err(valid) <= 1);
    else
        stats.mae           = NaN;
        stats.rmse          = NaN;
        stats.exact_pct     = NaN;
        stats.within_one_pct = NaN;
    end

    stats.err     = err;
    stats.abs_err = abs_err;
    stats.K       = K;
    stats.K_valid = sum(valid);
end
