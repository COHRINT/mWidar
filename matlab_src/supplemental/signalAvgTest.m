%%% Script to test heuristic track count process

%% --- Environment Configuration ---
clc; close all; clear
% Add paths for MATLAB functions
addpath(fullfile('DA_Track'))                          % base: DA_Filter, KF, HMM
addpath(fullfile('DA_Track', 'multi'))                 % multi-target filters
addpath(fullfile('supplemental'))
addpath(fullfile('supplemental', 'Final_Test_Tracks'))
addpath(fullfile('supplemental', 'Final_Test_Tracks', 'MultiObj'))

true_ct  = 3;

% Construct DATASET string correctly using string concatenation or sprintf
DATASET = "multi_obj_" + num2str(true_ct) + "_TI_test";
% If later code expects a char array, convert to char:
DATASET = char(DATASET);
GIF = false;
load(fullfile('supplemental', 'Final_Test_Tracks', 'MultiObj', [DATASET, '.mat']), 'Data');

Pfa = 0.285;   % false alarm probability (tuned for this scene)
Ng  = 15;       % guard cells
Nr  = 20;      % training (reference) cells

%% ---- Scene / grid setup ------------------------------------------------
npx    = 128;
Lscene = 4;           % scene height [m]
xgrid  = linspace(-2, 2,      npx);  % [-2, 2] m
ygrid  = linspace( 0, Lscene, npx);  % [0,  4] m
[pxgrid, pygrid] = meshgrid(xgrid, ygrid);

% Hyperparams
T = 25; % Sliding window for frames
thr = 0.7; % Intensity threshold
cluster_radius = 0.35; % Merge detections closer than this distance [m]


% Unpack Data
GT = Data.GT;
z = Data.y;
signal = Data.signal;


n_k = size(GT{1}, 2);
nt = size(GT,2);
track_ct = zeros(size(GT{1}, 2),1);
window = cell(1,T);
fig = [];
if GIF
    fig = figure('Color', 'w', 'Position', [100 100 1200 800]);
end

for k = 1:n_k
    
    filledIdx = find(~cellfun(@isempty, window), 1, 'last');
    if isempty(filledIdx)
        filledIdx = 0;
    end

    % Insert the newest frame at index 1 and shift older frames right.
    if filledIdx > 0
        window(2:min(T, filledIdx + 1)) = window(1:min(T - 1, filledIdx)); % Shift ove
    end
    window{1} = signal{k};

    % Average window
    if filledIdx == T-1 || filledIdx == T
        avgSignal = mean(cat(ndims(window{1}) + 1, window{:}), ndims(window{1}) + 1);
    else
        continue
    end
    

    % Process avgSignal
    blurred = imgaussfilt(avgSignal, 1.3);
    blurred(1:20,:) = NaN;
    signal_scaled = asinh(blurred);
    signal_normalized  = (signal_scaled - min(signal_scaled(:))) / (max(signal_scaled(:)) - min(signal_scaled(:)));
    
    % Run through detector
    [intensity, peak_x, peak_y] = CA_CFAR(signal_normalized(21:128,:), Pfa, Ng, Nr);
    peak_x = peak_x + 20;
    meas_xy = zeros(2,0);
    valid = false(1,0);
    valid_meas_xy = zeros(2,0);
    clustered_meas_xy = zeros(2,0);

    if ~isempty(peak_x)
        pvinds   = sub2ind([npx, npx], peak_x, peak_y);
        meas_xy  = [pxgrid(pvinds)'; pygrid(pvinds)'];
        % Remove detections below y=0.5 m (clutter floor)
        valid    = meas_xy(2,:) >= 0.5 & signal_normalized(pvinds)' > thr;
        valid_meas_xy = meas_xy(:,valid);
        clustered_meas_xy = clusterNearbyDetections(valid_meas_xy, cluster_radius);
    else
        fprintf('  [t=%d] No detections\n', k);

    end

    % Plot for validity
    if GIF
        clf(fig);
        axAvg = axes(fig);
        surf(axAvg, pxgrid, pygrid, signal_normalized, 'EdgeColor', 'none');
        view(axAvg, 2);
        hold(axAvg, 'on');
        if ~isempty(clustered_meas_xy)
            scatter3(axAvg, clustered_meas_xy(1,:), clustered_meas_xy(2,:), max(signal_normalized(:)) + 1, 'wo', 'LineWidth', 1.5);
            scatter3(axAvg, valid_meas_xy(1,:), valid_meas_xy(2,:), max(signal_normalized(:)) + 1, 'bo', 'LineWidth', 1.5);
        end
        axis(axAvg, 'equal');
        axis(axAvg, 'tight');
        xlim(axAvg, [-2, 2]);
        ylim(axAvg, [0, 4]);
        xlabel(axAvg, 'x (m)');
        ylabel(axAvg, 'y (m)');
        title(axAvg, sprintf('Average Signal + Detections (c=%d)', size(clustered_meas_xy,2)));
        colormap(axAvg, turbo);
        colorbar(axAvg);
        hold(axAvg, 'off');
        drawnow;
    end
    fprintf("Estimated obj count: %d [t=%d] \n",size(clustered_meas_xy,2),k )

    track_ct(k) = size(clustered_meas_xy,2);
end

% P(mk | Nk = true_ct)
P = zeros(max(track_ct) + 1,1);
for i = 1:numel(track_ct)
    det = track_ct(i) + 1;
    
    P(det) = P(det) + 1;
end

figure;
bar(P./sum(P))
xticklabels({0:max(track_ct)})
pct_correct = sum(track_ct(T:end) == true_ct)/ (n_k-T+1) * 100;

summaryFig = figure('Color', 'w');
summaryAx = axes(summaryFig);
hold(summaryAx, 'on');
plot(summaryAx, track_ct, 'LineWidth', 1.5)
plot(summaryAx, true_ct*ones(1,n_k), 'k--')

title(summaryAx, sprintf("Heuristic Track Init Object Count Estimate. Pct Correct = %.2f%%", round(pct_correct,2)))
xlim(summaryAx, [0 n_k])
legend(summaryAx, "Estimated Count", "True Object Count")
hold(summaryAx, 'off');
png_filename = "TI_Obj_count_" + num2str(true_ct);
print(summaryFig, png_filename, '-dpng')


function clustered_xy = clusterNearbyDetections(meas_xy, cluster_radius)
    if isempty(meas_xy)
        clustered_xy = zeros(2,0);
        return
    end

    n_det = size(meas_xy, 2);
    dist_mat = pdist2(meas_xy', meas_xy');
    adjacency = dist_mat <= cluster_radius;

    visited = false(1, n_det);
    clustered_xy = zeros(2,0);

    for i = 1:n_det
        if visited(i)
            continue
        end

        queue = i;
        component = i;
        visited(i) = true;

        while ~isempty(queue)
            current = queue(1);
            queue(1) = [];

            neighbors = find(adjacency(current,:) & ~visited);
            if ~isempty(neighbors)
                visited(neighbors) = true;
                queue = [queue, neighbors]; %#ok<AGROW>
                component = [component, neighbors]; %#ok<AGROW>
            end
        end

        clustered_xy(:, end+1) = mean(meas_xy(:,component), 2); %#ok<AGROW>
    end
end
