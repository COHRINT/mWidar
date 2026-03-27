%% plot_all_experiment_results.m
% Recursively scans a selected folder for filter-based .mat files,
% groups them by filter type, and plots Average +/- 2*sigma for:
%   - Position Error
%   - ESS / N_particles
% Splits the filters into 3 figures (KF, HMM, PF) for readability.
% Y-axes are plotted in log scale.

function plot_all_experiment_results(folder_path)

if nargin < 1
    folder_path = uigetdir(fullfile(pwd, 'data'), 'Select folder containing experiment .mat files');
    if isequal(folder_path, 0)
        disp('No folder selected. Exiting.');
        return;
    end
end

%% 1. Find all .mat files recursively
files = dir(fullfile(folder_path, '**', '*.mat'));
if isempty(files)
    disp('No .mat files found in the selected folder.');
    return;
end

%% 2. Load and group by filter_name
filter_data = containers.Map('KeyType', 'char', 'ValueType', 'any');

for i = 1:numel(files)
    fpath = fullfile(files(i).folder, files(i).name);
    try
        vars = whos('-file', fpath);
        if ismember('R', {vars.name})
            tmp = load(fpath, 'R');
            if isfield(tmp, 'R') && isfield(tmp.R, 'filter_name')
                R = tmp.R;
                fname = R.filter_name;
            
            % Compute metrics
            pos_err = vecnorm(R.x_est(1:2,:) - R.GT(1:2,:));
            
            ess_frac = [];
            if isfield(R, 'ESS') && any(~isnan(R.ESS)) && isfield(R, 'cfg') && isfield(R.cfg, 'N_particles')
                ess_frac = R.ESS ./ R.cfg.N_particles;
            end
            
            dt = 0.1;
            if isfield(R, 'params') && isfield(R.params, 'dt')
                dt = R.params.dt;
            end
            
            % Initialize dict entry if it's new
            if ~isKey(filter_data, fname)
                dat = struct();
                dat.pos_errs = [];
                dat.ess_fracs = [];
                dat.dt = dt;
                filter_data(fname) = dat;
            end
            
            % Append data
            dat = filter_data(fname);
            dat.pos_errs(end+1, 1:length(pos_err)) = pos_err;
            if ~isempty(ess_frac)
                dat.ess_fracs(end+1, 1:length(ess_frac)) = ess_frac;
            end
            filter_data(fname) = dat;
            end
        end
    catch ME
        % Not a valid filter result, ignore
    end
end

keys = filter_data.keys;
if isempty(keys)
    disp('No valid filter-based results found.');
    return;
end

%% 3. Classify filters into KF, HMM, PF
groups = struct('KF', {}, 'HMM', {}, 'PF', {});
groups(1).KF = {}; groups(1).HMM = {}; groups(1).PF = {};

for i = 1:length(keys)
    k = keys{i};
    ku = strrep(upper(k), '_', '-'); % normalize string to use dash
    if contains(ku, 'HMM')
        groups.HMM{end+1} = k;
    elseif contains(ku, 'GNN-KF') || contains(ku, 'PDA-KF') || contains(ku, 'KF-RBPF') || strcmp(ku, 'KF')
        groups.KF{end+1} = k;
    else
        % Remaining 3 are PF-tracking based
        groups.PF{end+1} = k;
    end
end

group_names = {'KF', 'HMM', 'PF'};
colors = lines(10); % Color palette

%% 4. Plot each group
for g_idx = 1:3
    gname = group_names{g_idx};
    gkeys = groups.(gname);
    if isempty(gkeys), continue; end
    
    % --- Position Error Figure ---
    fig_pos = figure('Name', sprintf('%s Variants - Position Error', gname), 'Position', [100+g_idx*50, 100, 800, 500]);
    hold on; grid on;
    set(gca, 'YScale', 'log');
    
    % --- ESS Fraction Figure ---
    fig_ess = figure('Name', sprintf('%s Variants - ESS/N_p', gname), 'Position', [100+g_idx*50, 650, 800, 500]);
    hold on; grid on;
    % set(gca, 'YScale', 'log'); % ESS fraction in log or linear? Let's use log as requested for y-axes.
    set(gca, 'YScale', 'log');
    
    for k_idx = 1:length(gkeys)
        fname = gkeys{k_idx};
        dat = filter_data(fname);
        col = colors(k_idx, :);
        
        tvec = (0:(size(dat.pos_errs,2)-1)) * dat.dt;
        
        % Plot Position Error
        figure(fig_pos);
        mean_pe = mean(dat.pos_errs, 1, 'omitnan');
        std_pe  = std(dat.pos_errs, 0, 1, 'omitnan');
        
        % Shaded region for +/- 2 sigma
        fill_x = [tvec, fliplr(tvec)];
        fill_y = [(mean_pe + 2*std_pe), fliplr(max(1e-10, mean_pe - 2*std_pe))]; % avoid <=0 for log scale
        
        fill(fill_x, fill_y, col, 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
        plot(tvec, mean_pe, 'Color', col, 'LineWidth', 2, 'DisplayName', strrep(fname, '_', '\_'));
        
        % Plot ESS/N_p if available
        if ~isempty(dat.ess_fracs)
            figure(fig_ess);
            mean_ess = mean(dat.ess_fracs, 1, 'omitnan');
            std_ess  = std(dat.ess_fracs, 0, 1, 'omitnan');
            
            fill_ess_y = [(mean_ess + 2*std_ess), fliplr(max(1e-10, mean_ess - 2*std_ess))];
            
            fill(fill_x, fill_ess_y, col, 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
            plot(tvec, mean_ess, 'Color', col, 'LineWidth', 2, 'DisplayName', strrep(fname, '_', '\_'));
        end
    end
    
    % Decorate Pos figure
    figure(fig_pos);
    xlabel('Time (s)');
    ylabel('Position Error (m) (Log Scale)');
    title(sprintf('Position Error vs Time (%s Variants)', gname));
    legend('Location', 'best');
    ylim([1e-3, 10]); % Generic limits that look good on log scale
    
    % Decorate ESS figure
    figure(fig_ess);
    if ~isempty(get(gca, 'Children'))
        xlabel('Time (s)');
        ylabel('ESS / N_{particles} (Log Scale)', 'Interpreter', 'tex');
        title(sprintf('ESS Fraction vs Time (%s Variants)', gname));
        legend('Location', 'best');
        ylim([1e-3, 1.5]);
    else
        close(fig_ess); % close if empty (e.g. KF has no ESS)
    end
end

disp('Plotting complete.');
end
