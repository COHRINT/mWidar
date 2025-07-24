function run_all_parallel()
% Run all dataset/filter combinations in parallel using MATLAB's parfor

% Check if Parallel Computing Toolbox is available
if ~license('test', 'Distrib_Computing_Toolbox')
    error('Parallel Computing Toolbox not available');
end

% Start parallel pool if not already running
if isempty(gcp('nocreate'))
    parpool('local', 4); % Adjust number of workers as needed
end

% Define all combinations
datasets = ["T1_near", "T2_far", "T3_border", "T4_parab", "T5_parab_noise"];
filters = ["HybridPF", "KF"];

% Create all combinations
combinations = [];
for i = 1:length(datasets)
    for j = 1:length(filters)
        combinations(end+1,:) = [datasets(i), filters(j)];
    end
end

% Run in parallel
fprintf('Starting %d parallel jobs...\n', size(combinations, 1));

parfor i = 1:size(combinations, 1)
    dataset = combinations(i, 1);
    filter = combinations(i, 2);
    
    try
        fprintf('Worker %d: Starting %s with %s\n', getCurrentTask().ID, dataset, filter);
        main(dataset, filter);
        fprintf('Worker %d: Completed %s with %s\n', getCurrentTask().ID, dataset, filter);
    catch ME
        fprintf('Worker %d: Error in %s with %s: %s\n', getCurrentTask().ID, dataset, filter, ME.message);
    end
end

fprintf('All parallel jobs completed!\n');

end
