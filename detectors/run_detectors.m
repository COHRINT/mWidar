function [y, Signal] = run_detectors(GT,M,G, detector, k)
    
    %% same as sim_mWidar_image, just changed to output in pixel space

    n_GT = length(GT); % Number of objects in scene

    checkbounds_idx = @(coordinate) coordinate > 0 && coordinate < 128; % Ensure object not out of bounds

    Pfa = 0.36; % Probability of false alarm -- 36 WORKS THE BEST
    Ng = 3;    % Guard cells
    Nr = 20;   % Training cells


    S = zeros(128,128);
    
    for i = 1:n_GT
        X = GT{i};

        px = floor(X(1,k));
        py = floor(X(2,k));

        if checkbounds_idx(px) && checkbounds_idx(py)
            S(py, px) = 1; % Populate the scene
        end
    end

    if all(S == 0)
        % No measurement - initialize empty cells
        y = [];
        Signal = [];
        return;
    end

    signal_flat = S';
    signal_flat = signal_flat(:);
    signal_flat = M * signal_flat;
    signal_flat = G' * signal_flat;
    sim_signal = reshape(signal_flat, 128, 128)';
    signal = imgaussfilt(sim_signal, 2);

    if detector == "peaks2"
        [~, peak_y, peak_x] = peaks2(signal, 'MinPeakHeight', 100, 'MinPeakDistance', 25);
    else
        [~, peak_y, peak_x] = CA_CFAR(signal, Pfa, Ng, Nr);
    end

    y = [peak_x peak_y];
    valid_idx = y(:, 2) >= 15;
    y = y(valid_idx,:);
    Signal = sim_signal;

end