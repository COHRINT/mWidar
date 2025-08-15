function [y, Signal] = sim_mWidar_image(n_t, GT, M, G, detector, NOISE_FLAG)
    % n_t -> Number of timesteps
    % GT -> Cell array of object Ground Truth positions (x,y). Each cell
    % corresponds to a different objects GT
    % y -> Output of all peaks2 detections
    % M -> Sampling Matrix
    % G -> Recovery Matrix
    % to_plot -> Boolean, true to plot, false to not plot

    n_GT = size(GT, 2);

    Lscene = 4; %physical length of scene in m (square shape)
    npx = 128; %number of pixels in image (same in x&y dims)
    npx2 = npx ^ 2;

    rng(100)

    xgrid = linspace(-2, 2, npx);
    ygrid = linspace(0, Lscene, npx);
    [pxgrid, pygrid] = meshgrid(xgrid, ygrid);
    pxyvec = [pxgrid(:), pygrid(:)];
    dx = xgrid(2) - xgrid(1);
    dy = ygrid(2) - ygrid(1);

    checkbounds_idx = @(coordinate) coordinate > 0 && coordinate < 128;
    checkbounds_x = @(coordinate) coordinate > -2 && coordinate < 2;
    checkbounds_y = @(coordinate) coordinate > 0 && coordinate < 4;

    y = cell(1, n_t);
    Signal = cell(1, n_t);

    for i = 1:n_t

        S = zeros(128, 128);

        for j = 1:n_GT

            X = GT{:, j};

            px = X(1, i);
            py = X(2, i);

            % Convert object positions to matrix coordinates
            if checkbounds_x(px) && checkbounds_y(py)
                Gx = find(px <= xgrid, 1, 'first');
                Gy = find(py <= ygrid, 1, 'first');

            end

            % Check if object is within bounds
            if checkbounds_idx(Gx) && checkbounds_idx(Gy)

                S(Gy, Gx) = 1;
            end

        end

        if all(S == 0)
            % No measurment
            continue
        end

        signal_flat = S';
        signal_flat = signal_flat(:);
        signal_flat = M * signal_flat;
        signal_flat = G' * signal_flat;
        sim_signal = reshape(signal_flat, 128, 128)';

        if NOISE_FLAG
            sim_signal = sim_signal + randn(128);
        end

        blurred = imgaussfilt(sim_signal, 2);
        signal = (blurred - min(blurred(:))) / (max(blurred(:)) - min(blurred(:)));

        if detector == "peaks2"
            [~, peak_y, peak_x] = peaks2(sim_signal', 'MinPeakHeight', 100, 'MinPeakDistance', 15);
        else
            [~, peak_x, peak_y] = CA_CFAR(signal, 0.4, 3, 10);
        end

        pvinds = sub2ind([npx npx], peak_x, peak_y);
        % 
        % figure(99), clf, hold on, view(2)
        % 
        % for j = 1:n_GT
        % 
        %     X = GT{:, j};
        % 
        %     px = X(1, i);
        %     py = X(2, i);
        % 
        %     plot3(px, py, 1000 * ones(length(GT), 1), 'mx', 'MarkerSize', 10, 'LineWidth', 10)
        %     plot3(pxgrid(pvinds), pygrid(pvinds), 1000 * ones(length(peak_x), 1), 'ms', 'MarkerSize', 12, 'LineWidth', 1.2)
        %     surface(pxgrid, pygrid, sim_signal, 'EdgeColor', 'none')
        %     pause(0.1)
        % 
        % end

        y{i} = [pxgrid(pvinds)'; pygrid(pvinds)'];
        Signal{i} = sim_signal;
    end

end
