function [y, Signal] = sim_mWidar_image(n_t,GT,M,G,detector)
% Inputs:
% n_t -> Number of timesteps
% GT -> Array/Matrix of objects GT
% y -> Output of all peaks2/CFAR detections
% M -> Sampling Matrix
% G -> Recovery Matrix
% detector -> string to indicate which detector to use. 
% "peaks2" will use peaks2, "CFAR" will use CA-CFAR
% Outputs:
% y -> Every detection at each timestep, stored in a cell array
% Signal -> mWidar image at each tstep, also stored in cell array

n_GT = size(GT,2);

Lscene = 4; %physical length of scene in m (square shape)
npx = 128; %number of pixels in image (same in x&y dims)
npx2 = npx^2;


xgrid = linspace(-2,2,npx);
ygrid = linspace(0,Lscene,npx);
[pxgrid,pygrid] = meshgrid(xgrid,ygrid);
pxyvec = [pxgrid(:), pygrid(:)];
dx = xgrid(2)-xgrid(1);
dy = ygrid(2)-ygrid(1);

checkbounds_idx = @(coordinate) coordinate > 0 && coordinate < 128;
checkbounds_x = @(coordinate) coordinate > -2 && coordinate < 2;
checkbounds_y = @(coordinate) coordinate > 0 && coordinate < 4;

y = cell(1,n_t);
Signal = cell(1,n_t);

for i = 1:n_t

    S = zeros(128,128);


    X = GT;

    px = X(1,i);
    py = X(2,i);


    % Convert object positions to matrix coordinates
    if checkbounds_x(px) && checkbounds_y(py)
       Gx = find(px <= xgrid,1,'first');
       Gy = find(py <= ygrid,1,'first');

    end
    % Check if object is within bounds
    if checkbounds_idx(Gx) && checkbounds_idx(Gy)
    S(Gy,Gx) = 1;
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
    
    blurred = imgaussfilt(sim_signal,2); % Blur
    signal = (blurred - min(blurred(:))) / (max(blurred(:)) - min(blurred(:))); % Normalize
    if detector == "peaks2"
        [~,peak_y,peak_x] = peaks2(sim_signal','MinPeakHeight',20,'MinPeakDistance',15);
    elseif detector == "CFAR"
            [~,peak_x, peak_y] = CA_CFAR(signal,0.4,3,10);
    else
        fprintf("Improper detector input, defaulting to peaks2 \n")
        [~,peak_y,peak_x] = peaks2(sim_signal','MinPeakHeight',100,'MinPeakDistance',15);
    end

    pvinds = sub2ind([npx npx],peak_x,peak_y);
   
    
    y{i} = [pxgrid(pvinds)';pygrid(pvinds)'];
    Signal{i} = sim_signal;
end

end