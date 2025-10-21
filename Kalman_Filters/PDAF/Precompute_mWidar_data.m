clear;close all; clc

load("../sampling.mat");
load("../recovery.mat");
load tracks/linear_const_v_mWidarsim.mat xplus
load tracks/linear_const_v_mWidarsim2.mat xplus2

% Length of timevector
n_k = length(xplus);

% Simulate measurments with mWidar Sim
to_plot = 1;

GT = cell(1,2);
GT{1} = xplus;
GT{2} = xplus2;
[y, signal] = sim_mWidar_image(n_k,GT,M,G,to_plot);

Data.y = y;
Data.signal = signal;

%save tracks/cv_mWidarSim_2obj_15.mat Data -mat
%% mWidar Sim Function

function [y, Signal] = sim_mWidar_image(n_t,GT,M,G,to_plot)
% n_t -> Number of timesteps
% GT -> Cell array of object Ground Truth positions (x,y). Each cell
% corresponds to a different objects GT
% y -> Output of all peaks2 detections
% M -> Sampling Matrix
% G -> Recovery Matrix
% to_plot -> Boolean, true to plot, false to not plot

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

    for j = 1:n_GT

        X = GT{j};

        px = X(1,i);
        py = X(4,i);


        % Convert object positions to matrix coordinates
        if checkbounds_x(px) && checkbounds_y(py)
            Gx = find(px <= xgrid,1,'first');
            Gy = find(py <= ygrid,1,'first');

        end
        % Check if object is within bounds
        if checkbounds_idx(Gx) && checkbounds_idx(Gy)

            S(Gy,Gx) = 1;
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
    
    blurred = imgaussfilt(sim_signal,2);
    signal = (blurred - min(blurred(:))) / (max(blurred(:)) - min(blurred(:)));
    %[~,peak_y,peak_x] = peaks2(sim_signal','MinPeakHeight',100,'MinPeakDistance',15);
    [~,peak_x, peak_y] = CA_CFAR(signal,0.4,3,10);
    pvinds = sub2ind([npx npx],peak_x,peak_y);
    
    if to_plot
        figure(99), clf, hold on, view(2)
        for j = 1:n_GT
            
            X = GT{j};

            px = X(1,i);
            py = X(4,i);

            plot3(px,py,1000*ones(length(GT),1),'mx','MarkerSize',10,'LineWidth',10)
            plot3(pxgrid(pvinds),pygrid(pvinds),1000*ones(length(peak_x),1),'ms','MarkerSize',12,'LineWidth',1.2)
            surface(pxgrid,pygrid,sim_signal,'EdgeColor','none')

        end  
    end
    
    y{i} = [pxgrid(pvinds)';pygrid(pvinds)'];
    Signal{i} = sim_signal;
end

end
