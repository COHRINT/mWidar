function [Meas,Height] = SimulateImages(X,M,G)

%{
%%%%%%% SimulateImages %%%%%%%%%%%
This function will simulate an mWidar image for a given state/sigma point,
and use a peak finding algorithm to return the expected measurment from an
mWidar image. This will likely be extremely expensive, but will allow us to
accuratly simulate the measurment noise associated with mWidar images.

Inputs :
    X - Our full object state
    M - Sampling Matrix
    G - Recovery Matrix
Outputs:
    Meas - [x,y] coordinate of our simulated mWidar image peak
    Height - the height of the peak
%}

%% Define grid


Lscene = 4; %physical length of scene in m (square shape)
npx = 128; %number of pixels in image (same in x&y dims)
npx2 = npx^2;


xgrid = linspace(-2,2,npx);
ygrid = linspace(0,Lscene,npx);
[pxgrid,pygrid] = meshgrid(xgrid,ygrid);
pxyvec = [pxgrid(:), pygrid(:)];
dx = xgrid(2)-xgrid(1);
dy = ygrid(2)-ygrid(1);


% Ensure object is withing image frame
checkbounds = @(coordinate) coordinate >= 0 && coordinate <= 128;


S = zeros(128,128);

% Issue Likely Here
x = X(1); y = X(4);

Gridx = find(x <= pxgrid(1,:),1,'first');
Gridy = find(y <= pygrid(:,1),1,'first');

% Check if x & y are outside of the grid
if isempty(Gridx)
    Gridx = 128;
elseif isempty(Gridy)
    Gridy = 128;
end

l = 1;

if checkbounds(Gridx) && checkbounds(Gridy)
            S(Gridy, Gridx) = 1;
end

% Row-major flatten the signal 
signal_flat = S';
signal_flat = signal_flat(:);
signal_flat = M * signal_flat;
signal_flat = G' * signal_flat;
sim_signal = reshape(signal_flat, 128, 128)';
sim_signal = sim_signal./max(sim_signal);

[H,peak_y,peak_x] = peaks2(sim_signal,'MinPeakHeight',0.6,'MinPeakDistance',128);

i = find(H == max(H));

peak_y = peak_y(i);
peak_x = peak_x(i);

Height = H(i);

pvinds = sub2ind([npx npx],peak_y,peak_x);

xLoc = pxgrid(pvinds); yLoc = pygrid(pvinds);

Meas = [xLoc;yLoc];

% figure(1); clf; hold on;
% s = surface(sim_signal, 'FaceAlpha', 0.5);
% scatter(peak_x,peak_y)
% s.EdgeColor = 'none';
% colormap jet



end

