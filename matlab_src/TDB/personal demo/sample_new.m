function [xk] = sample_new(z,p)

% Find index's of z that surpass the threshold value
idx = find(z>p.gamma);
[rx,ry] = ind2sub(size(z),idx);

assert(size(rx,1) == size(ry,1))

% Match
r = [rx ry];
n = numel(rx);
m = randi(n);

posk = r(m,:);

vx = p.v_min + (p.v_max - p.v_min) * rand();
vy = p.v_min + (p.v_max - p.v_min) * rand();

xk = [posk(1); vx; posk(2); vy];


end