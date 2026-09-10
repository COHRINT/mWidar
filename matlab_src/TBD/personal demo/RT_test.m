%% Unit Test script for the RT function

%%% Initialize a prior existence prob (all false)
p = init_params();

Em = false(1,p.N);

Ep = RT(Em, p);

% sanity check first
fprintf('T1: Em: %d of %d set | Ep: %d of %d set\n', ...
        nnz(Em), numel(Em), nnz(Ep), numel(Ep));

Ep2 = RT(Ep, p);

% sanity check 
fprintf('T2: Em: %d of %d set | Ep: %d of %d set\n', ...
        nnz(Ep), numel(Ep), nnz(Ep2), numel(Ep2));

Ep3 = RT(Ep2, p);

% sanity check 
fprintf('T3: Em: %d of %d set | Ep: %d of %d set\n', ...
        nnz(Ep2), numel(Ep2), nnz(Ep3), numel(Ep3));