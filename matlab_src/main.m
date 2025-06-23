% Do the main we discussed

performance = cell(1,n_k);

for i = 1:n_k
    performance{i}.x = X(:,1); % State
    performance{i}.P = P0; % State Covaraince
    
    % Include measurements, GT
    
end

% current_class = GNN_KF(X, P0, Q, R, H, F);
current_class = GNN_HMM(X, P0, Q, R, H, F);

%% State Estimation

for i = 1:n_k

    current_class.timestep(current_measurement)

    % update performance
    perofmrance{i}.x = current_class.x;
    performance{i}.P = current_class.P;

end

%% Plotting