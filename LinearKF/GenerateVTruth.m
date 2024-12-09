function [V] = GenerateVTruth(x0)

V = zeros(6,100);
tspan = linspace(0,10,100);
% 2 seperate SS modals
A_V1 =  [0 1 0 0 0 0;
          0 0 0 0 0 0 ;
          0 0 0 0 0 0;
          0 0 0 0 1 0;
          0 0 0 0 0 0;
          0 0 0 0 0 0];



A_V3 =   [0 0 0 0 0 0;
          0 0 0 0 0 0 ;
          0 0 0 0 0 0;
          0 0 0 0 1 0;
          0 0 0 0 0 1.5;
          0 0 0 0 0 0];

% Simulate
for k = 1:100
    if k <= 50
        V(:,k) = expm(A_V1.*tspan(k))*x0;
    elseif k > 50 && k <= 75
        V(:,k) = [V(1,50);0;0;V(4,50);0;0];
    elseif k > 75
        if k ==76
            x0 = [V(1,50);0;0;V(4,50);0.25;0.5];
        end
        V(:,k) = expm(A_V3.*tspan(k-75))*x0;
    end

end

end

