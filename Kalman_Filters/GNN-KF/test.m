
Z = mvnrnd([0;0],[1 0.9;0.9 1],10)';
S = cov(Z');

X = [0 0]';

for i = 1:size(Z,2)
    innov = X - Z(:,i);
    d2(i) = mahalanobis(innov,S);
end

d2

figure(); hold on; grid on
scatter(Z(1,:),Z(2,:),'r')
scatter(X(1),X(2),'k')

function [d2] = mahalanobis(innov,S)
    d2 = sqrt(innov\S*innov);
end