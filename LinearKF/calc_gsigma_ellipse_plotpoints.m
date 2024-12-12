%calc_gsigma_ellipse_plotpoints.m
%%Does all the computations needed to plot 2D Gaussian ellipses properly.
%%Takes in the Gaussian mean (muin), cov matrix (Sigma, positive definite),
%%g-sigma value, and number of points to generate for plotting. 
function [X, Y] = calc_gsigma_ellipse_plotpoints(muin,Sigma,g,npoints)
%%align the Gaussian along its principal axes
[R,D,thetalocx] = subf_rotategaussianellipse(Sigma,g);
%%pick semi-major and semi-minor "axes" (lengths)
if Sigma(1)<Sigma(4) %use if sigxx<sigyy
    a = 1/sqrt(D(4)); 
    b = 1/sqrt(D(1)); 
elseif Sigma(1)>=Sigma(4)
    a= 1/sqrt(D(1)); %use if sigxx<sigyy
    b= 1/sqrt(D(4)); 
end

%%calculate points of ellipse:
mux = muin(1);
muy = muin(2);
if Sigma(2)~=0
    [X Y] = calculateEllipse(mux, muy, a, b, rad2deg(thetalocx),npoints); 
else %if there are no off-diagonal terms, then no rotation needed
    [X Y] = calculateEllipse(mux, muy, a, b, 0,npoints);
end

%%call the rotate gaussian ellipse thingy as a local subfunction (there is
%%also a separate fxn m-file for this, too)
function [R,D,thetalocx] = subf_rotategaussianellipse(Sigma,g)
P = inv(Sigma);
P = 0.5*(P+P'); %symmetrize

a11 = P(1);
a12 = P(2);
a22 = P(4);
c = -g^2;

mu = 1/(-c); %can define mu this way since b1=0,b2=0 b/c we are mean centered
m11 = mu*a11;
m12 = mu*a12;
m22 = mu*a22;

%%solve for eigenstuff
lambda1 = 0.5*(m11 + m22 + sqrt((m11-m22).^2 + 4*m12.^2) );
lambda2 = 0.5*(m11 + m22 - sqrt((m11-m22).^2 + 4*m12.^2) );
% % b = 1/sqrt(lambda1); %semi-minor axis for standard ellipse (length)
% % a = 1/sqrt(lambda2); %semi-major axis for standard ellipse (length)
D = diag([lambda2,lambda1]); %elements are 1/a^2 and 1/b^2, respectively
%%Choose the mahor axis direction of the ellipse
if m11>=m22
    u11 = lambda1 - m22;
    u12 = m12;
elseif m11<m22
    u11 = m12;
    u12 = lambda1 - m11;    
end
norm1 = sqrt( u11.^2 + u12.^2  );
U1 =  ([u11; u12]) / norm1; %major axis direction
U2 = [-u12; u11]; %minor axis direction
R = [U1,U2];
% if sum(sum(isnan(R)))>0
%    R = eye(2); %default hack for now in case of degenerate stuff
% end

thetalocx = 0.5*atan( -2*a12/(a22-a11) );
% if isnan(thetalocx)
%     thetalocx = 0; %default hack for now...
% end

end

end