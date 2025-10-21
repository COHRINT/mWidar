function [xplus] = RKIntegrator(f,x,h,t)

%%RK4 Integrator, non variable time step numerical integrator used to extrapolate a solution to the next time step
% x ~ state at time step h
% h ~ time step
% f ~ function handle, Non-linear ODE we wish to integrate over.
% t ~ current time

k1 = h*f(x,t);
k2 = h*f(x+k1/2,t+h/2);
k3 = h*f(x+k2/2,t+h/2);
k4 = h*f(x+k3,t+h);

xplus = x+(k1+2*k2+2*k3+k4)/6;

end

