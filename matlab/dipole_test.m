% Simple test of the Sarvas formula

% rQ    dipole position (1x3, units m)
% Q     dipole moment (1x3, units Am)
% R     field points (Nx3, units m)
% r0    center of sphere (1x3, units m)
rQ = [0, 0.07, 0];
Q = [0, 0, -100] * 1e-9;

% sample on a regular grid
[x, y, z] = meshgrid(-0.2:0.01:0.2, -0.2:0.01:0.2, 0);
R = [x(:), y(:), z(:)];

r0 = [0, 0, 0];
B = dipsph_fld(rQ,Q,R,r0);

% set everything inside 10-cm sphere to zero
B(vecnorm(R, 2, 2) < 0.1, :) = 0;

% plotting
hold on;
quiver(x, y, reshape(B(:,1), size(x)), reshape(B(:,2), size(x)));

[xc, yc] = pol2cart(0:pi/100:2*pi, 0.1*ones(1, 2*100+1));
plot(xc, yc);

plot(rQ(:,1), rQ(:,2), 'o');