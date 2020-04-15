%function B=dipfld1V(rQ,Q,R,r0);
%
% Calculate the magnetic field caused by current dipoles, assumed to
% be located inside a r0-centered conductor sphere.  Formula from
% Hamalainen et al (1993). 
%
% Input:
% rQ    dipole positions (Mx3, units m)
% Q     dipole moments (Mx3, units Am)
% R     field points (Nx3, units m)
% r0    center of sphere (1x3, units m)
%
% Output:
% B    magnetic field (1x3, units T)
%
function B=dipsph_fld(rQ,Q,R,r0);
N=size(R,1);
R=R-ones(N,1)*r0;  % field points rel to origin
rQ=rQ-r0;  % rQ rel to origin
rn=sqrt(sum(R.^2,2));  % field point distances
A=R-ones(N,1)*rQ;  % field point - dipole 
an=sqrt(sum(A.^2,2));  % field point - dipole distances
F=an.*(rn.*an+rn.^2-R*rQ');
% gradient formula
v1=an.^2./rn+dot(A,R,2)./an+2*an+2*rn;
v2=an+2*rn+dot(A,R,2)./an;
M2=repmat(rQ,N,1);
gF=bsxfun(@times,R,v1)-bsxfun(@times,M2,v2);
% cross product: Q x rQ 
QxrQ=[Q(2)*rQ(3)-Q(3)*rQ(2) Q(3)*rQ(1)-Q(1)*rQ(3) Q(1)*rQ(2)-Q(2)*rQ(1)];
% final formula, with temporary matrices
M1=bsxfun(@times,repmat(QxrQ,N,1),F);
M2=bsxfun(@times,gF,R*QxrQ');
B=1e-7*bsxfun(@times,M1-M2,F.^-2);