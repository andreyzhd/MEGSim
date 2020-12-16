import numpy as np
from mayavi import mlab

from mne.preprocessing.maxwell import _sss_basis
from megsimutils.utils import spherepts_golden, _prep_mf_coils_pointlike, _deg_ord_idx
from megsimutils.viz import viz_field

N_POINTS = 10000
LIN = 16
LOUT = 0

DEG = 13
ORDER = 9
CMAP = 'cool'

# Generate the surface and the normals
rhelm = spherepts_golden(N_POINTS, hcylind=1)
npoints = rhelm.shape[0]
nhelm = rhelm.copy()
nhelm[rhelm[:,2]<0, 2] = 0

# Compute VSH's
sss_params = {'origin': np.zeros(3), 'int_order': LIN, 'ext_order': LOUT}
allcoils = _prep_mf_coils_pointlike(np.tile(rhelm, (3,1)), np.repeat(np.eye(3), npoints, axis=0))
S = _sss_basis(sss_params, allcoils)
vsh = np.stack((S[:npoints,:], S[npoints:2*npoints,:], S[2*npoints:]), axis=2) # npoints-by-ncomp-by-3

# Plot
vmax = np.max(np.linalg.norm(vsh[:,_deg_ord_idx(DEG, ORDER),:], axis=1))

fig0 = mlab.figure()
viz_field(rhelm, vsh[:,_deg_ord_idx(DEG, ORDER),:], nhelm, figure=fig0, colormap=CMAP, cmap_range=(-vmax, vmax), show_arrows=True, opacity=0.7, inner_surf=0.3)

fig1 = mlab.figure()
viz_field(rhelm, vsh[:,_deg_ord_idx(DEG, ORDER),:], nhelm, figure=fig1, colormap=CMAP, cmap_range=(-vmax, vmax), proj_field=False)

mlab.sync_camera(fig0, fig1)

mlab.show()