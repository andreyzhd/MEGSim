"""
Iteratively optimize sensor array locations -- visualize the results
"""

#%% Inits
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab

DATA_FNAME = '/home/andrey/scratch/iter_opt_chin_0.000rad.pkl'

#%% Read the data
fl = open(DATA_FNAME, 'rb')
dt = pickle.load(fl)
fl.close()

x_helm = dt['x_sphere'][dt['helm_indx']]
y_helm = dt['y_sphere'][dt['helm_indx']]
z_helm = dt['z_sphere'][dt['helm_indx']]
sens_indx = dt['sens_indx']
best_cond_nums = dt['best_cond_nums']
l = dt['L']
is_rand_control = dt['RANDOM_CONTROL']

#%% Print some info
print('The results are %s' % ('real', 'fake (randomly scramled control)')[is_rand_control])
print('%i sensors placed over %i possible locations (location density is %i sensors / 4*pi steradians), L = %i' % (len(sens_indx), len(x_helm), dt['x_sphere'].shape[0], l))
print('The final condition number is %f' % best_cond_nums[-1])

#%% Plot the helmet
fig1 = mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
mlab.clf(fig1)
mlab.points3d(x_helm, y_helm, z_helm, scale_factor=0.02, color=(0, 1, 0))
mlab.points3d(x_helm[sens_indx], y_helm[sens_indx], z_helm[sens_indx], scale_factor=0.03, color=(0, 0, 0))

#%% Look at the condition numbers
plt.plot(np.log10(best_cond_nums))
plt.xlabel('Number of sensors')
plt.ylabel('Log10 of condition number')
plt.show()
