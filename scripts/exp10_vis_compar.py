"""
Summarize/visualize optimization results pickled in multiple files
"""
#%% Inits
import pickle
import numpy as np
import matplotlib.pyplot as plt

CHIN_STRAP_ANGLES = (0, np.pi/16, np.pi/8, 3*np.pi/16, np.pi/4)

OPT_FNAME_TMPL = '/home/andrey/storage/Data/MEGSim/2020-11-19_iterative/iter_opt_chin_%0.3frad.pkl'
RAND_FNAME_TMPL = '/home/andrey/storage/Data/MEGSim/2020-11-19_iterative/iter_opt_rand_chin_%0.3frad.pkl'
UNIFORM_FNAME_TMPL = '/home/andrey/storage/Data/MEGSim/2020-11-19_iterative/uniform_chin_%0.3frad.pkl'

MAX_Y_SCALE = 14

#%% Load the data
dt_opt = []
dt_rand = []
dt_uni = []

for chin_strap_angle in CHIN_STRAP_ANGLES:
    fl = open(OPT_FNAME_TMPL % chin_strap_angle, 'rb')
    dt_opt.append(pickle.load(fl))
    fl.close()

    fl = open(RAND_FNAME_TMPL % chin_strap_angle, 'rb')
    dt_rand.append(pickle.load(fl))
    fl.close()

    fl = open(UNIFORM_FNAME_TMPL % chin_strap_angle, 'rb')
    dt_uni.append(pickle.load(fl))
    fl.close()

#%% Plot all by chin strap angle
for i in range(len(CHIN_STRAP_ANGLES)):
    plt.figure()
    plt.plot(np.log10(dt_opt[i]['best_cond_nums']))
    plt.plot(np.log10(dt_rand[i]['best_cond_nums']), ':')
    plt.plot(np.log10(dt_uni[i]['best_cond_nums']), '-.')
    plt.ylim((0, MAX_Y_SCALE))
    plt.xlabel('Number of sensors')
    plt.ylabel('Log10 of condition number')
    plt.title('Chin strap is %0.3f steradian' % CHIN_STRAP_ANGLES[i])
    plt.legend(('Optimal', 'Random', 'Uniform'))

#%% All optimal in one plot
plt.figure()
legend = []
for i in range(len(CHIN_STRAP_ANGLES)):
    plt.plot(np.log10(dt_opt[i]['best_cond_nums']))
    legend.append('Chin strap is %0.3f steradian' % CHIN_STRAP_ANGLES[i])
        
plt.ylim((0, MAX_Y_SCALE))
plt.xlabel('Number of sensors')
plt.ylabel('Log10 of condition number')
plt.legend(legend)
plt.title('Optimal')

#%% All random in one plot
plt.figure()
legend = []
for i in range(len(CHIN_STRAP_ANGLES)):
    plt.plot(np.log10(dt_rand[i]['best_cond_nums']), ':')
    legend.append('Chin strap is %0.3f steradian' % CHIN_STRAP_ANGLES[i])
        
plt.ylim((0, MAX_Y_SCALE))
plt.xlabel('Number of sensors')
plt.ylabel('Log10 of condition number')
plt.legend(legend)
plt.title('Random')

#%% All uniform in one plot
plt.figure()
legend = []
for i in range(len(CHIN_STRAP_ANGLES)):
    plt.plot(np.log10(dt_uni[i]['best_cond_nums']), '-.')
    legend.append('Chin strap is %0.3f steradian' % CHIN_STRAP_ANGLES[i])
        
plt.ylim((0, MAX_Y_SCALE))
plt.xlabel('Number of sensors')
plt.ylabel('Log10 of condition number')
plt.legend(legend)
plt.title('Uniform')


plt.show()

