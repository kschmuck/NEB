import numpy as np
import NEB as neb
import xyz_file_writer as xyz_writer
import IDPP as idpp
from copy import deepcopy
from pes import energy_gradient
from data_reader_writer import Reader, Writer

# Ethane
# opt minima --> with permutation
minima_a = np.array([
    -1.15706, -1.01589,  0.06129, # N
    -1.15706,  0.45486, -0.91044, # F
    -1.15706,  0.56102,  0.84914, # O
    -0.76316, -0.00000, -0.00000, # C
     0.76316,  0.00000,  0.00000, # C
     1.15706, -0.45486,  0.91043, # H
     1.15707, -0.56102, -0.84914, # H
     1.15706,  1.01589, -0.06129]) # H


minima_b = np.array([
    -1.15706,  0.45486, -0.91044, # F
    -1.15706,  0.56102,  0.84914, # O
    -1.15706, -1.01589,  0.06129, # N
    -0.76316, -0.00000, -0.00000, # C
     0.76316,  0.00000,  0.00000, # C
     1.15706, -0.45486,  0.91043, # H
     1.15707, -0.56102, -0.84914, # H
     1.15706,  1.01589, -0.06129]) # H

atom_list = ['H', 'H', 'H', 'C', 'C', 'H', 'H', 'H']

# Ammonia
minima_a = np.array([-0.14924, -0.18579,  0.12047,
                      0.15186,  0.50780,  0.77412,
                      0.66350, -0.59496, -0.29304,
                     -0.66612,  0.27295, -0.60155])

minima_b = np.array([0.14924,   0.18579, -0.12047,
                     0.15186,   0.50780,  0.77412,
                     0.66350,  -0.59496, -0.29304,
                    -0.66612,   0.27295, -0.60155])
atom_list = ['N', 'H', 'H', 'H']
# chemical accuracy 0.043 eV
number_of_images = 7
k = 10**-6# 10**-10 #10**-6
delta_t_fire = 3.5 #3.5
delta_t_verlete = 0.8 #0.2
force_max = 0.00001
max_steps = 800
epsilon = 0.01 #00001
trust_radius = 0.2 #.1
# # --------------------

import time
t = time.clock()

images = neb.create_images(minima_a, minima_b, number_of_images)
images = neb.ImageSet(images, atom_list=atom_list)
images.set_spring_constant(k)

opt_steepest_decent = neb.Optimizer.SteepestDecentNeb(epsilon, trust_radius)
opt_fire = neb.Optimizer.FireNeb(delta_t_fire, 2*delta_t_fire, trust_radius)
opt_verlete = neb.Optimizer.VerleteNeb(delta_t_verlete, trust_radius)
# opt_cg = neb.Optimizer.ConjugateGradientNeb(trust_radius, gamma=0.05, n_back=50)

idpp_potential = idpp.IDPP(images)
images.energy_gradient_func = idpp_potential.energy_gradient_idpp_fucntion
opt = neb.Optimizer()

name_list = ['steepest', 'verlete', 'fire', 'cg']
opt_list = [opt_steepest_decent, opt_verlete, opt_fire]#, opt_cg]

ii = 0
for element in opt_list:
    imgs = deepcopy(images)
    imgs = opt.run_opt(imgs, element, max_steps=max_steps, force_max=force_max, opt_minima=False, rm_rot_trans=True, freezing=0, tangent_method='improved')
    xyz_writer.write_images2File(imgs.get_positions(), name_list[ii] + '_FinishedIDDP.xyz', atom_list)
    ii += 1