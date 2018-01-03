import numpy as np
import NEB as neb
import xyz_file_writer as xyz_writer
import optimize as opt
from data_reader_writer import Reader, Writer


# opt minima --> with perumtation
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

# HCN --> initial guess with idpp would fail because no favoured direction
# minima_a = np.array([ 9.32095277e-01, -9.01031444e-05, -2.02342355e-04,
#                        -1.68360028e-04,  1.38338646e-04,  7.68642769e-05,
# 			           -1.08299466e+00,  2.29788120e-04,  1.88635856e-04])
# minima_b = np.array([-2.11111682e+00, -6.03169809e-05, -1.35452321e-04,
# 			            2.77792047e-04, -3.05541286e-04,  9.26068620e-05,
# 		               -1.05555841e+00,  5.14545986e-05,  1.15738781e-06])
# atom_list = ['H','C','N']

# Gold55Nickel1
# minima_a = np.array([0.022735	,	0.022735	,	0.0525357	,
# 1.04132	,	1.04132	,	2.38293	,
# 2.68926	,	0.652021	,	0.0519511	,
# 1.67078	,	-1.62507	,	1.4927	,
# 0.652021	,	2.68926	,	0.0519511	,
# -1.62507	,	1.67078	,	1.4927	,
# -0.995357	,	-0.995357	,	2.38333	,
# -2.68515	,	-0.630159	,	0.0114595	,
# -0.630159	,	-2.68515	,	0.0114595	,
# 1.66753	,	-1.65752	,	-1.44169	,
# 1.03249	,	1.03249	,	-2.33993	,
# -1.65752	,	1.66753	,	-1.44169	,
# -1.02282	,	-1.02282	,	-2.34075	,
# 2.01309	,	2.01309	,	4.60725	,
# 3.74666	,	1.68947	,	2.41218	,
# 2.71791	,	-0.610269	,	3.86717	,
# 5.33925	,	1.26271	,	0.00852748	,
# 4.5109	,	-1.127	,	1.40846	,
# 3.30091	,	-3.29401	,	2.89065	,
# 1.68947	,	3.74666	,	2.41218	,
# 3.43416	,	3.43416	,	-0.114476	,
# 1.26271	,	5.33925	,	0.00852748	,
# -0.610269	,	2.71791	,	3.86717	,
# -1.127	,	4.5109	,	1.40846	,
# -3.29401	,	3.30091	,	2.89065	,
# 0.0257636	,	0.0257636	,	4.76623	,
# -2.8687	,	0.615761	,	3.8724	,
# -2.03404	,	-2.03404	,	4.6724	,
# 0.615761	,	-2.8687	,	3.8724	,
# -3.82498	,	-1.72126	,	2.41992	,
# -1.72126	,	-3.82498	,	2.41992	,
# -5.33227	,	-1.26141	,	-0.00732297	,
# -3.42447	,	-3.42447	,	0.0674543	,
# -1.26141	,	-5.33227	,	-0.00732297	,
# 1.06512	,	-4.48306	,	1.48923	,
# 3.417	,	-3.43113	,	0.00163691	,
# 1.09677	,	-4.49148	,	-1.442	,
# 3.29033	,	-3.2967	,	-2.88638	,
# 4.48148	,	-1.0512	,	-1.50396	,
# 3.83151	,	1.70188	,	-2.42312	,
# 2.82286	,	-0.63034	,	-3.88497	,
# 2.03283	,	2.03283	,	-4.66552	,
# 1.70188	,	3.83151	,	-2.42312	,
# -1.0512	,	4.48148	,	-1.50396	,
# -0.63034	,	2.82286	,	-3.88497	,
# -3.2967	,	3.29033	,	-2.88638	,
# -3.43113	,	3.417	,	0.00163691	,
# -4.48306	,	1.06512	,	1.48923	,
# -4.49148	,	1.09677	,	-1.442	,
# -2.03279	,	-2.03279	,	-4.65202	,
# -0.000633005	,	-0.000633005	,	-4.83421	,
# 0.652	,	-2.76411	,	-3.9115	,
# -1.70821	,	-3.82033	,	-2.41795	,
# -3.82033	,	-1.70821	,	-2.41795	,
# -2.76411	,	0.652	,	-3.9115
# ])
#
# minima_b = np.array([
# -0.0179414	,	-0.0179414	,	-0.0410077	,
# 1.92376	,	1.92376	,	4.40152	,
# 2.69691	,	0.629162	,	-0.022331	,
# 1.66308	,	-1.68273	,	1.43971	,
# 0.629162	,	2.69691	,	-0.022331	,
# -1.68273	,	1.66308	,	1.43971	,
# -1.04371	,	-1.04371	,	2.34341	,
# -2.7199	,	-0.647188	,	-0.0150668	,
# -0.647188	,	-2.7199	,	-0.0150668	,
# 1.67018	,	-1.68355	,	-1.48084	,
# 1.02963	,	1.02963	,	-2.38681	,
# -1.68355	,	1.67018	,	-1.48084	,
# -1.0424	,	-1.0424	,	-2.38541	,
# 1.00565	,	1.00565	,	2.30119	,
# 3.77744	,	1.71858	,	2.47612	,
# 2.74811	,	-0.583422	,	3.93178	,
# 5.33793	,	1.27672	,	0.0498304	,
# 4.49487	,	-1.10898	,	1.42507	,
# 3.30729	,	-3.26386	,	2.92164	,
# 1.71858	,	3.77744	,	2.47612	,
# 3.42459	,	3.42459	,	-0.0883888	,
# 1.27672	,	5.33793	,	0.0498304	,
# -0.583422	,	2.74811	,	3.93178	,
# -1.10898	,	4.49487	,	1.42507	,
# -3.26386	,	3.30729	,	2.92164	,
# 0.0527798	,	0.0527798	,	4.83163	,
# -2.84059	,	0.622738	,	3.87386	,
# -2.00859	,	-2.00859	,	4.69661	,
# 0.622738	,	-2.84059	,	3.87386	,
# -3.84254	,	-1.69833	,	2.445	,
# -1.69833	,	-3.84254	,	2.445	,
# -5.35256	,	-1.27243	,	-0.026385	,
# -3.44351	,	-3.44351	,	0.184526	,
# -1.27243	,	-5.35256	,	-0.026385	,
# 1.05598	,	-4.4929	,	1.52532	,
# 3.45339	,	-3.42074	,	0.00916205	,
# 1.1695	,	-4.5325	,	-1.35561	,
# 3.28922	,	-3.31254	,	-2.9114	,
# 4.50533	,	-1.06817	,	-1.47879	,
# 3.8425	,	1.73854	,	-2.41586	,
# 2.93155	,	-0.592649	,	-3.84779	,
# 2.02831	,	2.02831	,	-4.69498	,
# 1.73854	,	3.8425	,	-2.41586	,
# -1.06817	,	4.50533	,	-1.47879	,
# -0.592649	,	2.93155	,	-3.84779	,
# -3.31254	,	3.28922	,	-2.9114	,
# -3.42074	,	3.45339	,	0.00916205	,
# -4.4929	,	1.05598	,	1.52532	,
# -4.5325	,	1.1695	,	-1.35561	,
# -2.04417	,	-2.04417	,	-4.67793	,
# -0.00323273	,	-0.00323273	,	-4.85385	,
# 0.651372	,	-2.77535	,	-3.92817	,
# -1.71631	,	-3.83433	,	-2.43051	,
# -3.83433	,	-1.71631	,	-2.43051	,
# -2.77535	,	0.651372	,	-3.92817
# ])
# atom_list = []
# for ii in range(0, 55):
#     if ii == 1:
#         atom_list.append('Ni')
#     else:
#         atom_list.append('Au')

#
#
# chemical accuracy 0.043 eV
number_of_images = 1
k = 10**-6# 10**-10 #10**-6
delta_t_fire = 3.5 #3.5
delta_t_verlete = 0.4 #0.2
force_max = 0.00001
max_steps = 800
epsilon = 0.01 #00001
trust_radius = 0.1 #.1
# # --------------------

import time
t = time.clock()

images = neb.create_images(minima_a, minima_b, number_of_images)
images = neb.Images(images, atom_list=atom_list)
images.set_spring_constant(k)

opt_steepest_decent = neb.Optimizer.SteepestDecentNeb(epsilon, trust_radius)
opt_fire = neb.Optimizer.FireNeb(delta_t_fire, 2*delta_t_fire, trust_radius)
opt_verlete = neb.Optimizer.VerleteNeb(delta_t_verlete, trust_radius)

opt_cg = neb.Optimizer.ConjugateGradientNeb(epsilon, trust_radius)
opt_bfgs = neb.Optimizer.BFGSNeb(trust_radius)

idpp = neb.IDPP(images.get_images())
images.set_energy_gradient_func(idpp.energy_gradient_idpp_fucntion)
xyz_writer.write_images2File(images.get_positions(), 'InitialLineGuess.xyz', atom_list)
opt = neb.Optimizer()

test = opt.run_opt(images, opt_fire, idpp.energy_gradient_idpp_fucntion, max_steps=max_steps, force_max=force_max, rm_rot_trans=False, idpp=True)
xyz_writer.write_images2File(test.get_positions(), 'InitialIdppGuess.xyz', atom_list)


k = 10**-1 # 10**-10 #10**-6 -4 -3
delta_t_fire = 3.5 #3.5
delta_t_verlete = 0.01 #0.2
force_max = 0.01#0.001
max_steps = 50
epsilon = 0.01 #00001
trust_radius = 0.005

opt_steepest_decent = neb.Optimizer.SteepestDecentNeb(epsilon, trust_radius)
opt_fire = neb.Optimizer.FireNeb(delta_t_fire, 2*delta_t_fire, trust_radius)
opt_verlete = neb.Optimizer.VerleteNeb(delta_t_verlete, trust_radius)

# filename = 'Ethane_13.xyz'
# read_er = Reader()
# read_er.read(filename)
# geom = read_er.geometries
# atom_list = read_er.atom_list
# images = neb.set_images(geom)
# images = neb.Images(images, atom_list=read_er.atom_list)

opt = neb.Optimizer()
test = opt.run_opt(images, opt_verlete, neb.energy_gradient_surface, max_steps=max_steps, force_max=force_max, opt_minima=False, rm_rot_trans=False)
xyz_writer.write_images2File(test.get_positions(), 'FinishedPath.xyz', atom_list)
