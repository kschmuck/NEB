import numpy as np
import NEB
import IDPP as idpp
import ammonia_PES, ethane_PES, Pt4C3H8_PES, Pt4C1H4_PES
# from pes import energy_gradient
import data_reader_writer
import copy, os, time, sys

# molecule = 'Ethane'
#molecule = 'Ammonia'
# molecule = 'Pt4C3H8'
molecule = 'Pt4C1H4'

n_imgs = 7

k = 1e-3 # 10**-10 #10**-6
delta_t_fire = 3.5 #3.5
force_max = 2e-2 # 0.001 Ammonia 0.002 Ethane 0.02 Pt4C1H4
max_steps = 500
trust_radius = 0.05 #.1
# neb_method = 'simple_improved'
neb_method = 'improved'

max_displacement = None

calc_idpp = True
print_step = True # True
write_geom = True # True

file_path = os.path.dirname(__file__)
result_path = os.path.join(file_path, 'Results', molecule)
reader = data_reader_writer.Reader()
reader.read_new(os.path.join(result_path, 'minima.xyz'))
imgs = reader.images
atoms = imgs[0]['atoms']
minima_a = np.array(imgs[0]['geometry']).flatten()
minima_b = np.array(imgs[1]['geometry']).flatten()
# print(minima_a)

if molecule == 'Ammonia':
	energy_gradient = ammonia_PES.energy_and_gradient
elif molecule == 'Ethane':
	energy_gradient = ethane_PES.energy_and_gradient
elif molecule == 'Pt4C3H8':
	energy_gradient = Pt4C3H8_PES.energy_and_gradient
elif molecule == 'Pt4C1H4':
	energy_gradient = Pt4C1H4_PES.energy_and_gradient

images = NEB.create_images(minima_a, minima_b, n_imgs)
images = NEB.ImageSet(images, atoms)
writer = data_reader_writer.Writer()

writer.write(os.path.join(result_path, molecule + '_' + neb_method + '_initial_Path.xyz'), images.get_positions(), atoms)
if calc_idpp :
	k_idpp = 1e-4
	delta_t_fire_idpp= 3.5
	force_max_idpp = force_max*10
	max_steps_idpp = 1000
	trust_radius_idpp = 0.05
	neb_method_idpp = neb_method

	print("molecule idpp = " + molecule + "\n k = %f \n opt_method = Fire \n delta_t_fire = %f \n force_max = %f \n"
										  " max_steps = %d \n trust_radius = %f \n tangent_method = %s") \
		 % (k_idpp, delta_t_fire_idpp, force_max_idpp, max_steps_idpp, trust_radius_idpp, neb_method_idpp)
	images.set_spring_constant(k_idpp)
	opt_fire = NEB.Optimizer.FireNeb(delta_t_fire_idpp, 2 * delta_t_fire_idpp, trust_radius_idpp)
	idpp_potential = idpp.IDPP(images)
	images.energy_gradient_func = idpp_potential.energy_gradient_idpp_function
	opt = NEB.Optimizer()
	images = opt.run_opt(images, opt_fire, max_steps=max_steps_idpp, force_max=force_max_idpp, rm_rot_trans=False, freezing=3, opt_minima=False)
	writer.write(os.path.join(result_path, molecule + '_idpp_structure.xyz'), images.get_positions(), atoms)
	for img in images:
		img.frozen = False


sys.stdout.flush()
print("molecule = " + molecule + "\n k = %f \n opt_method = Fire \n delta_t_fire = %f \n force_max = %f \n max_steps = %d \n trust_radius = %f \n tangent_method = %s") %(k, delta_t_fire, force_max, max_steps, trust_radius, neb_method)

images.set_spring_constant(k)
images.energy_gradient_func = energy_gradient
t = time.clock()

opt_fire = NEB.Optimizer.FireNeb(delta_t_fire, 2*delta_t_fire, trust_radius)

opt = NEB.Optimizer()


images = opt.run_opt(images, opt_fire, max_steps=max_steps, force_max=force_max, opt_minima=False, rm_rot_trans=False,
					 tangent_method=neb_method, print_step=print_step, write_geom=write_geom)
writer.write(os.path.join(result_path, molecule + '_' + neb_method + '_finished_Path.xyz'), images.get_positions(), atoms, energy=images.get_image_energy_list())


images.set_climbing_image()
force_max = force_max * 0.5
trust_radius = 0.05
opt_fire = NEB.Optimizer.FireNeb(delta_t_fire, 2*delta_t_fire, trust_radius)

print("molecule = " + molecule + "\n k = %f \n opt_method = Fire \n delta_t_fire = %f \n force_max = %f \n max_steps = %d \n trust_radius = %f \n tangent_method = %s") %(k, delta_t_fire, force_max, max_steps, trust_radius, neb_method)

images = opt.run_opt(images, opt_fire, max_steps=max_steps, force_max=force_max, opt_minima=False, rm_rot_trans=False,
					 freezing=3, tangent_method=neb_method, print_step=print_step, write_geom=write_geom)
writer.write(os.path.join(result_path, molecule + '_' + neb_method + '_finished_Path_Climbing.xyz'), images.get_positions(), atoms, energy=images.get_image_energy_list())
print(time.clock()- t)
