import numpy as np
import os, sys
import ethane_PES, ammonia_PES, Pt4C3H8_PES #, AuO2_PES
import NEB, optimize, data_reader_writer, Kernels, MLDerivative, copy, IDPP
import time
import warnings


t = time.clock()
#
molecule = 'Ammonia'
file_path = os.path.dirname(__file__)
result_path = os.path.join(file_path, 'Results', molecule)
sys.stdout.flush()

# NEB parameters
n_imgs = 7
trust_radius = 0.05
kernel = Kernels.RBF([0.7])
k = 1e-4
max_force = 0.001
max_iter = 1000
C1 = 1e10
C2 = 1e10
max_steps = 21
# neb_method = 'simple_improved'
neb_method = 'improved'

# optimizer
delta_t = 3.5

opt_fire = NEB.Fire(delta_t, 2*delta_t, trust_radius)
opt = NEB.Optimizer()

eps = 1e-5
restarts = 5
opt_steps = 1
optimize_parameters = True
norm_y = False
# ml_method = MLDerivative.IRWLS(kernel, C1=C1, C2=C2, epsilon=5*1e-5, epsilon_prime=5*1e-5, max_iter=1e4)
# ml_method = MLDerivative.RLS(kernel,C1=C1, C2=C2)
ml_method = MLDerivative.GPR(kernel, opt_restarts=restarts, opt_parameter=optimize_parameters, noise_value = 1./C1,
                             noise_derivative=1./C2,  normalize_y=norm_y)


# reading minima and creating imagesAu_O2
reader = data_reader_writer.Reader()
reader.read_new(os.path.join(result_path, 'minima.xyz'))
# reader.read_new(os.path.join(file_path, molecule+'_minima.xyz'))Au_O2Au_O2
imgs = reader.images
atoms = imgs[0]['atoms']
minima_a = np.array(imgs[0]['geometry']).flatten()
minima_b = np.array(imgs[1]['geometry']).flatten()

images = NEB.create_images(minima_a, minima_b, n_imgs)
images = NEB.ImageSet(images, atoms)
writer = data_reader_writer.Writer()

# energy and gradient evaluation
if molecule == 'Ammonia':
    energy_gradient = ammonia_PES.energy_and_gradient
elif molecule == 'Ethane':
    energy_gradient = ethane_PES.energy_and_gradient
elif molecule == 'Pt4C3H8':
    energy_gradient = Pt4C3H8_PES.energy_and_gradient
elif molecule == 'Au_O2':

    k_idpp = 10 ** -4  # 10**-10 #10**-6
    delta_t_fire = 1.5  # 3.5
    delta_t_verlete = 0.4  # 0.2
    force_max = 0.001
    max_steps = 500
    epsilon = 0.01  # 00001
    trust_radius = 0.05  # .1
    # # --------------------

    images.set_spring_constant(k_idpp)
    opt_fire = NEB.Optimizer.FireNeb(delta_t_fire, 2*delta_t_fire, trust_radius)

    idpp_potential = IDPP.IDPP(images)
    images.energy_gradient_func = idpp_potential.energy_gradient_idpp_function
    opt = NEB.Optimizer()
    writer.write(os.path.join(result_path, 'ML_Data', 'start_structure.xyz'), images.get_positions(),
                 atoms)
    images = opt.run_opt(images, opt_fire, max_steps=max_steps, force_max=force_max, rm_rot_trans=False)
    writer.write(os.path.join(result_path, 'ML_Data', molecule + 'idpp_structure.xyz'), images.get_positions(),
                 atoms)
    energy_gradient = AuO2_PES.energy_and_gradient
#

images.set_spring_constant(k)
images.energy_gradient_func = energy_gradient
images.update_images(neb_method)

pos = images.get_image_position_2D_array()
grad = images.get_image_gradient_2Darray()
energy = images.get_image_energy_list()

nan_flag = False
idx = []
for uu in range(n_imgs):
    force_norm = images[uu].force_norm()
    if np.isnan(force_norm):
        idx.append(uu)
        nan_flag = True
if nan_flag:
    pos = np.delete(pos, np.array(idx) - 1, axis=0)
    energy = np.delete(energy, np.array(idx) - 1, axis=0)
    grad = np.delete(grad, np.array(idx) - 1, axis=0)
    if pos.size[0] == 0:
        raise ValueError('Can not fit any curve all values are NaN, please have a look at the ab initio method')

train_list = [list([pos]), list([energy]), list([grad])]

writer.write(os.path.join(result_path, 'ML_Data', 'start_structure.xyz'), images.get_positions(),
             atoms)
converged = False
step = 0
ml_method_copy = copy.deepcopy(ml_method)
restarts = 5
# trust_radius = trust_radius*0.5
optimize_parameters = True
print('#####################################\n #### '+ molecule +' '+ml_method.__class__.__name__ +' '+ neb_method+' #### \n #####################################')
print(kernel)
print('-- Parameters --')
print("k = %f \n opt_method = Fire \n delta_t_fire = %f \n force_max = %f \n max_steps = %d \n trust_radius = %f \n tangent_method = %s \n C1 = %f \n C2 = %f \n max_iter = %d \n eps = %f \n restarts = %d") %(k, delta_t, max_force, max_steps, trust_radius, neb_method, C1, C2, max_iter, eps, restarts)
print('-----------------')
while not converged:
    print('Step = ' + str(step))
    calc_images = copy.deepcopy(images)
    # ml_method_copy = copy.deepcopy(ml_method)
    x_train = np.array(np.concatenate([train_list[0][ii] for ii in range(step + 1)]))
    y_train = np.array(np.concatenate([train_list[1][ii] for ii in range(step + 1)]))
    y_prime_train = np.array(np.concatenate([train_list[2][ii] for ii in range(step + 1)]))

    print('start fitting')
    if ml_method.__class__.__name__ == 'GPR':
        if step < opt_steps:
            ml_method_copy.opt_restarts = 5
            ml_method_copy.opt_parameter = True
        else:
            ml_method_copy.opt_parameter = False
            ml_method_copy.opt_restarts = 0
        ml_method_copy.fit(x_train, y_train, x_prime_train=x_train, y_prime_train=y_prime_train)
        print('length_scale = ' + str(ml_method_copy.kernel))
    elif ml_method.__class__.__name__ == 'RLS':
        ml_method_copy.fit(x_train, y_train, x_prime_train=x_train, y_prime_train=y_prime_train)
    elif ml_method.__class__.__name__ == 'IRWLS':
        ml_method_copy.fit(x_train, y_train, x_prime_train=x_train, y_prime_train=y_prime_train,
                           eps=eps)
        num_beta = sum(len(ii) for ii in ml_method_copy._support_index_beta)
        num_deriv = sum(len(ii) for ii in y_prime_train)
        print('function values:  %d number of support vectors, %d number of non support vectors') % (
        len(ml_method_copy._support_index_alpha), len(y_train) - len(ml_method_copy._support_index_alpha))
        print('derivatives values: %d number of support vectors, %d number of non support vectors') % (num_beta,
                                                                                                       num_deriv - num_beta)

    print('start neb')
    calc_images.energy_gradient_func = ml_method_copy.predict_val_der
    # force_max = 1e-3
    calc_images = opt.run_opt(calc_images, copy.deepcopy(opt_fire), force_max=max_force, max_steps=max_iter, freezing=0, tangent_method=neb_method, opt_minima=False)
    writer.write(os.path.join(result_path, 'ML_Data', 'Step_'+str(step)+ '_structure.xyz'), calc_images.get_positions(), atoms)
    print('---- predicted forces ----')
    uu = 0
    for element in calc_images:
        print(str(element.force_norm())+' of image ' + str(uu))
        uu += 1
    calc_images.energy_gradient_func = energy_gradient
    calc_images.update_images(neb_method)

    print('---- true forces ----')
    idx = []
    nan_flag = False
    for uu in range(len(calc_images)):
        force_norm = calc_images[uu].force_norm()
        print(str(force_norm) + ' of image ' + str(uu))
        if np.isnan(force_norm):
            idx.append(uu)
            nan_flag = True

    if ml_method.__class__.__name__ == 'IRWLS':
        pos = calc_images.get_image_position_2D_array()
        grad = calc_images.get_image_gradient_2Darray()
        energy = calc_images.get_image_energy_list()
    else:
        pos = calc_images.get_image_position_2D_array()[1:-1, :]
        grad = calc_images.get_image_gradient_2Darray()[1:-1]
        energy = calc_images.get_image_energy_list()[1:-1]

    # ind = opt.get_max_force(calc_images[1:-1]) - 1
    if not nan_flag:
        opt.fmax = max_force
        if opt.is_converged(calc_images[1:-1]):
            print('solution found')
            writer.write(os.path.join(result_path, 'ML_Data', 'end_structure.xyz'), calc_images.get_positions(), atoms)
            converged = True
    else:
        # pos = calc_images.get_image_position_2D_array()
        # grad = calc_images.get_image_gradient_2Darray()[1:-1]
        # # grad = np.array(grad)
        # energy = calc_images.get_image_energy_list()[1:-1]
        pos = np.delete(pos, np.array(idx)-1, axis=0)
        energy = np.delete(energy, np.array(idx)-1, axis=0)
        grad = np.delete(grad, np.array(idx)-1, axis=0)
    train_list[0].append(pos)
    train_list[1].append(energy)
    train_list[2].append(grad)
    step += 1
    if step >= max_steps:
        converged = True
        print('no solution obtained')

max_steps = 31
old_step = step
step = 0
calc_images.set_climbing_image()
images = copy.deepcopy(calc_images)
max_force *= 0.5
# trust_radius = trust_radius*.5
converged = False
print('#####################################\n #### '+ molecule +' '+ml_method.__class__.__name__ +' _climbing image #### \n #####################################')
print(kernel)
print('-- Parameters --')
print("k = %f \n opt_method = Fire \n delta_t_fire = %f \n force_max = %f \n max_steps = %d \n trust_radius = %f \n tangent_method = %s \n C1 = %f \n C2 = %f \n max_iter = %d \n eps = %f \n restarts = %d") %(k, delta_t, max_force, max_steps, trust_radius, neb_method, C1, C2, max_iter, eps, restarts)
print('-----------------')
while not converged:
    print('Step = ' + str(step))
    calc_images = copy.deepcopy(images) # reset to the obtained path of the nudged elastic band without climbing.
    ml_method_copy = copy.deepcopy(ml_method)
    x_train = np.array(np.concatenate([train_list[0][ii] for ii in range(step + 1+old_step)]))
    y_train = np.array(np.concatenate([train_list[1][ii] for ii in range(step + 1+old_step)]))
    y_prime_train = np.array(np.concatenate([train_list[2][ii] for ii in range(step + 1+old_step)]))

    print('start fitting')
    if ml_method.__class__.__name__ == 'GPR':
        if step < 1:
            ml_method_copy.opt_restarts = 5
            ml_method_copy.opt_parameter = True
        else:
            ml_method_copy.opt_parameter = False
            ml_method_copy.opt_restarts = 0
        ml_method_copy.fit(x_train, y_train, x_prime_train=x_train, y_prime_train=y_prime_train)
        print('length_scale = ' + str(ml_method_copy.kernel))
    elif ml_method.__class__.__name__ == 'RLS':
        ml_method_copy.fit(x_train, y_train, x_prime_train=x_train, y_prime_train=y_prime_train)
    elif ml_method.__class__.__name__ == 'IRWLS':
        ml_method_copy.fit(x_train, y_train, x_prime_train=x_train, y_prime_train=y_prime_train,
                           eps=eps)
        num_beta = sum(len(ii) for ii in ml_method_copy._support_index_beta)
        num_deriv = sum(len(ii) for ii in y_prime_train)
        print('function values:  %d number of support vectors, %d number of non support vectors') % (
        len(ml_method_copy._support_index_alpha), len(y_train) - len(ml_method_copy._support_index_alpha))
        print('derivatives values: %d number of support vectors, %d number of non support vectors') % (num_beta,
                                                                                                       num_deriv - num_beta)
    print('start neb')
    calc_images.energy_gradient_func = ml_method_copy.predict_val_der
    opt.run_opt(calc_images, copy.deepcopy(opt_fire), force_max=max_force, max_steps=max_iter, freezing=0, tangent_method=neb_method)

    writer.write(os.path.join(result_path, 'ML_Data', 'Climbing_Step_' + str(step) + '_structure.xyz'),
                 calc_images.get_positions(), atoms)
    print('---- predicted forces ----')
    uu = 0
    for element in calc_images:
        print(str(element.force_norm())+ ' of image ' + str(uu))
        uu += 1
    calc_images.energy_gradient_func = energy_gradient
    calc_images.update_images(neb_method)
    opt.fmax = max_force
    print('---- true forces ----')
    uu = 0
    for element in calc_images:
        print(str(element.force_norm())+ ' of image ' + str(uu))
        if np.isnan(force_norm):
            idx.append(uu)
            nan_flag = True
        uu += 1
    pos = calc_images.get_image_position_2D_array()[1:-1, :]
    grad = calc_images.get_image_gradient_2Darray()[1:-1]
    # grad = np.array(grad)
    energy = calc_images.get_image_energy_list()[1:-1]

    # ind = opt.get_max_force(calc_images[1:-1]) - 1
    if not nan_flag:
        if opt.is_converged(calc_images[1:-1]):
            print('solution found')
            # writer = data_reader_writer.Writer()
            writer.write(os.path.join(result_path, 'ML_Data', 'end_structure.xyz'), calc_images.get_positions(), atoms)
            converged = True
        train_list[0].append(pos)
        train_list[1].append(energy)
        train_list[2].append(grad)
    else:
        pos = np.delete(pos, idx, axis=0)
        energy = np.delete(energy, idx, axis=0)
        grad = np.delete(grad, idx, axis=0)
        train_list[0].append(pos)
        train_list[1].append(energy)
        train_list[2].append(grad)
    step += 1
    if step >= max_steps:
        converged = True
        print('no solution obtained')
writer.write(os.path.join(result_path, 'ML_Data', 'end_structure.xyz'), calc_images.get_positions(), atoms)
print('used time ' + str(t-time.clock()))
