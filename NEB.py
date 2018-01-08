# from pes import energy, gradient
from ethane_PES import energy_and_gradient
# from HNCPES import energy_and_gradient
# from Au55Ni import energy_and_gradient
import numpy as np
import copy
import xyz_file_writer as xyz_writer
# import sympy.geometry as sym_geom
# import sympy as sym
from optimize import SteepestDecent, Verlete, Fire, ConjuageGradient, BFGS
import time
from data_reader_writer import Reader, Writer


''' creation of the image positions --> class images '''
def create_images(product, reactant, number_of_images):
# cartesian coordinates!!
# check if product size == as reactant size
    if not (len(product) == len(reactant)):
        print('size of product is not equal to size of reactant')
    images = []
    img_positions = np.array([np.linspace(p, r, number_of_images + 2) for p, r in zip(product, reactant)])
    for element in img_positions.T:
        img = Image(element)
        images.append(img)
    return images


def set_images(positions):
    # positions list of np.arrays
    images = []
    for element in positions:
        pos = np.array([])
        for ele in element:
            pos = np.append(pos, ele)
        img = Image(pos)
        images.append(img)
    return images


def center_geometry(positions):
    # cartesian
    # atoms in row and coordiantes in columns x, y, z
    mean_value = np.zeros([len(positions), 3])
    for kk in range(0, len(positions)):
        pos = positions[kk, :, :]
        mean_value[kk, :] = np.array([np.mean(pos[:, 0]), np.mean(pos[:, 1]), np.mean(pos[:, 2])])
    return mean_value


def rotation_geometry(positions):
    # cartesian
    # atoms in row and coordiantes in columns x, y, z
    rotation_matrix_a = []
    center_geometry(positions)

    for ii in range(1, len(positions)):
        u, w, v = np.linalg.svd(np.dot(positions[ii, :, :].T, positions[ii-1, :, :]))
        a_j = np.dot(u, v)

        if np.linalg.det(a_j) < 0:
            u[:, -1] = -u[:, -1]
            a_j = np.dot(u, v)
        positions[ii, :, :] = np.dot(positions[ii, :, :], a_j)
        rotation_matrix_a.append(a_j)

    return positions, rotation_matrix_a


def get_tangent(img_0, img_1, img_2):
    # img_i is one image
    # img_0 = image before
    # img_1 = current image, at this image the tangent is set.
    # img_2 = image after
    # if img_i is None the tangent will be the spring to the next img
    if img_0 is None:
        tangent = img_1.position - img_2.position
        return tangent
    if img_2 is None:
        tangent = img_0.position - img_1.position
        return tangent

    def min_max(energy_0, energy_1, energy_2):
        a = abs(energy_2 - energy_1)
        b = abs(energy_0 - energy_1)
        return min([a, b]), max([a, b])

    if img_2.energy > img_1.energy > img_0.energy:
        tangent = img_2.position - img_1.position
    elif img_2.energy < img_1.energy < img_0.energy:
        tangent = img_1.position - img_0.position
    else:
        minimum, maximum = min_max(img_0.energy, img_1.energy, img_2.energy)
        t_p = (img_2.position - img_1.position) / np.linalg.norm((img_2.position - img_1.position))
        t_m = (img_1.position - img_0.position) / np.linalg.norm((img_1.position - img_0.position))
        if img_2.energy > img_1.energy:
            tangent = t_p * maximum + t_m * minimum
        else:
            tangent = t_p * minimum + t_m * maximum
    tangent = tangent / np.linalg.norm(tangent)
    return tangent


def spring_force(img_0, img_1, img_2):
    # img_i is one image
    # img_0 = image before
    # img_1 = current image.
    # img_2 = image after
    force = img_1.spring_constant * (img_2.position - img_1.position) \
            + img_1.spring_constant * (img_0.position - img_1.position)
    return force


def energy_gradient_surface(positions):
    # here place the fucntion for the evaluation of the gradient and energy
    # position = img.position
    energy_surface, gradient_surface = energy_and_gradient(positions)
    return energy_surface, gradient_surface

# def energy_gradient_surface(*args):
#     # here place the fucntion for the evaluation of the gradient and energy
#     # position = img.position
#     energy_surface, gradient_surface = energy_and_gradient('dummy')
#     return energy_surface, gradient_surface


class IDPP:
    def __init__(self, images, coordinate_system='cartesian'):
        if coordinate_system == 'cartesian':
            self.num_atoms = int(len(images[0].position)/3)
            self.number_of_coordinates = 3

        d_product = self.calc_distance(images[0].position)
        d_reactant = self.calc_distance(images[-1].position)

        for kk in range(0, len(images)):
            images[kk].d_ij_k = d_product + (1.0*kk)/(len(images)-1) * (d_reactant - d_product)

    def calc_distance(self, position):
        distance = np.zeros([self.num_atoms, self.num_atoms])

        def distance_function(atom_i, atom_j):
            return np.linalg.norm(atom_i-atom_j)

        for ii in range(0, self.num_atoms):
            index_ii = np.arange(ii*3, (ii+1)*3)
            for jj in range(ii+1, self.num_atoms):
                index_jj = np.arange(jj * 3, (jj + 1) * 3)
                distance[ii, jj] = distance_function(position[index_ii], position[index_jj])
        return distance

    def calc_forces_numerical(self, positions, d_ij_k):
        # weighting with 1/d**4
        def func_derivative(dx_i, dx_j, atom_i, atom_j, value):
            # dx_i derivative of the coordinate
            distance = np.linalg.norm(atom_i-atom_j)
            derivative = -((4 * (dx_i - dx_j) * (value - distance) ** 2) / distance ** (6./2.)) - \
                          ((2 * (dx_i - dx_j) * (value - distance)) / distance ** (5./2.))
            return derivative
        force_ixyz = np.zeros(np.shape(positions))

        # ij = 0
        # for ii in range(0, self.num_atoms):
        #     index_ii = np.arange(ii * 3, (ii + 1) * 3)
        #     atom_ii = positions[index_ii]
        #     for cc in range(0, 3):
        #         force = 0.0
        #         for jj in range(ii+1, self.num_atoms): #ii+1
        #             # if ii == jj:
        #             #     continue
        #             index_jj = np.arange(jj * 3, (jj + 1) * 3)
        #             atom_jj = positions[index_jj]
        #             force = force + func_derivative(atom_ii[cc], atom_jj[cc], atom_ii, atom_jj, d_ij_k[ii, jj])
        #
        #         force_ixyz[ij] = force
        #         ij += 1
        lower_index = np.tril_indices(self.num_atoms)
        for cc in range(0, 3):
            test_force = np.zeros([self.num_atoms, self.num_atoms])
            for ii in range(0, self.num_atoms):
                index_ii = np.arange(ii * 3, (ii + 1) * 3)
                atom_ii = positions[index_ii]
                for jj in range(ii+1, self.num_atoms):
                    if ii == jj:
                        continue
                    index_jj = np.arange(jj * 3, (jj + 1) * 3)
                    atom_jj = positions[index_jj]
                    test_force[ii, jj] = func_derivative(atom_ii[cc], atom_jj[cc], atom_ii, atom_jj, d_ij_k[ii, jj])
            test_force[lower_index] = -test_force.T[lower_index]
            force_ixyz[cc::3] = np.sum(test_force, axis=1)/2
        return force_ixyz

    def calc_energy_numerical(self, positions, d_ij_k):
        def func_distance(atom_i, atom_j):
            return np.linalg.norm(atom_i-atom_j)

        energy = 0.0
        for ii in range(0, self.num_atoms):
            index_ii = np.arange(ii * 3, (ii + 1) * 3)
            atom_ii = positions[index_ii]
            for jj in range(ii+1, self.num_atoms): #ii+1
                # if ii == jj:
                #     continue
                index_jj = np.arange(jj * 3, (jj + 1) * 3)
                atom_jj = positions[index_jj]
                distance = func_distance(atom_ii, atom_jj)
                energy = energy + 1./distance**(4./2.) * (d_ij_k[ii, jj]-distance)**2
        return energy

    def energy_gradient_idpp_function(self, position, d_ij_k):
        grad = self.calc_forces_numerical(position, d_ij_k)
        e = self.calc_energy_numerical(position, d_ij_k)
        return e, grad


class Images:
    def __init__(self, images, atom_list=None):
        # atom_list ... string list
        # idpp ... image dependent pair potential
        self.images = images
        self.number_images_check()
        self.set_optimizer_check = False
        self.energy_gradient_func = None
        self.atom_list = atom_list
        self.opt_minima = False # if True the minima should get closer to their real minimum

    def get_energy_gradient_of_image(self, position):
        for element in self.images:
            if all(element.position) == position:
                self.energy_gradient_func(element)

    def set_energy_gradient_func(self, energy_gradient_func):
        self.energy_gradient_func = energy_gradient_func
        for element in self.images:
            element.energy_gradient_func = energy_gradient_func

    def set_positions(self, positions):
        # positions in a 3D matrix num_images x atoms x coordinates
        for ii in range(0, len(positions)):
            self.images[ii].position = positions[ii].reshape(np.shape(self.images[0].position))

    def get_image_with_index(self, index):
        return self.images[index]

    def get_positions(self):
        atoms = int(len(self.images[0].position)/3)
        positions = np.zeros([len(self.images), atoms, 3])
        for ii in range(0, len(self.images)):
            positions[ii, :, :] = self.images[ii].position.reshape([atoms, 3])
        return positions

    def get_images(self):
        return self.images

    def get_image_position_list(self):
        positions = []
        for element in self.images:
            positions.append(np.reshape(element.position, [len(self.atom_list), 3]))
        return positions

    def update_rot_Mat(self):
        positions, rot_mat = rotation_geometry(self.get_positions())
        self.set_positions(positions)
        # since the first image is not affected by rotation len(rot_mat) -1 = len(self.images)
        # rot mat of first images is identiy
        self.images[0].rot_mat = np.eye(len(rot_mat[0]))
        for ii in range(1, len(self.images)):
            self.images[ii].rot_mat = rot_mat[ii-1]

    def update_tangents(self):
        self.number_images_check()
        for ii in range(1, len(self.images) - 1):
            self.images[ii].set_tangent(get_tangent(self.images[ii - 1], self.images[ii], self.images[ii + 1]))
        if self.opt_minima:
            self.images[0].set_tangent(get_tangent(None, self.images[0], self.images[1]))
            self.images[-1].set_tangent(get_tangent(self.images[-2], self.images[-1], None))

    def update_spring_force(self):
        self.number_images_check()
        for ii in range(1, len(self.images) - 1):
            self.images[ii].spring_force = spring_force(self.images[ii - 1], self.images[ii], self.images[ii + 1])

    def update_energy_gradient(self, check):
        if not check:
            for element in self.images:
                element.set_energy_gradient(self.energy_gradient_func)
        else:
            for element in self.images:
                element.set_energy_gradient(self.energy_gradient_func, element.d_ij_k)

    def update_images(self, check):
        self.update_energy_gradient(check)
        self.update_tangents()
        self.update_spring_force()

    def set_optimizer(self, optimizer):
        for element in self.images:
            element.optimizer = copy.deepcopy(optimizer)
        self.set_optimizer_check = True

    def set_spring_constant(self, spring_constant):
        if isinstance(spring_constant, list):
            if len(spring_constant) == len(self.images)-2:
                for element in self.images[1:-1]:
                    element.spring_constant = spring_constant
        else:
            for element in self.images:
                element.spring_constant = spring_constant

    def add_image(self, index, position, spring_constant):
        img = Image(position)
        self.images.insert(index, img)

    def number_images_check(self):
        if len(self.images) < 3:
            print('Error to less images ')

    def get_max_force_surface_image(self):
        force = np.linalg.norm(self.images[1].force_norm())
        for ii in range(2, len(self.images) - 1):
            if force < np.linalg.norm(self.images[ii].force_norm):
                force = np.linalg.norm(self.images[ii].force_norm)
        return force

    def set_climbing_image(self, climbing_image):
        if climbing_image:
            index = self.get_index_image_energy_max()
            self.images[index].spring_constant = 0.0
            self.images[index].climbing_image = True
        else:
            return
        return index

    def get_index_image_energy_max(self):
        energy_image = self.images[0].energy
        index = 0
        for ii in range(1, len(self.images)):
            if energy_image < self.images[ii].energy:
                energy_image = self.images[ii].energy
                index = ii
        return index


class Image:
    def __init__(self, position):
        self.position = position

        # initialize this values
        self.spring_constant = 0.0
        self.spring_force = 0.0
        self.energy = 0.0
        self.gradient = 0.0
        self.tangent = None

        self.rot_mat = None
        self.optimizer = None
        self.climbing_image = False
        self.d_ij_k = None

        #
        self.force = None

    def set_tangent(self, tangent):
        self.tangent = tangent

    def set_energy_gradient(self, energy_gradient_function, *args):
        self.energy, self.gradient = energy_gradient_function(self.position, *args)

#
    def get_position(self):
        return np.reshape(self.position, [len(self.position)/3,3])

    def get_force(self, position, *args):
        self.position = position
        self.images_parent.update_single_image(self, args[0][0], args[0][1])

        if not self.climbing_image:
            force = np.dot(self.spring_force, self.tangent)*self.tangent - \
                   self.gradient + np.dot(self.gradient, self.tangent) * self.tangent  # -(gradient - parallel Component)
        else:
            force = 2.*np.dot(self.gradient, self.tangent) * self.tangent
        self.force = force
        return self.energy, force

    def new_get_force(self, *args):
        if not self.climbing_image:
            # print(self.gradient.reshape([8,3]))
            # print('--------')
            force = np.dot(self.spring_force, self.tangent)*self.tangent - \
                   self.gradient + np.dot(self.gradient, self.tangent) * self.tangent  # -(gradient - parallel Component)
        else:
            force = 2.*np.dot(self.gradient, self.tangent) * self.tangent
        return self.energy, force

    def force_norm(self):
        energy, grad = self.new_get_force()
        #
        # grad = self.force
        return np.linalg.norm(grad)

    def gradient_norm(self):
        normal_grad = -self.gradient + np.dot(self.gradient, self.tangent) *self.tangent
        return np.linalg.norm(normal_grad) # self.gradient

    def get_norm_force(self):
        return -self.gradient + np.dot(self.gradient, self.tangent) *self.tangent

    def get_para_force(self):
        return np.dot(self.spring_force, self.tangent)*self.tangent

    def ratio_spring_grad(self):
        sp_force = np.dot(self.spring_force, self.tangent)*self.tangent
        grad = -self.gradient + np.dot(self.gradient, self.tangent) * self.tangent
        return sp_force/grad


class Optimizer:
    # Optimizer for the Nudged elastic band
    def __init__(self): # , trust_radius, fmax
        # self.trust_radius = trust_radius
        self.fmax = None

    def is_converged(self, images):
        for element in images[1:---1]:
            # if element.force_norm() > self.fmax:
            #     return False
            if element.gradient_norm() > self.fmax:
                return False
        return True

    def get_max_force(self, images):
        images = images.get_images()
        # force = images[1].force_norm()
        force = images[1].gradient_norm()
        jj = 1
        ii = 1
        for element in images[2:-1]:
            f = element.gradient_norm()
            # f = element.force_norm()
            ii = ii + 1
            if f > force:
                jj = ii
                force = f
        print(str(force) + ' of image ' + str(jj+1))

    def run_opt(self, images, optimizer, energy_gradient_func, max_steps=10000, force_max=0.05, opt_minima=False, rm_rot_trans=False, idpp=False):
        self.fmax = force_max
        images.set_optimizer(optimizer)
        images.set_energy_gradient_func(energy_gradient_func)
        data_writer = Writer()
        if opt_minima:
            index_start = 0
            index_end = len(images.get_images())
            images.opt_minima = opt_minima
        else:
            index_start = 1
            index_end = len(images.get_images())-1

        converged = False
        step = 0
        if rm_rot_trans:
            images.update_rot_Mat()
        images.update_images(idpp)
        while not converged:

            for ii in range(index_start, index_end):
                opt_method = images.get_images()[ii].optimizer
                # energy, grad = images.get_images()[ii].new_get_force()
                # print(grad)
                # print(images.get_images()[ii].get_norm_force())
                # print(images.get_images()[ii].get_para_force())
                # print('*************************************')
                images.get_images()[ii].position = opt_method.new_step(images.get_images()[ii].new_get_force, images.get_images()[ii].position)

            if rm_rot_trans:
                images.update_rot_Mat()
                for ii in range(index_start, index_end):
                    images.get_images()[ii].optimizer.update(images.get_images()[ii])
            images.update_images(idpp)

            if not idpp:
                # print('*************************************')
                xyz_writer.write_images2File(images.get_positions(), 'Test_step.xyz', images.atom_list)
                data_writer.write('Ethane_3_images_'+str(step)+'.xyz', images.get_image_position_list(), images.atom_list)

            if self.is_converged(images.get_images()):
                converged = True
                print('converged ' + str(step))
                for element in images.get_images()[1:-1]:
                    print(element.force_norm())
            print(str(step))
            self.get_max_force(images)
            print('---------------------------------------')

            step += 1
            if step >= max_steps:
                print('not converged' + str(step))
                for element in images.get_images()[1:-1]:
                    print(element.force_norm())
                break
        return images

    class SteepestDecentNeb(SteepestDecent):

        def update(self, img):
            # nothing has to be rotated
            pass

    class VerleteNeb(Verlete):

        def update(self, img):
            rot_mat = img.rot_mat
            self.velocity = np.dot(self.velocity.reshape([int(len(self.velocity)/3), 3]), rot_mat).flatten()

    class FireNeb(Fire):

        def update(self, img):
            rot_mat = img.rot_mat
            self.velocity = np.dot(self.velocity.reshape([int(len(self.velocity)/3), 3]), rot_mat).flatten()

    class BFGSNeb(BFGS):

        def update(self, img):
            rot_mat = img.rot_mat
            self.gradient = np.dot(rot_mat, self.gradient.reshape([int(len(self.gradient)/3), 3])).flatten()
            self.positions = np.dot(rot_mat, self.positions.reshape([int(len(self.positions)/3), 3])).flatten()
            self.hessian = np.dot(rot_mat, np.dot(self.hessian, rot_mat))

    class ConjugateGradientNeb(ConjuageGradient):

        def update(self, img):
            rot_mat = img.rot_mat
            self.force = np.dot(self.force.reshape([int(len(self.force)/3), 3]), rot_mat).flatten()
            self.force_before = np.dot(self.force_before.reshape([int(len(self.force_before)/3), 3]), rot_mat).flatten()
            self.s = np.dot(self.s.reshape([int(len(self.s)/3), 3]), rot_mat).flatten()
