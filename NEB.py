import numpy as np
import copy
import xyz_file_writer as xyz_writer
# import sympy.geometry as sym_geom
# import sympy as sym
import sys
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


def center_geometry(positions, number_of_coordinates=3):
    # cartesian
    # atoms in row and coordiantes in columns x, y, z
    mean_value = np.zeros([len(positions), number_of_coordinates])
    for kk in range(0, len(positions)):
        pos = positions[kk, :, :]
        for cc in range(0, number_of_coordinates):
            mean_value[kk, cc] = np.mean(pos[:, cc])
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


def get_tangent(img_0, img_1, img_2, method='improved'):
    # img_i is one image
    # img_0 = image before
    # img_1 = current image, at this image the tangent is set.
    # img_2 = image after
    # if img_i is None the tangent will be the spring to the next img  --> only valid for i = 0, 2

    if img_0 is None:
        tangent = img_1.get_current_position() - img_2.get_current_position()
        return tangent/np.linalg.norm(tangent)
    if img_2 is None:
        tangent = img_0.get_current_position() - img_1.get_current_position()
        return tangent/np.linalg.norm(tangent)

    # Methods for finding saddle points and minimum energy paths
    # Henkelman, Johannesson and Jonnson
    # Part of the Book series "Progress in theoretical Chemistry and Physics", Volume 5, Chapter 10
    # DOI: https://doi.org/10.1007/0-306-46949-9_10
    # Simple method, equation 5
    if method == 'simple':
        tangent = img_2.get_current_position() - img_0.get_current_position()

    # Improved simple method, equation 6
    elif method =='simple_improved':
        t_a = img_1.get_current_position() - img_0.get_current_position()
        t_b = img_2.get_current_position() - img_1.get_current_position()
        tangent = t_a / np.linalg.norm(t_a) + t_b / np.linalg.norm(t_b)

    # Improved tangent estimate in the nudged elastic band method for minimum energy paths and saddle points
    # Henkelman, Jonsson
    # Journal of chemical Physics, Volume 113, number 22
    # equation: 8-11
    # energy weighted tangent, useful if energy changes rapidly between the images
    elif method == 'improved':
        def min_max(energy_0, energy_1, energy_2):
            a = abs(energy_2 - energy_1)
            b = abs(energy_0 - energy_1)
            return min([a, b]), max([a, b])

        if img_2.get_current_energy() > img_1.get_current_energy() > img_0.get_current_energy():
            tangent = img_2.get_current_position() - img_1.get_current_position()
        elif img_2.get_current_energy() < img_1.get_current_energy() < img_0.get_current_energy():
            tangent = img_1.get_current_position() - img_0.get_current_position()
        else:
            minimum, maximum = min_max(img_0.get_current_energy(), img_1.get_current_energy(), img_2.get_current_energy())
            t_p = (img_2.get_current_position() - img_1.get_current_position()) / np.linalg.norm((img_2.get_current_position() - img_1.get_current_position()))
            t_m = (img_1.get_current_position() - img_0.get_current_position()) / np.linalg.norm((img_1.get_current_position() - img_0.get_current_position()))
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
    force = img_1.spring_constant * (img_2.get_current_position() - img_1.get_current_position()) \
            + img_1.spring_constant * (img_0.get_current_position() - img_1.get_current_position())
    return force


class ImageSet(list):
    def __init__(self, image_list, atom_list=None):
        # images ... list of image
        # atom_list ... string list
        list.__init__(self, image_list)
        self.number_images_check()
        self.energy_gradient_func = None
        self.atom_list = atom_list
        # self.tangent_calc_name=

    def set_positions(self, positions):
        # positions in a 3D matrix num_images x atoms x coordinates
        for ii in range(0, len(positions)):
            self[ii].set_position(positions[ii].reshape(np.shape(self[0].get_current_position())))

    def get_positions(self):
        atoms = int(len(self[0].get_current_position())/3)
        positions = np.zeros([len(self), atoms, 3])
        for ii in range(0, len(self)):
            positions[ii, :, :] = self[ii].get_current_position().reshape([atoms, 3])
        return positions

    def get_image_position_list(self):
        positions = []
        for element in self:
            positions.append(np.reshape(element.position, [len(self.atom_list), 3]))
        return positions

    def get_image_energy_list(self):
        energy = []
        for element in self:
            energy.append(element.get_current_energy())
        return energy

    def get_image_position_2D_array(self):
        positions = np.zeros([len(self), len(self[0].get_current_position())])
        for ii in range(0, len(self)):
            positions[ii, :] = self[ii].get_current_position()
        return positions

    def get_image_gradient_2Darray(self):
        force = np.zeros([len(self), len(self[0].get_current_gradient())])
        for ii in range(0, len(self)):
            force[ii, :] = self[ii].get_current_gradient()
        return force

    def update_rot_Mat(self):
        positions, rot_mat = rotation_geometry(self.get_positions())
        self.set_positions(positions)
        # since the first image is not affected by rotation len(rot_mat) -1 = len(self.images)
        # rot mat of first images is identity
        self[0].rot_mat = np.eye(len(rot_mat[0]))
        for ii in range(1, len(self)):
            self[ii].rot_mat = rot_mat[ii-1]

    def update_tangents(self, tangent_method):
        for ii in range(0, len(self)):
            if ii == 0:
                self[0].tangent = get_tangent(None, self[0], self[1], method=tangent_method)
            elif ii == (len(self)-1):
                self[-1].tangent = get_tangent(self[-2], self[-1], None, method=tangent_method)
            else:
                self[ii].tangent = get_tangent(self[ii - 1], self[ii], self[ii + 1], method=tangent_method)

    def update_spring_force(self):
        for ii in range(1, len(self) - 1):
            self[ii].spring_force = spring_force(self[ii - 1], self[ii], self[ii + 1])

    def update_energy_gradient(self):
        ii = 0
        for element in self:
            element.update_energy_gradient(self.energy_gradient_func, element.d_ij_k, ii)
            ii += 1

    def update_images(self, tangent_method):
        self.update_energy_gradient()
        self.update_tangents(tangent_method)
        self.update_spring_force()

    def set_optimizer(self, optimizer):
        for element in self:
            element.optimizer = copy.deepcopy(optimizer)

    def set_spring_constant(self, spring_constant):
        if isinstance(spring_constant, list):
            if len(spring_constant) == len(self)-2:
                for element in self[1:-1]:
                    element.spring_constant = spring_constant
        else:
            for element in self[1:-1]:
                element.spring_constant = spring_constant
        self[0].spring_constant = 0.0
        self[-1].spring_constant = 0.0

    def number_images_check(self):
        if len(self) < 3:
            print('Error to less images ')

    def set_climbing_image(self):
        index = self.get_index_image_energy_max()
        self[index].spring_constant = 0.0
        self[index].climbing_image = True
        print('**************************')
        print('climbing image: ' + str(index))
        print('**************************')
        return index

    def get_index_image_energy_max(self):
        energy = np.asarray(self.get_image_energy_list())
        return np.argmax(energy)


class Image:
    def __init__(self, position):
        # initialize values
        # history of position, energy and gradient
        self.position = []
        self.energy = []
        self.gradient = []
        # for faster convergence use the previous wave function guess
        self.scf_guess = None

        # values of nuged elastic band
        self.frozen = False
        self.count_unfrozen = 0
        self.count_frozen = 0

        self.climbing_image = False
        self.spring_constant = 0.0
        self.spring_force = 0.0
        self.tangent = None
        self.rot_mat = None
        self.optimizer = None
        self.d_ij_k = None

        self.position.append(position)

    def set_position(self, position):
        self.position.append(position)

    def set_gradient(self, gradient):
        self.gradient.append(gradient)

    def set_energy(self, energy):
        self.energy.append(energy)

    def update_energy_gradient(self, energy_gradient_function, *args):
        if self.frozen:
            self.energy.append(self.get_current_energy())
            self.gradient.append(self.get_current_gradient())
        else:
            energy, gradient, scf_guess = energy_gradient_function(self.get_current_position(), *args)
            self.energy.append(energy)
            self.gradient.append(gradient)
            self.scf_guess = scf_guess

    def get_current_position(self):
        return self.position[-1]

    def get_current_energy(self):
        return self.energy[-1]

    def get_current_gradient(self):
        return self.gradient[-1]

    def get_energy_force(self, *args):
        if not self.climbing_image:
            force = np.dot(self.spring_force, self.tangent)*self.tangent - \
                   self.gradient[-1] + np.dot(self.gradient[-1], self.tangent) * self.tangent  # -(gradient - parallel Component)
        else:
            force = 2.*np.dot(self.gradient[-1], self.tangent) * self.tangent
        return self.energy[-1], force

    def force_norm(self):
        energy, force = self.get_energy_force()
        return np.linalg.norm(force)


class Optimizer:
    # Optimizer for the Nudged elastic band
    def __init__(self): # , trust_radius, fmax
        self.fmax = None

    def is_converged(self, imgs):
        for element in imgs:
            if element.force_norm() > self.fmax:
                return False
        return True

    def get_max_force(self, images):
        force = images[0].force_norm()
        jj = 1
        ii = 1
        for element in images[1:]:
            f = element.force_norm()
            ii = ii + 1
            if f > force:
                jj = ii
                force = f
        print(str(force) + ' of image ' + str(jj))

    def run_opt(self, images, optimizer, max_steps=10000, force_max=0.05, opt_minima=False, rm_rot_trans=False,
                freezing=0, tangent_method='improved', write_geom=False, print_step=False):

        self.fmax = force_max
        images.set_optimizer(optimizer)
        sys.stdout.flush()
        converged = False
        step = 0
        if rm_rot_trans:
            images.update_rot_Mat()
        images.update_images(tangent_method)

        if not opt_minima:
            lower = 1
            upper = -1
            images[0].frozen = True
            images[-1].frozen = True
        else:
            lower = 0
            upper = len(images)+1

        while not converged:
            sys.stdout.flush()
            for element in images[lower:upper]:
                if not element.frozen:
                    opt_method = element.optimizer
                    element.set_position(opt_method.step(element.get_energy_force, element.get_current_position()))
            if print_step:
                print('nudged elastic band step = %d') % (step)
                uu = 0
                for element in images:
                    print('force %f of image %d ') %(element.force_norm(), uu)
                    uu += 1

            if write_geom:
                xyz_writer.write_images2File(images.get_positions(), 'NEB_'+ str(step)+'.xyz', images.atom_list)

            if rm_rot_trans:
                images.update_rot_Mat()
                for element in images[lower:upper]:
                    element.optimizer.update(element)
            images.update_images(tangent_method)

            if freezing > 0:
                for element in images[lower:upper]:
                    if element.frozen:
                        element.count_frozen -= 1
                        if element.count_frozen < 0:
                            element.frozen = False
                        else:
                            if element.force_norm() < self.fmax:
                                element.frozen = True
                                element.count_frozen = freezing

            if self.is_converged(images[lower:upper]):
                converged = True
                print('converged ' + str(step))
                # for element in images[lower:upper]:
                #     print(element.force_norm())
            step += 1
            if step >= max_steps:
                print('not converged ' + str(step))
                for element in images[lower:upper]:
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
            if self.skip:
                return
            rot_mat = img.rot_mat
            self.force = np.dot(self.force.reshape([int(len(self.force)/3), 3]), rot_mat).flatten()
            self.force_before = np.dot(self.force_before.reshape([int(len(self.force_before)/3), 3]), rot_mat).flatten()
            self.s = np.dot(self.s.reshape([int(len(self.s)/3), 3]), rot_mat).flatten()
