import numpy as np


class IDPP:
    def __init__(self, images, coordinate_system='cartesian', number_of_coordinates=3):
        if coordinate_system == 'cartesian':
            self.num_atoms = int(len(images[0].get_current_position())/number_of_coordinates)
            self.number_of_coordinates = number_of_coordinates

        d_product = self.calc_distance(images[0].get_current_position())
        d_reactant = self.calc_distance(images[-1].get_current_position())

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

        lower_index = np.tril_indices(self.num_atoms)
        for cc in range(0, self.number_of_coordinates):
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
            force_ixyz[cc::self.number_of_coordinates] = np.sum(test_force, axis=1)/2
        return force_ixyz

    def calc_energy_numerical(self, positions, d_ij_k):
        def func_distance(atom_i, atom_j):
            return np.linalg.norm(atom_i-atom_j)

        energy = 0.0
        for ii in range(0, self.num_atoms):
            index_ii = np.arange(ii * 3, (ii + 1) * 3)
            atom_ii = positions[index_ii]
            for jj in range(ii+1, self.num_atoms):
                index_jj = np.arange(jj * 3, (jj + 1) * 3)
                atom_jj = positions[index_jj]
                distance = func_distance(atom_ii, atom_jj)
                energy = energy + 1./distance**(4./2.) * (d_ij_k[ii, jj]-distance)**2
        return energy

    def energy_gradient_idpp_fucntion(self, position, *args):
        d_ij_k = args[0]

        gradient = self.calc_forces_numerical(position, d_ij_k)
        energy = self.calc_energy_numerical(position, d_ij_k)
        return energy, gradient, None
