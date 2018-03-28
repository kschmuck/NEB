import numpy as np


class Writer:

    def __init__(self):
        pass

# positions list of np.arrays
# first line --> atom number
# second line --> comment generally empty
#
    def write(self, filename, positions, atom_list, energy=None):
        atom_number = len(atom_list)
        f = open(filename, 'w')

        if energy is not None:
            comment = 'energy '
        else:
            comment = ''

        for element in positions:
            f.write(str(atom_number) + '\n')
            f.write(comment+'\n')
            f.writelines(self.create_position_string(element, atom_list))
        f.close()

    def create_position_string(self, position, atom_list):
        ii = 0
        position_string = []
        for element in atom_list:
            # print(position[ii])
            position_string.append(element + ' ' + ' '.join(map(str, position[ii]))+'\n')
            ii += 1
        position_string.append('\n')
        return position_string


# get the geometries of an xyz file with the atom number
class Reader:

    def __init__(self):
        self.atom_number = 0
        self.geometries = []
        self.images = []
        self.energies = []
        self.atom_list = []

    def read(self, filename):
        self.atom_number = 0
        self.geometries = []
        self.images = []

        self.energies = []

        self.atom_list = []
        count = 0
        for lines in open(filename).xreadlines():
            count += 1

        f = open(filename, 'r')
        ii = -1
        while True:
            ii = ii + 1
            if ii >= count:
                break
            line = f.readline()
            atom_number = int(line)

            atoms = []
            image = []
            geometry = []
            for jj in range(0, atom_number + 1):
                ii = ii + 1
                line = f.readline().split()
                if jj > 0:
                    image.append([line[0], np.array(map(float, line[1:]))])
                    geometry.append(np.array(map(float, line[1:])))
                    atoms.append(line[0])
                elif len(line) > 0:
                    if line[0] == 'energy':
                        self.energies.append(float(line[1]))
            ii = ii + 1
            f.readline()
            self.images.append(image)
            self.geometries.append(geometry)
        self.atom_list = atoms
        f.close()

    def read_new(self, filename):
        self.images = []

        with open(filename, 'r') as fin:
            if filename[-4:] == '.xyz':
                # first line should be the number of atoms
                # second is a comment line
                # then geometry lines occur until a new block starts
                atom_counter = 0
                for line in fin:
                    # atom number line
                    if atom_counter == 0:
                        atoms = []
                        geometry = []
                        if (line == '\n') | (line == ''):
                            continue
                        else:
                            atom_counter = int(line)
                            comment = True
                            continue
                    else:
                        if comment:
                            # comment line
                            line_elements = line.split()
                            if line.startswith('energy'):
                                energy = float(line_elements[1])
                                energy_flag = True
                            else:
                                comment = line
                                energy_flag = False
                            comment = False
                        else:
                            # geometry block
                            line_elements = line.split()
                            atoms.append(line_elements[0])
                            geometry.append(np.array(map(float, line_elements[1:])))
                            atom_counter -= 1
                        # after all lines of geometry block are readed
                        if atom_counter == 0:
                            if energy_flag:
                                image = {'atoms': atoms, 'geometry': geometry, 'energy': energy}
                            else:
                                image = {'atoms': atoms, 'geometry': geometry, 'comment': comment}
                            self.images.append(image)
                            energy_flag = False
