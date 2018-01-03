import numpy as np


class Writer:

    def __init__(self):
        pass

# positions list of np.arrays
# first line --> atom number

    def write(self, filename, positions, atom_list):
        atom_number = len(atom_list)
        f = open(filename, 'w')

        for element in positions:
            f.write(str(atom_number) + '\n')
            f.write('\n')
            f.writelines(self.create_position_string(element, atom_list))
        f.close()

    def create_position_string(self, position, atom_list):
        ii = 0
        position_string = []
        for element in atom_list:
            print(position[ii])
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

        self.atom_list = []


    def read(self, filename):
        count = 0
        for lines in open(filename).xreadlines():
            count += 1

        f = open(filename, 'r')
        ii = -1
        while True:
            ii = ii + 1
            if ii == count:
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
            ii = ii + 1
            f.readline()
            self.images.append(image)
            self.geometries.append(geometry)
        self.atom_list = atoms
        f.close()