import numpy as np

def write2file(number_of_atoms, atom_list, coordinates, file):
    file.write(str(number_of_atoms) + '\n\n')
    for ii in range(0, number_of_atoms):
        file.write(atom_list[ii] + ' ')
        file.write(' '.join(map(str, coordinates[ii, :]))+'\n')
    # file.write('C ' + ' '.join(map(str, coordinates[[3,4,5]]))+'\n')
    # file.write('N ' + ' '.join(map(str, coordinates[[6,7,8]]))+'\n')
    file.write('\n')

def write_images2File(positions, fileName, atom_list):
    f = open(fileName, 'w')
    number_atoms = len(atom_list)
    for element in positions:
        write2file(number_atoms, atom_list, element, f)

def write_image2File(positions, file_name, atom_list):
    f = open(file_name, 'w')
    number_atoms = len(atom_list)
    for ii in range(0, number_atoms):
        f.write(atom_list[ii] + ' ')
        for jj in range(0,3):
            f.write(str(positions[ii,jj]))
            f.write(' ')
        f.write('\n')