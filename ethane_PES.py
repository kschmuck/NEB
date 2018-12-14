import pyqchem as qc
import os
import numpy as np

rem = qc.input_classes.rem_array(rem_init="jobtype force")
rem.add("SYM_IGNORE", "TRUE")
rem.add("exchange", "hf")
rem.add("basis", "STO-3G")
rem.add("GEOM_OPT_COORDS", "0")

scratch_path = os.environ.get('QCSCRATCH')


def energy_and_gradient(xyzs, *args): # argmuent of image
    if len(args) != 0:
        dijk, number = args
        name = os.path.join('./Input_Files', 'Ethane_image_' + str(number))
    else:
        name = os.path.join('./Input_Files', 'Ethane')

    in_file = qc.input_classes.inputfile()
    geo = qc.input_classes.cartesian(
        atom_list = [["H", str(xyzs[0]), str(xyzs[1]), str(xyzs[2])],
                     ["H", str(xyzs[3]),  str(xyzs[4]),  str(xyzs[5])],
                     ["H", str(xyzs[6]),  str(xyzs[7]),  str(xyzs[8])],
                     ["C", str(xyzs[9]),  str(xyzs[10]), str(xyzs[11])],
                     ["C", str(xyzs[12]), str(xyzs[13]), str(xyzs[14])],
                     ["H", str(xyzs[15]), str(xyzs[16]), str(xyzs[17])],
                     ["H", str(xyzs[18]), str(xyzs[19]), str(xyzs[20])],
                     ["H", str(xyzs[21]), str(xyzs[22]), str(xyzs[23])]])

    if os.path.isfile(os.path.join(scratch_path, name + '.dir', str(53) + '.' + str(0))):
        rem.add("SCF_GUESS", "READ")

    in_file.add(geo)
    in_file.add(rem)
    in_file.run(name=name+'.in')
    out_file = qc.read(name + ".out", silent=True)
    if out_file.force.gradient_vector is None:
        gradient = np.array([np.NAN] * 24).reshape(-1, 1)
    else:
        try:
            gradient = out_file.force.gradient_vector.T.flatten()
            gradient = gradient * (1/0.52917721067) # https://de.wikipedia.org/wiki/Bohrscher_Radius
        except:
            print("gradient error occurred")
    if out_file.general.energy is None:
        energy = np.NAN
    else:
        try:
            energy = out_file.general.energy
        except:
            print('energy error occurred')
    scf_guess = None
    return energy, gradient, scf_guess # out_file.opt.energies[0], out_file.opt.gradient_vector[0]
