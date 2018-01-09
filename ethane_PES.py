import pyQChem as qc

rem = qc.input_classes.rem_array(rem_init="jobtype opt")
rem.add("GEOM_OPT_TOL_GRADIENT", "10000000000")
rem.add("GEOM_OPT_TOL_DISPLACEMENT", "100000000000000")
rem.add("GEOM_OPT_TOL_ENERGY", "100000000000000000")
rem.add("GEOM_OPT_MAX_CYCLES", "1")
rem.add("SYM_IGNORE", "TRUE")
rem.add("exchange", "hf")
rem.add("basis", "aug-cc-pVDZ")
rem.add("GEOM_OPT_COORDS", "0")
# rem.add("SCF_GUESS", "READ")

def energy_and_gradient(xyzs, *args): # argmuent of image
    number = args[1]
    flag = args[2]
    if flag:
        rem.add("SCF_GUESS", "READ")
    name = 'ethane_' + str(number)
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
    in_file.add(geo)
    in_file.add(rem)
    in_file.run(name=name)
    out_file = qc.read(name + ".out", silent=True)
    gradient = out_file.opt.gradient_vector[0].T.flatten()
    energy = out_file.opt.energies[0]
    scf_guess = None
    return energy, gradient, scf_guess # out_file.opt.energies[0], out_file.opt.gradient_vector[0]
