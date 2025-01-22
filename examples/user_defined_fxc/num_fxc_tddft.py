import pyscf
from my_utils import *
from matplotlib import pyplot as plt

mol_name = "n2"
folder = ""
xyz_file_name = folder + mol_name + ".xyz"

functional = "PBE"
guide = "PBE"
xctype = "GGA" # GGA or LDA ! At the moment, you have to change this manually!
basis_name = "cc-pvdz"

molecule = pyscf.gto.Mole()
molecule.atom = xyz_file_name
molecule.basis = basis_name
molecule.build()

# Run the DFT reference
pyscf_drv = pyscf.scf.RKS(molecule)
pyscf_drv.xc = functional
pyscf_drv.verbose = 4
pyscf_drv.grids.level = 6
pyscf_drv.kernel()

# Target density
dm_0 = pyscf_drv.make_rdm1()

wy_dict = compute_numerical_fxc(molecule=molecule, dm_target=dm_0, coords=pyscf_drv.grids.coords,
                                guide=guide, xctype=xctype)
num_fxc = wy_dict['fxc']

# Run TDDFT/TDA by exact diagonalization using the numerical fxc:
num_diag_dict = exact_diagonalization(molecule, pyscf_drv, tda=False, fxc=num_fxc)

# Run TDDFT/TDA by exact diagonalization using the regular fxc
tda_drv = pyscf.tdscf.TDA(pyscf_drv)
diag_dict = exact_diagonalization(molecule, pyscf_drv, tda=False, fxc=None)

# add broadening to be able to plot the spectra
x, y = add_broadening(diag_dict['eigenvalues'] * hartree_in_ev(),
                    diag_dict['oscillator strengths'], step=0.02, line_param=0.5, interval=(0, 50))
xn, yn = add_broadening(num_diag_dict['eigenvalues'] * hartree_in_ev(),
              num_diag_dict['oscillator strengths'], step=0.02, line_param=0.5, interval=(0, 50))

# calculate relative errors
# relative difference energies
rel_diff_energies = (
    (diag_dict['eigenvalues'] - num_diag_dict['eigenvalues']) * 100 / diag_dict['eigenvalues']
            )
# relative difference osc. strengths
rel_diff_osc = (
    (diag_dict['oscillator strengths'] - num_diag_dict['oscillator strengths']) * 100
    / diag_dict['eigenvalues']
            )

# plot absorption spectra, relative errors in energies and relative errors in osc. strengths
outname = mol_name + "_" + functional + "_" + basis_name
svg_name = outname + ".svg"
fig = plt.figure(figsize=(5, 10))
plt.subplot(311)
plt.title(outname)
plt.plot(x, y, label="PyScf TDDFT, exact diagonalization")
plt.plot(xn, yn, "-.", label="Numerical fxc, symm. qotient")
#plt.xlabel("Energy (eV)")
plt.ylabel("Oscillator strength")
plt.axis(xmin=0, xmax=50)
plt.legend()

plt.subplot(312)
plt.bar(diag_dict['eigenvalues']*hartree_in_ev(), rel_diff_energies)
plt.title("Relative difference: excitation energies")
#plt.xlabel("Energy (eV)")
plt.ylabel("Relative difference (%)")
plt.axis(xmin=0, xmax=50, ymin=-5, ymax=0.1)

plt.subplot(313)
plt.bar(diag_dict['eigenvalues']*hartree_in_ev(), rel_diff_osc)
plt.title("Relative difference: oscillator strengths")
plt.xlabel("Energy (eV)")
plt.ylabel("Relative difference (%)")
plt.axis(xmin=0, xmax=50)
#plt.savefig(svg_name)
plt.show()
