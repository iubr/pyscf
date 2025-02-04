import numpy as np
from pyscf import gto, scf, fci, dft
from pyscf.dft.numint import eval_rho, eval_ao
from kspies import wy, util
import csv
import matplotlib.pyplot as plt
from my_utils import *

# Use a simpler system for testing
mol = gto.M(atom="Be 0 0 0", basis="unc-aug-cc-pvtz")
mf_hf = scf.RHF(mol).run()

# Run FCI
# FIXME: doens't work for me for some reason, I need to look into this!
# For now, I send the HF results
# cisolver = fci.FCI(mol, mf_hf.mo_coeff)
# e_fci, ci = cisolver.kernel()
# D = cisolver.make_rdm1(ci, mol.nao, mol.nelectron)
D = mf_hf.make_rdm1(mf_hf.mo_coeff, mf_hf.mo_occ)

# Define grid and perturbations
# In order to calculate a TDDFT spectrum, fxc must be calculated on the DFT grid.
# We need to define a DFT grid, it's not enough to have a linear grid.
# The easiest is to use a DFT driver:
dft_drv = scf.RKS(mol)
dft_drv.xc = "LDA"
dft_drv.grids.level = 6
dft_drv.kernel()

# coords = np.array([(0., 0., x) for x in np.linspace(-5, 5, 101)])
coords = dft_drv.grids.coords
ao_values = dft.numint.eval_ao(mol, coords)

# Save grid to file -- just in case it's needed:
np.save("grid.npy", coords)

# Test different values of epsilon
epsilon_values = [1e-4, 1e-5, 1e-6]
for epsilon in epsilon_values:
    # Run a Wu-Yang inversion using FCI target density and
    # save the MO coefficients and MO energies for the TDDFT
    # spectrum calculation.
    wy_fci = wy.RWY(mol, D)
    wy_fci.method = 'L-BFGS-B'
    wy_fci.tol = 1e-8
    wy_fci.reg = 1e-6  # Regularization
    wy_fci.guide = 'faxc'
    wy_fci.run()
    wy_fci.info()

    mo_energy = wy_fci.mo_energy  # MO eigenvalues
    mo_coeff = wy_fci.mo_coeff    # MO coefficients

    delta_rho = epsilon * np.eye(D.shape[0])
    drho = np.einsum('ij,gi,gj->g', delta_rho, ao_values, ao_values)

    # Symmetrize and normalize perturbed density matrices
    P_tar_1 = D + delta_rho
    #P_tar_1 = 0.5 * (P_tar_1 + P_tar_1.T)
    #P_tar_1 *= mol.nelectron / np.trace(P_tar_1 @ mol.intor("int1e_ovlp"))

    P_tar_2 = D - delta_rho
    #P_tar_2 = 0.5 * (P_tar_2 + P_tar_2.T)
    #P_tar_2 *= mol.nelectron / np.trace(P_tar_2 @ mol.intor("int1e_ovlp"))

    # Wu-Yang inversion with regularization
    wy_f0 = wy.RWY(mol, P_tar_1)
    wy_f0.method = 'L-BFGS-B'
    wy_f0.tol = 1e-8
    wy_f0.reg = 1e-6  # Regularization
    wy_f0.guide = 'faxc'
    wy_f0.run()
    wy_f0.info()
    vg0 = util.eval_vh(mol, coords, -1.0/mol.nelectron * P_tar_1)
    vC0 = np.einsum('t,rt->r', wy_f0.b, ao_values)

    wy_f1 = wy.RWY(mol, P_tar_2)
    wy_f1.method = 'L-BFGS-B'
    wy_f1.tol = 1e-8
    wy_f1.reg = 1e-6  # Regularization
    wy_f1.guide = 'faxc'
    wy_f1.run()
    wy_f1.info()
    vg1 = util.eval_vh(mol, coords, -1.0/mol.nelectron * P_tar_2)
    vC1 = np.einsum('t,rt->r', wy_f1.b, ao_values)

    # Calculate f_xc kernel using LDA functional at FCI density
    dummy_mf = dft.RKS(mol)
    dummy_mf.xc = 'lda'
    dummy_mf.grids.coords = coords
    dummy_mf.grids.weights = np.ones(len(coords))
    fxc = dummy_mf._numint.cache_xc_kernel1(mol, dummy_mf.grids, dummy_mf.xc, D, spin=0)[2]
    fxc_flat = fxc[0, 0, :]

    # Since these are numpy arrays, the following code should
    # take the difference and division element by element.
    vxc_diff = vg0 - vg1 + vC0 - vC1
    fxc_our = vxc_diff / ( 2 * drho)

    # Write data to CSV
    #csv_data = [["X (Bohr)", "(vxc+ - vxc-)/(2rho)", "f_xc"]]
    #for i in range(len(coords)):
    #    vxc_diff = vg0[i] - vg1[i] + vC0[i] - vC1[i]
    #    fxc_our = vxc_diff / (2 * drho[i])
    #    fxc_value = fxc_flat[i]
    #    csv_data.append([coords[i, 2], fxc_our, fxc_value])

    #csv_filename = f"fci_density_analysis_epsilon_{epsilon}.csv"
    #with open(csv_filename, "w", newline="") as f:
    #    writer = csv.writer(f)
    #    writer.writerows(csv_data)
    
    npy_filename = f"fci_density_numerical_fxc_epsilon_{epsilon}.npy"
    np.save(npy_filename, fxc_our)
    print(f"Results saved to {npy_filename}")

    # Calculate spectra for numerical fxc
    # we need the mo energies and mo coefficiets.
    scf_drv = scf.RHF(mol)
    scf_drv.mo_coeff = mo_coeff #mf_hf.mo_coeff
    scf_drv.mo_energy = mo_energy #mf_hf.mo_energy
    scf_drv.mo_occ = mf_hf.mo_occ

    # I enabled also a user-defined density matrix, but it does not make a difference.
    rsp_results = exact_diagonalization(mol, scf_drv, tda=False, fxc=fxc_our, dm0=D)

    # add broadening and plot spectrum
    xn, yn = add_broadening(rsp_results['eigenvalues'] * hartree_in_ev(),
              rsp_results['oscillator strengths'], step=0.02, line_param=0.5, interval=(0, 50))
    outname = f"fci_density_numerical_fxc_epsilon_{epsilon}" 
    fig = plt.figure()
    plt.title(outname)
    plt.plot(xn, yn, "-", label="Numerical fxc, symm. qotient")
    plt.bar(rsp_results['eigenvalues'] * hartree_in_ev(), rsp_results['oscillator strengths'], width=0.1)
    plt.xlabel("Energy (eV)")
    plt.ylabel("Oscillator strength")
    plt.axis(xmin=0, xmax=50)
    plt.savefig(outname + ".png")
    plt.show()
