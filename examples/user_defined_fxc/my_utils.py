from kspies import wy, util
import numpy as np
import pyscf
from pyscf import tdscf

def hartree_in_ev():
    return 27.211386245988

def compute_numerical_fxc(molecule, dm_target, coords, epsilon=5e-7, guide="Slater",
                          regularization=0.0, method="SLSQP", xctype="LDA", wy_density=False):
    """ Computes fxc numerically using WY inversion and a target density matrix.

        :param dm_target: the target density matrix.
        :param coords: the grid coordinates.
        :param epsilon: the small change to be performed in the target density matrix elements.
                        to obtain dm_plus and dm_minus (used in the numerical differentiation procedure).
        :param guide: the guiding potential for the inversion procedure.
        :param regularization: the regularization parameter.
        :param method: the inversion method.

        returns a dictionary with:
            - the numerical fxc on the grid
            - the MO coefficients after the inversion proceudre for the target density matrix.
            - the orbital eigenvalues obtained after inversion for the target density matrix.
    """
    delta_rho = epsilon * np.diag(np.ones(dm_target.shape[0]))
    dm_plus = dm_target + delta_rho
    dm_minus = dm_target - delta_rho
    
    ao_deriv_order = 0
    ao_values = pyscf.dft.numint.eval_ao(molecule, coords,
                                         deriv = ao_deriv_order)
    
    print("Running WY using kspies")
    # WY plus
    mw_plus = wy.RWY(molecule, dm_plus)
    mw_plus.guide = guide
    mw_plus.tol = 1e-6
    mw_plus.method = method
    mw_plus.reg = regularization
    mw_plus.run()
    wy_dm_plus = mw_plus.dm   
 
    # WY minus
    mw_minus = wy.RWY(molecule, dm_minus)
    mw_minus.guide = guide
    mw_minus.tol = 1e-6
    mw_minus.method = method
    mw_minus.reg = regularization
    mw_minus.run()
    wy_dm_minus = mw_minus.dm
    print("Finished!")   
 
    # TODO: Extract data (! VG is extracted differently for LDA, GGA, and FA!!!)
    #
    wy_dm_plus_on_grid = pyscf.dft.numint.eval_rho(molecule, ao_values,
                                               wy_dm_plus, xctype='LDA')
    wy_dm_minus_on_grid = pyscf.dft.numint.eval_rho(molecule, ao_values,
                                               wy_dm_minus, xctype='LDA')
    diff_wy_plus_mius = wy_dm_plus_on_grid - wy_dm_minus_on_grid
    wy_delta_rho_on_grid = pyscf.dft.numint.eval_rho(molecule, ao_values,
                                               wy_dm_plus - wy_dm_minus, xctype='LDA')

    vc_plus = np.einsum('t,rt->r', mw_minus.b, ao_values) # correction potential on grid
    vc_minus = np.einsum('t,rt->r', mw_minus.b, ao_values) # correction potential on grid

    if xctype == "GGA":
        vg_plus = util.eval_vxc(molecule, dm_plus, mw_plus.guide, coords) # guiding potential on grid
        vg_minus = util.eval_vxc(molecule, dm_minus, mw_minus.guide, coords) # guiding potential on grid
    elif xctype == "LDA":
        dm_plus_on_grid = pyscf.dft.numint.eval_rho(molecule, ao_values,
                                               dm_plus, xctype='LDA')
        dm_minus_on_grid = pyscf.dft.numint.eval_rho(molecule, ao_values,
                                               dm_minus, xctype='LDA')
        vg_plus = pyscf.dft.libxc.eval_xc(guide, dm_plus_on_grid)[1][0]
        vg_minus = pyscf.dft.libxc.eval_xc(guide, dm_minus_on_grid)[1][0]
    elif xctype == "FAXC":
        dmxc_plus = -1.0/molecule.nelectron * dm_plus
        vg_plus = util.eval_vh(molecule, coords, dmxc_plus) # guiding potential on grid
        dmxc_minus = -1.0/molecule.nelectron * dm_minus
        vg_minus = util.eval_vh(molecule, coords, dmxc_minus) # guiding potential on grid
    else:
        raise NotImplementedError("Unrecognized xctype: %s." % xctype)
    # xctype = LDA because we need only the density and not the density gradient
    delta_rho_on_grid = pyscf.dft.numint.eval_rho(molecule, ao_values,
                                      delta_rho, xctype='LDA')
    
    # calculate fxc numerically; symmetric quotient
    if wy_density:
        fxc_symm = ( vg_plus + vc_plus - vg_minus - vc_minus ) / ( 2 * wy_delta_rho_on_grid)
    else:
        fxc_symm = (vg_plus + vc_plus - vg_minus - vc_minus) / (2 * delta_rho_on_grid)
   
    mw = wy.RWY(molecule, dm_target)
    mw.guide = guide
    mw.tol = 1e-6
    mw.method = method
    mw.reg = regularization
    mw.run()

    mo_energies = mw.mo_energy  # MO eigenvalues
    mo_coeff = mw.mo_coeff    # MO coefficients
 
    return {
        'fxc': fxc_symm,
        'delta_rho_on_grid': delta_rho_on_grid,
        'wy_delta_rho_on_grid_dm_diff': wy_delta_rho_on_grid,
        'wy_delta_rho_on_grid_grid_diff': diff_wy_plus_mius,
        'mo_energies': mo_energies,
        'mo_coeff': mo_coeff,
        }

def add_broadening(bge, bgi, line_param=0.1, line_profile="lorentzian",
                   step=0.1, interval=None):
    """ Adds a Gaussian or Lorentzian broadening to a bar graph spectrum.

        :param bge         : the numpy array of energies.
        :param bgi         : the numpy array of intensities.
        :param line_param  : the line parameter.
        :param line_profile: the line profile (guassian or lrentzian)
        :param step        : the step size.
        :param interval    : the energy interval where the broadening
                             should be applied.
    """
    if interval is None:
        x_min = np.min(bge) - 5
        x_max = np.max(bge) + 5
    else:
        x_min = interval[0]
        x_max = interval[-1]

    x = np.arange(x_min, x_max, step)
    y = np.zeros((len(x)))
    
    # go through the frames and calculate the spectrum for each frame
    for xp in range(len(x)):
        for e, f in zip(bge, bgi):
            if line_profile in ['Gaussian', 'gaussian', "Gauss", "gauss"]:
                y[xp] += f * np.exp(-(
                    (e - x[xp]) / line_param)**2)
            elif line_profile in ['Lorentzian', 'lorentzian',
                                  'Lorentz', 'lorentz']:
                y[xp] += 0.5 * line_param * f / (np.pi * (
                        (x[xp] - e)**2 + 0.25 * line_param**2))
    return x, y

def exact_diagonalization(pyscf_molecule, scf_gs,
                          tda=False, cvs_space=None,
                          fxc=None):
    """ Computes the absorption spectrum using TDDFT.

        :param pyscf_molecule: the pyscf molecule object
        :param scf_gs        : the SCF reference state.
        :param tda           : if to use the Tamm-Dancoff approximation.
        :param cvs_space     : a list of the core orbital indices
                               (for the CVS approximation).
    """

    if tda:
        tdscf_drv = pyscf.tdscf.TDA(scf_gs)
    else:
        tdscf_drv = pyscf.tdscf.TDDFT(scf_gs)

    tdscf_drv.user_defined_fxc = fxc

    A, B = tdscf_drv.get_ab()

    nocc = A.shape[0]
    nvir = A.shape[1]

    mo_occ = scf_gs.mo_coeff[:, :nocc]
    mo_vir = scf_gs.mo_coeff[:, nocc:]
    nao = mo_occ.shape[0]

    electric_dipole_integrals_ao = np.sqrt(2) * pyscf_molecule.intor("int1e_r",
                                                                    aosym='s1')
    
    if tda:
        if cvs_space is None:
            E2 = A.reshape(nocc*nvir, nocc*nvir)
            S2 = np.identity(nocc * nvir)
            mu_mo = np.einsum("mi,xmn,na->xia", mo_occ,
                    electric_dipole_integrals_ao, mo_vir).reshape(3, nocc*nvir)
        else:
            ncore = len(cvs_space)
            eI = A[cvs_space, :, :, :]
            E2 = eI[:,:,cvs_space,:].reshape(ncore*nvir, ncore*nvir)
            S2 = np.identity(ncore * nvir)
            mu_mo = np.einsum("mi,xmn,na->xia", mo_occ[:,cvs_space],
                electric_dipole_integrals_ao, mo_vir).reshape(3, ncore*nvir)
        prop_grad = mu_mo
        omega, x = np.linalg.eigh(E2)

    else:
        if cvs_space is None:
            E2 = np.zeros((2*nocc*nvir, 2*nocc*nvir))
            E2[:nocc*nvir, :nocc*nvir] = A.reshape(nocc*nvir, nocc*nvir)
            E2[nocc*nvir:, nocc*nvir:] = A.reshape(nocc*nvir, nocc*nvir)
            E2[:nocc*nvir, nocc*nvir:] = -B.reshape(nocc*nvir, nocc*nvir)
            E2[nocc*nvir:, :nocc*nvir] = -B.reshape(nocc*nvir, nocc*nvir)
            S2 = np.identity(2 * nocc * nvir)
            S2[nocc*nvir:, nocc*nvir:] *= -1
            mu_mo = np.einsum("mi,xmn,na->xia", mo_occ,
                    electric_dipole_integrals_ao, mo_vir).reshape(3, nocc*nvir)
            prop_grad = np.zeros((3, 2*nocc*nvir))
            prop_grad[:, :nocc*nvir] = mu_mo
            prop_grad[:, nocc*nvir:] = -mu_mo
        else:
            ncore = len(cvs_space)
            E2 = np.zeros((2*ncore*nvir, 2*ncore*nvir))
            AI = A[cvs_space, :, :, :]
            BI = B[cvs_space, :, :, :]
            E2[:ncore*nvir, :ncore*nvir] = AI[:, :, cvs_space, :].reshape(
                                                         ncore*nvir, ncore*nvir)
            E2[ncore*nvir:, ncore*nvir:] = AI[:, :, cvs_space, :].reshape(
                                                        ncore*nvir, ncore*nvir)
            E2[:ncore*nvir, ncore*nvir:] = -BI[:, :, cvs_space, :].reshape(
                                                        ncore*nvir, ncore*nvir)
            E2[ncore*nvir:, :ncore*nvir] = -BI[:, :, cvs_space, :].reshape(
                                                        ncore*nvir, ncore*nvir)
            S2 = np.identity(2 * ncore * nvir)
            S2[ncore*nvir:, ncore*nvir:] *= -1
            mu_mo = np.einsum("mi,xmn,na->xia", mo_occ[:,cvs_space],
                    electric_dipole_integrals_ao, mo_vir).reshape(3, ncore*nvir)
            prop_grad = np.zeros((3, 2*ncore*nvir))
            prop_grad[:, :ncore*nvir] = mu_mo
            prop_grad[:, ncore*nvir:] = -mu_mo

        eigs, X = np.linalg.eig(np.matmul(np.linalg.inv(S2), E2))
        idx = np.argsort(eigs)
        omega = np.array(eigs)[idx]
        x = np.array(X)[:, idx]
        
        if cvs_space is None:
            omega = omega[nocc*nvir:]
            x = x[:, nocc*nvir:]
        else:
            omega = omega[ncore*nvir:]
            x = x[:, ncore*nvir:]

    n = omega.shape[0]
    tdms = np.zeros((n, 3))
    osc = np.zeros_like(omega)
    for k in range(n):
        Xf = x[:, k]
        Xf = Xf / np.sqrt(np.matmul(Xf.T, np.matmul(S2, Xf)))
        tdms[k] = np.einsum("i,xi->x", Xf, prop_grad)
        osc[k] = 2.0/3.0 * omega[k] * ( tdms[k,0]**2 
                                      + tdms[k,1]**2
                                      + tdms[k,2]**2)
            
    return {'eigenvalues': omega,
            'eigenvectors': x,
            'oscillator strengths': osc,
            'transition dipole moments': tdms,
            'E2': E2,
            'S2': S2}
