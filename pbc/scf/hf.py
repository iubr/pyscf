'''
Hartree-Fock for periodic systems at a single k-point

See Also:
    pyscf.pbc.scf.khf.py : Hartree-Fock for periodic systems with k-point sampling
'''

import sys
import numpy as np
import scipy.linalg
import pyscf.lib
import pyscf.scf
import pyscf.scf.hf
import pyscf.dft
import pyscf.pbc.dft
import pyscf.pbc.dft.numint
import pyscf.pbc.scf
from pyscf.lib import logger
from pyscf.lib.numpy_helper import cartesian_prod
from pyscf.pbc import tools
from pyscf.pbc import ao2mo
from pyscf.pbc.gto import pseudo
from pyscf.pbc.scf import scfint
#import pyscf.pbc.scf.scfint as scfint
import pyscf.pbc.scf.chkfile


def get_ovlp(cell, kpt=np.zeros(3)):
    '''Get the overlap AO matrix.
    '''
    coords = pyscf.pbc.dft.gen_grid.gen_uniform_grids(cell)
    aoR = pyscf.pbc.dft.numint.eval_ao(cell, coords, kpt)
    ngs = len(aoR)

    s = (cell.vol/ngs) * np.dot(aoR.T.conj(), aoR)
    return s


def get_hcore(cell, kpt=np.zeros(3)):
    '''Get the core Hamiltonian AO matrix, following :func:`dft.rks.get_veff_`.
    '''
    hcore = get_t(cell, kpt)
    if cell.pseudo:
        hcore += ( get_pp(cell, kpt) + get_jvloc_G0(cell, kpt) )
    else:
        hcore += get_nuc(cell, kpt)

    return hcore


def get_t(cell, kpt=np.zeros(3)):
    '''Get the kinetic energy AO matrix.
    
    Due to `kpt`, this is evaluated in real space using orbital gradients.
    '''
    coords = pyscf.pbc.dft.gen_grid.gen_uniform_grids(cell)
    aoR = pyscf.pbc.dft.numint.eval_ao(cell, coords, kpt, isgga=True)
    ngs = aoR.shape[1]  # because we requested isgga, aoR.shape[0] = 4

    t = 0.5*(np.dot(aoR[1].T.conj(), aoR[1]) +
             np.dot(aoR[2].T.conj(), aoR[2]) +
             np.dot(aoR[3].T.conj(), aoR[3]))
    t *= (cell.vol/ngs)
    
    return t


# def get_t2(cell, kpt=np.zeros(3)):
#     '''Get the kinetic energy AO matrix.
    
#     Due to `kpt`, this is evaluated in real space using orbital gradients.

#     '''
#     coords = pyscf.pbc.dft.gen_grid.gen_uniform_grids(cell)
#     # aoR = pyscf.pbc.dft.numint.eval_ao(cell, coords, kpt, isgga=True)
#     # ngs = aoR.shape[1]  # because we requested isgga, aoR.shape[0] = 4
#     aoR = pyscf.pbc.dft.numint.eval_ao(cell, coords, kpt, isgga=False)
#     ngs = aoR.shape[0]  # because we requested isgga, aoR.shape[0] = 4

#     Gv = cell.Gv
#     G2 = np.einsum('gi,gi->g', Gv+kpt, Gv+kpt)

#     aoG = np.empty(aoR.shape, np.complex128)
#     TaoG = np.empty(aoR.shape, np.complex128)
#     nao = cell.nao_nr()
#     for i in range(nao):
#         aoG[:,i] = pyscf.pbc.tools.fft(aoR[:,i], cell.gs)
#         TaoG[:,i] = 0.5*G2*aoG[:,i]

#     t = np.dot(aoG.T.conj(), TaoG)
#     t *= (cell.vol/ngs**2)


#     # t = 0.5*(np.dot(aoR[1].T.conj(), aoR[1]) +
#     #          np.dot(aoR[2].T.conj(), aoR[2]) +
#     #          np.dot(aoR[3].T.conj(), aoR[3]))
#     # t *= (cell.vol/ngs)
    
#     return t


def get_nuc(cell, kpt=np.zeros(3)):
    '''Get the bare periodic nuc-el AO matrix, with G=0 removed.

    See Martin (12.16)-(12.21).
    '''
    coords = pyscf.pbc.dft.gen_grid.gen_uniform_grids(cell)
    aoR = pyscf.pbc.dft.numint.eval_ao(cell, coords, kpt)

    chargs = [cell.atom_charge(i) for i in range(cell.natm)]
    SI = cell.get_SI()
    coulG = tools.get_coulG(cell)
    vneG = -np.dot(chargs,SI) * coulG
    vneR = tools.ifft(vneG, cell.gs)

    vne = np.dot(aoR.T.conj(), vneR.reshape(-1,1)*aoR)
    return vne


def get_pp(cell, kpt=np.zeros(3)):
    '''Get the periodic pseudotential nuc-el AO matrix, with G=0 removed.
    '''
    coords = pyscf.pbc.dft.gen_grid.gen_uniform_grids(cell)
    aoR = pyscf.pbc.dft.numint.eval_ao(cell, coords, kpt)
    nao = cell.nao_nr() 

    SI = cell.get_SI()
    vlocG = pseudo.get_vlocG(cell)
    vpplocG = -np.sum(SI * vlocG, axis=0)
    
    # vpploc evaluated in real-space
    vpplocR = tools.ifft(vpplocG, cell.gs)
    vpploc = np.dot(aoR.T.conj(), vpplocR.reshape(-1,1)*aoR)

    # vppnonloc evaluated in reciprocal space
    aokplusG = np.empty(aoR.shape, np.complex128)
    for i in range(nao):
        aokplusG[:,i] = tools.fft(aoR[:,i]*np.exp(-1j*np.dot(kpt,coords.T)), 
                                  cell.gs)
    ngs = len(aokplusG)

    vppnl = np.zeros((nao,nao), dtype=np.complex128)
    hs, projGs = pseudo.get_projG(cell, kpt)
    for ia, [h_ia,projG_ia] in enumerate(zip(hs,projGs)):
        for l, h in enumerate(h_ia):
            nl = h.shape[0]
            for m in range(-l,l+1):
                SPG_lm_aoG = np.zeros((nl,nao), dtype=np.complex128)
                for i in range(nl):
                    SPG_lmi = SI[ia,:] * projG_ia[l][m][i]
                    SPG_lm_aoG[i,:] = np.einsum('g,gp->p', SPG_lmi.conj(), aokplusG)
                for i in range(nl):
                    for j in range(nl):
                        # Note: There is no (-1)^l here.
                        vppnl += h[i,j]*np.einsum('p,q->pq', 
                                                   SPG_lm_aoG[i,:].conj(), 
                                                   SPG_lm_aoG[j,:])
    vppnl *= (1./ngs**2)

    return vpploc + vppnl


def get_jvloc_G0(cell, kpt=np.zeros(3)):
    '''Get the (separately divergent) Hartree + Vloc G=0 contribution.
    '''
    return 1./cell.vol * np.sum(pseudo.get_alphas(cell)) * get_ovlp(cell, kpt)


def get_j(cell, dm, kpt=np.zeros(3)):
    '''Get the Coulomb (J) AO matrix.
    '''
    coords = pyscf.pbc.dft.gen_grid.gen_uniform_grids(cell)
    aoR = pyscf.pbc.dft.numint.eval_ao(cell, coords, kpt)
    ngs, nao = aoR.shape

    coulG = tools.get_coulG(cell)

    rhoR = pyscf.pbc.dft.numint.eval_rho(cell, aoR, dm)
    rhoG = tools.fft(rhoR, cell.gs)

    vG = coulG*rhoG
    vR = tools.ifft(vG, cell.gs)

    vj = (cell.vol/ngs) * np.dot(aoR.T.conj(), vR.reshape(-1,1)*aoR)
    return vj


def ewald(cell, ew_eta, ew_cut, verbose=logger.NOTE):
    '''Perform real (R) and reciprocal (G) space Ewald sum for the energy.

    Formulation of Martin, App. F2.

    Returns:
        float
            The Ewald energy consisting of overlap, self, and G-space sum.

    See Also:
        pyscf.pbc.gto.get_ewald_params
    '''
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(cell.stdout, verbose)

    chargs = [cell.atom_charge(i) for i in range(len(cell._atm))]
    coords = [cell.atom_coord(i) for i in range(len(cell._atm))]

    ewovrl = 0.

    # set up real-space lattice indices [-ewcut ... ewcut]
    ewxrange = range(-ew_cut[0],ew_cut[0]+1)
    ewyrange = range(-ew_cut[1],ew_cut[1]+1)
    ewzrange = range(-ew_cut[2],ew_cut[2]+1)
    ewxyz = cartesian_prod((ewxrange,ewyrange,ewzrange)).T

    # SLOW = True
    # if SLOW == True:
    #     ewxyz = ewxyz.T
    #     for ic, (ix, iy, iz) in enumerate(ewxyz):
    #         L = np.einsum('ij,j->i', cell._h, ewxyz[ic])

    #         # prime in summation to avoid self-interaction in unit cell
    #         if (ix == 0 and iy == 0 and iz == 0):
    #             print "L is", L
    #             for ia in range(cell.natm):
    #                 qi = chargs[ia]
    #                 ri = coords[ia]
    #                 #for ja in range(ia):
    #                 for ja in range(cell.natm):
    #                     if ja != ia:
    #                         qj = chargs[ja]
    #                         rj = coords[ja]
    #                         r = np.linalg.norm(ri-rj)
    #                         ewovrl += qi * qj / r * scipy.special.erfc(ew_eta * r)
    #         else:
    #             for ia in range(cell.natm):
    #                 qi = chargs[ia]
    #                 ri = coords[ia]
    #                 for ja in range(cell.natm):
    #                     qj=chargs[ja]
    #                     rj=coords[ja]
    #                     r=np.linalg.norm(ri-rj+L)
    #                     ewovrl += qi * qj / r * scipy.special.erfc(ew_eta * r)

    # # else:
    nx = len(ewxrange)
    ny = len(ewyrange)
    nz = len(ewzrange)
    Lall = np.einsum('ij,jk->ik', cell._h, ewxyz).reshape(3,nx,ny,nz)
    #exclude the point where Lall == 0
    Lall[:,ew_cut[0],ew_cut[1],ew_cut[2]] = 1e200
    Lall = Lall.reshape(3,nx*ny*nz)
    Lall = Lall.T

    for ia in range(cell.natm):
        qi = chargs[ia]
        ri = coords[ia]
        for ja in range(ia):
            qj = chargs[ja]
            rj = coords[ja]
            r = np.linalg.norm(ri-rj)
            ewovrl += 2 * qi * qj / r * scipy.special.erfc(ew_eta * r)

    for ia in range(cell.natm):
        qi = chargs[ia]
        ri = coords[ia]
        for ja in range(cell.natm):
            qj = chargs[ja]
            rj = coords[ja]
            r1 = ri-rj + Lall
            r = np.sqrt(np.einsum('ji,ji->j', r1, r1))
            ewovrl += (qi * qj / r * scipy.special.erfc(ew_eta * r)).sum()

    ewovrl *= 0.5

    # last line of Eq. (F.5) in Martin 
    ewself  = -1./2. * np.dot(chargs,chargs) * 2 * ew_eta / np.sqrt(np.pi)
    ewself += -1./2. * np.sum(chargs)**2 * np.pi/(ew_eta**2 * cell.vol)
    
    # g-space sum (using g grid) (Eq. (F.6) in Martin, but note errors as below)
    SI = cell.get_SI()
    ZSI = np.einsum("i,ij->j", chargs, SI)

    # Eq. (F.6) in Martin is off by a factor of 2, the
    # exponent is wrong (8->4) and the square is in the wrong place
    #
    # Formula should be
    #   1/2 * 4\pi / Omega \sum_I \sum_{G\neq 0} |ZS_I(G)|^2 \exp[-|G|^2/4\eta^2]
    # where
    #   ZS_I(G) = \sum_a Z_a exp (i G.R_a)
    # See also Eq. (32) of ewald.pdf at 
    #   http://www.fisica.uniud.it/~giannozz/public/ewald.pdf

    coulG = tools.get_coulG(cell)
    absG2 = np.einsum('gi,gi->g', cell.Gv, cell.Gv)

    ZSIG2 = np.abs(ZSI)**2
    expG2 = np.exp(-absG2/(4*ew_eta**2))
    JexpG2 = coulG*expG2
    ewgI = np.dot(ZSIG2,JexpG2)
    ewg = .5*np.sum(ewgI)
    ewg /= cell.vol

    #log.debug('Ewald components = %.15g, %.15g, %.15g', ewovrl, ewself, ewg)
    return ewovrl + ewself + ewg


#FIXME: project initial guess for k-point
def init_guess_by_chkfile(cell, chkfile_name, project=True):
    '''Read the HF results from checkpoint file, then project it to the
    basis defined by ``cell``

    Returns:
        Density matrix, 2D ndarray
    '''
    from pyscf.pbc.scf import addons
    chk_cell, scf_rec = pyscf.pbc.scf.chkfile.load_scf(chkfile_name)

    def fproj(mo):
        if project:
            return addons.project_mo_nr2nr(chk_cell, mo, cell)
        else:
            return mo
    if scf_rec['mo_coeff'].ndim == 2:
        mo = scf_rec['mo_coeff']
        mo_occ = scf_rec['mo_occ']
        dm = pyscf.scf.hf.make_rdm1(fproj(mo), mo_occ)
    else:  # UHF
        mo = scf_rec['mo_coeff']
        mo_occ = scf_rec['mo_occ']
        dm = pyscf.scf.hf.make_rdm1(fproj(mo[0]), mo_occ[0]) \
           + pyscf.scf.hf.make_rdm1(fproj(mo[1]), mo_occ[1])
    return dm


# TODO: Maybe should create PBC SCF class derived from pyscf.scf.hf.SCF, then
# inherit from that.
class RHF(pyscf.scf.hf.RHF):
    '''RHF class adapted for PBCs.

    Attributes:
        kpt : (3,) ndarray
            The AO k-point in Cartesian coordinates, in units of 1/Bohr.
        analytic_int : bool
            Whether to use analytic (libcint) integrals instead of grid-based. 
    '''
    def __init__(self, cell, kpt=None, analytic_int=None):
        if not cell._built:
            sys.stderr.write('Warning: cell.build() is not called in input\n')
            cell.build()
        self.cell = cell
        pyscf.scf.hf.RHF.__init__(self, cell)
        self.grids = pyscf.pbc.dft.gen_grid.UniformGrids(cell)

        if kpt is None:
            self.kpt = np.zeros(3)
        else:
            self.kpt = kpt

        if analytic_int is None:
            self.analytic_int = False
        else:
            self.analytic_int = True

        self._keys = self._keys.union(['cell', 'grids', 'kpt', 'analytic_int'])

    def dump_flags(self):
        pyscf.scf.hf.RHF.dump_flags(self)
        logger.info(self, '\n')
        logger.info(self, '******** PBC SCF flags ********')
        logger.info(self, 'Grid size = (%d, %d, %d)', 
                    self.cell.gs[0], self.cell.gs[1], self.cell.gs[2])

    def get_hcore(self, cell=None, kpt=None):
        if cell is None: cell = self.cell
        if kpt is None: kpt = self.kpt

        if self.analytic_int:
            logger.info(self, "Using analytic integrals")
            return scfint.get_hcore(cell, kpt)
        else:
            return get_hcore(cell, kpt)

    def get_ovlp(self, cell=None, kpt=None):
        if cell is None: cell = self.cell
        if kpt is None: kpt = self.kpt
        
        if self.analytic_int:
            logger.info(self, "Using analytic integrals")
            return scfint.get_ovlp(cell, kpt)
        else:
            return get_ovlp(cell, kpt)

    def get_j(self, cell=None, dm=None, hermi=1, kpt=None):
        if cell is None: cell = self.cell
        if dm is None: dm = self.make_rdm1()
        if kpt is None: kpt = self.kpt
        return get_j(cell, dm, kpt)

    def get_jk_(self, cell=None, dm=None, hermi=1, verbose=logger.DEBUG, kpt=None):
        '''Get Coulomb (J) and exchange (K) following :func:`scf.hf.RHF.get_jk_`.

        *Incore* version of Coulomb and exchange build only.
        Currently RHF always uses PBC AO integrals (unlike RKS), since
        exchange is currently computed by building PBC AO integrals.
        '''
        if cell is None: cell = self.cell
        if dm is None: dm = self.make_rdm1()
        if kpt is None: kpt = self.kpt
        
        log = logger.Logger
        if isinstance(verbose, logger.Logger):
            log = verbose
        else:
            log = logger.Logger(cell.stdout, verbose)

        log.debug('JK PBC build: incore only with PBC integrals')

        if self._eri is None:
            log.debug('Building PBC AO integrals')
            if kpt is not None and pyscf.lib.norm(kpt) > 1.e-15:
                raise RuntimeError("Non-zero k points not implemented for exchange")
            self._eri = ao2mo.get_ao_eri(cell)

        if np.iscomplexobj(dm) or np.iscomplexobj(self._eri):

            eri_re = np.ascontiguousarray(self._eri.real)
            eri_im = np.ascontiguousarray(self._eri.imag)
            
            dm_re = np.ascontiguousarray(dm.real)
            dm_im = np.ascontiguousarray(dm.imag)

            vj_rr, vk_rr = pyscf.scf.hf.dot_eri_dm(eri_re, dm_re, hermi)
            vj_ir, vk_ir = pyscf.scf.hf.dot_eri_dm(eri_im, dm_re, hermi)
            vj_ri, vk_ri = pyscf.scf.hf.dot_eri_dm(eri_re, dm_im, hermi)
            vj_ii, vk_ii = pyscf.scf.hf.dot_eri_dm(eri_im, dm_im, hermi)
            
            vj = vj_rr - vj_ii + 1j*(vj_ir + vj_ri)
            vk = vk_rr - vk_ii + 1j*(vk_ir + vk_ri)
            
        else:
            vj, vk = pyscf.scf.hf.dot_eri_dm(self._eri, dm, hermi)
        
        return vj, vk

    def energy_tot(self, dm=None, h1e=None, vhf=None):
        etot = self.energy_elec(dm, h1e, vhf)[0] + self.ewald_nuc()
        return etot.real
    
    def ewald_nuc(self, cell=None):
        if cell is None: cell = self.cell
        return ewald(cell, cell.ew_eta, cell.ew_cut, self.verbose)
        
    def get_band_fock_ovlp(self, fock, ovlp, band_kpt):
        '''Reconstruct Fock operator at a given 'band' k-point, not necessarily 
        in list of k-points.

        Returns:
            fock : (nao, nao) ndarray
            ovlp : (nao, nao) ndarray
        '''
        sinv = scipy.linalg.inv(ovlp)
        sinvFocksinv = np.dot(np.conj(sinv.T), np.dot(fock, sinv))

        # band_ovlp[p,q] = <p(0)|q(k)>
        band_ovlp = self.get_ovlp(self.cell, band_kpt)
        # Fb[p,q] = \sum_{rs} <p(k)|_r(0)> <r(0)|F|s(0)> <_s(0)|q(k)>
        Fb = np.dot(np.conj(band_ovlp.T), np.dot(sinvFocksinv, band_ovlp))

        return Fb, band_ovlp

    def init_guess_by_chkfile(self, chk=None, project=True):
        return init_guess_by_chkfile(self.cell, chk, project)
    def from_chk(self, chk=None, project=True):
        return self.init_guess_by_chkfile(chk, project)

