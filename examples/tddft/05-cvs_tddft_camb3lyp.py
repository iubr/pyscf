#!/usr/bin/env python
#
# Author: Iulia Emilia Brumboiu <iubr@umk.pl>
#

'''
An example of how to calculate X-ray absorption using TDDFT and 
the core-valence separation (CVS) approximation. The example shows
how to calculate the C 1s XAS of the ethylene molecule.

With CAM-B3LYP/6-31G, the first three singlet core-excited states are:

** Singlet excitation energies and oscillator strengths **
Excited State   1:    274.81228 eV      4.51 nm  f=0.0861
Excited State   2:    274.81375 eV      4.51 nm  f=0.0000
Excited State   3:    278.97499 eV      4.44 nm  f=0.0223
'''

import pyscf

mol = pyscf.M(
        atom = '''C     0.000000000     0.000000000    -0.663984000;
                  C     0.000000000     0.000000000     0.663984000;
                  H     0.000000000     0.919796000    -1.223061000;
                  H     0.000000000    -0.919796000    -1.223061000;
                  H     0.000000000     0.919796000     1.223061000;
                  H     0.000000000    -0.919796000     1.223061000;''',
        basis = '6-31G'
)

mf = mol.RKS()
mf.xc = "camb3lyp"
mf.run()

mytd = mf.TDDFT()

# To run a CVS calculation, you need to define the space of core orbitals
# from which electrons will be excited. Here, we define a CVS space containing
# the first two orbitlas which are the C 1s. 
mytd.cvs_space = [0, 1]
mytd.kernel()
mytd.analyze(verbose=3)
