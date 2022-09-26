import numpy as np
import scipy.integrate as spi
from scipy.sparse import diags
from scipy.sparse.linalg import eigs
from findiff import FinDiff
import matplotlib.pyplot as plt
import pandas as pd

""" 
Reference
https://medium.com/@mathcube7/two-lines-of-python-to-solve-the-schrödinger-equation-2bced55c2a0e
"""

INFINITY = 100000

class Operator:
    X = np.linspace(-10,10,1001)

    def __init__(self, x=None):
        if x is not None:
            self.x = x
        else:
            self.x = np.array(Operator.X)

        self.m = np.identity(n=len(self.x))

        self._eigenvalues = None
        self._eigenvectors = None

    @property
    def dx(self):
        """ This is the differential element dx for our x vector"""
        return self.x[1]-self.x[0]

    @property
    def d_dx(self):
        """ This is the differential operator d/dx in matrix form """
        return FinDiff(0, self.dx, 1) # component 0 of the array, dx, first derivative

    @property
    def d2_dx2(self):
        """ This is the differential operator d2/dx2 in matrix form """
        return FinDiff(0, self.dx, 2) # component 0 of the array, dx, second derivative

    def eigenstates(self, k=3, which='SR'):
        if self._eigenvalues is None or self._eigenvectors is None:
            self.compute_eigenstates(k=k, which=which)

        return self._eigenvalues, self._eigenvectors

    def compute_eigenstates(self, k=3, which='SR'):
        while k > 0:
            try:
                self._eigenvalues, self._eigenvectors = eigs( self.m , k=k, which=which)
                break                
            except np.linalg.LinAlgError as err:
                print(err)
                k -= 1
                if k == 0:
                    raise ValueError("No eigenstates found")

        self._eigenvalues = self._eigenvalues.real
        for eig in self._eigenvectors:
            if abs(max(eig.real))< abs(min(eig.real)):
                self._eigenvectors *= -1

    # def show_eigenstates(self, which=None):
    #     energies, eigenstates = self.eigenstates()
    #     eMin = min(energies.real)
    #     eMax = max(energies.real)
    #     deltaE = eMax - eMin

    #     fig,ax = plt.subplots()

    #     if which is None:
    #         which = range(eigenstates.shape[1])

    #     for i in which:
    #         if i == 0:
    #             delta = energies[1]-energies[0]
    #         elif i == eigenstates.shape[1]-1:
    #             delta = energies[i]-energies[i-1]
    #         else:
    #             deltaPlus = energies[i+1]-energies[i]
    #             deltaMinus = energies[i]-energies[i-1]
    #             delta = min(deltaPlus, deltaMinus)

    #         scaling = 0.5*delta/max(abs(eigenstates[:, i].real))
    #         ax.plot(self.x, eigenstates[:, i].real * scaling + energies[i], label=r'$\psi_{0}$'.format(i))
    #         print(energies[i], eigenstates[-1,i].real)

    #     ax.plot(self.x, self.V,'k--',label="Potential")
        
    #     ax.set_ylim(eMin-deltaE*0.5, eMax+deltaE*0.5)
  
    #     ax.grid()
    #     ax.legend()
    #     ax.set_ylabel("Energy [arb.u.]")
    #     ax.set_xlabel("Distance [arb.u]")
    #     ax.set_xlim(min(self.x), max(self.x))
    #     plt.show()


class Hamiltonian(Operator):
    def __init__(self, xMin=-10, xMax=10, N=1001):
        self.x = np.linspace(xMin, xMax, N)
        self.V = np.zeros((len(self.x),))
        self._energies = None
        self._eigenstates = None

    def set_infinite_well(self, a):
        """ This sets to potential to a infinite well of width a """
        self.V = np.zeros((len(self.x),))

        for i, x in enumerate(self.x):
            if abs(x) >= abs(a)/2:
                self.V[i]   = INFINITY

        self.compute_eigenstates()

    def set_finite_well(self, a, vo):
        """ This sets to potential to a finite well of width a and depth vo"""
        self.V = np.zeros((len(self.x),))

        for i, x in enumerate(self.x):
            if abs(x) >= abs(a)/2:
                self.V[i]   = vo

        self.compute_eigenstates()

    def set_harmonic_well(self, omega=0.5):
        """ This sets to potential to a quadratic well of constant V(x) = omega * x^2 """
        self.V = omega*self.x*self.x

        self.compute_eigenstates()

    def set_harmonic_halfwell(self, omega=0.5):
        """ This sets to potential to a quadratic half-well of constant V(x) = omega * x^2 """
        self.V = omega*self.x*self.x

        for i, x in enumerate(self.x):
            if x < 0:
                self.V[i] = INFINITY

        self.compute_eigenstates()

    def show_eigenstates(self, which=None):
        energies, eigenstates = self.eigenstates()
        eMin = min(energies.real)
        eMax = max(energies.real)
        deltaE = eMax - eMin

        fig,ax = plt.subplots()

        if which is None:
            which = range(eigenstates.shape[1])

        for i in which:
            if i == 0:
                delta = energies[1]-energies[0]
            elif i == eigenstates.shape[1]-1:
                delta = energies[i]-energies[i-1]
            else:
                deltaPlus = energies[i+1]-energies[i]
                deltaMinus = energies[i]-energies[i-1]
                delta = min(deltaPlus, deltaMinus)

            scaling = 0.5*delta/max(abs(eigenstates[:, i].real))
            ax.plot(self.x, eigenstates[:, i].real * scaling + energies[i], label=r'$\psi_{0}$'.format(i))
            print(energies[i], eigenstates[-1,i].real)

        ax.plot(self.x, self.V,'k--',label="Potential")
        
        ax.set_ylim(eMin-deltaE*0.5, eMax+deltaE*0.5)
  
        ax.grid()
        ax.legend()
        ax.set_ylabel("Energy [arb.u.]")
        ax.set_xlabel("Distance [arb.u]")
        ax.set_xlim(min(self.x), max(self.x))
        plt.show()


class Wavefunction:
    def __init__(self, x = None, psi = None, energy = None, label=r"$\psi$"):
        if x is None:
            x = np.array([])
        if psi is None:
            psi = np.array([])

        self.label = label
        self.x = x
        self.psi = psi

    @property
    def dx(self):
        """ This is the differential element dx for our x vector"""
        return self.x[1]-self.x[0]

    @property
    def d_dx(self):
        """ This is the differential operator d/dx in matrix form """
        return FinDiff(0, self.dx, 1) # component 0 of the array, dx, first derivative

    @property
    def d2_dx2(self):
        """ This is the differential operator d2/dx2 in matrix form """
        return FinDiff(0, self.dx, 2) # component 0 of the array, dx, second derivative

    def normalize(self):
        sum_psi2 = self.norm2()
        self.psi /= np.sqrt(sum_psi2)

    def norm2(self):
        return spi.trapezoid( np.conj(self.psi) * self.psi, x=self.x, dx=self.x[1]-self.x[0])

    def is_normalized(self):
        if abs(self.norm2() - 1.0).real < 1e-4:
            return True
        else:
            return False

    def set_to_gaussian(self, x, sigma = 1):
        self.x = x
        self.psi = np.exp(-self.x*self.x/sigma/sigma)

    def show(self):
        fig, axis = plt.subplots()
        self.add_to_plot(axis)
  
        axis.grid()
        axis.legend()
        axis.set_ylabel("Wavefunction [arb.u.]")
        axis.set_xlabel("Distance [arb.u]")
        axis.set_xlim(min(self.x), max(self.x))

        plt.show()

    def add_to_plot(self, axis):
        axis.plot(self.x, np.real(self.psi), label=self.label)

op = Operator()
print(op.eigenstates())
exit()


x = np.linspace(-10, 10, 1001)
sigma = complex(3)
psi = Wavefunction()
psi.set_to_gaussian(x=x, sigma=sigma)
psi.normalize()
psi.psi = psi.d2_dx2.matrix(x.shape)*psi.psi

print(psi.is_normalized())
psi.show()


H = Hamiltonian(xMin=-25, xMax=25, N=1000)
print(type(H.d2_dx2.matrix(((10,10)))))
exit()
H.set_harmonic_well(omega=0.1)
# H.set_finite_well(a=10, vo=1)
H.compute_eigenstates(k=4)
H.show_eigenstates()
# H.show_energies()
