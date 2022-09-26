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
    x = np.linspace(-10,10,1001)

    def __init__(self):
        self.matrix = np.identity(n=len(self.x))

        self._eigenvalues = None
        self._eigenvectors = None

    @property
    def dx(self):
        """ This is the differential element dx for our x vector"""
        return self.x[1]-self.x[0]

    def eigenstates(self, k=3, which='SR'):
        if self._eigenvalues is None or self._eigenvectors is None:
            self.compute_eigenstates(k=k, which=which)

        return self._eigenvalues, self._eigenvectors

    def compute_eigenstates(self, k=3, which='SR'):
        while k > 0:
            try:
                self._eigenvalues, self._eigenvectors = eigs( self.matrix , k=k, which=which)
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

class Potential(Operator):

    def set_infinite_well(self, a):
        """ This sets to potential to a infinite well of width a """
        self.matrix = np.zeros((len(self.x),))

        for i, x in enumerate(self.x):
            if abs(x) >= abs(a)/2:
                self.matrix[i]   = INFINITY

    def set_finite_well(self, a, vo):
        """ This sets to potential to a finite well of width a and depth vo"""
        self.matrix = np.zeros((len(self.x),))

        for i, x in enumerate(self.x):
            if abs(x) >= abs(a)/2:
                self.matrix[i]   = vo

    def set_harmonic_well(self, omega=0.5):
        """ This sets to potential to a quadratic well of constant V(x) = omega * x^2 """
        self.matrix = omega*self.x*self.x

    def set_harmonic_halfwell(self, omega=0.5):
        """ This sets to potential to a quadratic half-well of constant V(x) = omega * x^2 """
        self.matrix = omega*self.x*self.x

        for i, x in enumerate(self.x):
            if x < 0:
                self.matrix[i] = INFINITY

class Hamiltonian(Operator):
    def __init__(self):
        self.V = np.zeros((len(self.x),))

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
    def __init__(self, psi = None, energy = None, label=r"$\psi$"):
        if psi is None:
            psi = np.array([])

        self.label = label
        self.psi = psi
        self.x = Operator.x

    @property
    def dx(self):
        """ This is the differential element dx for our x vector"""
        return self.x[1]-self.x[0]

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

    def set_to_gaussian(self, sigma = 1):
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


def d_dx(vector_or_operator):
    D = FinDiff(0, vector_or_operator.dx, 1)
    if type(vector_or_operator) == Operator:
        return D*vector_or_operator.m # component 0 of the array, dx, first derivative
    elif type(vector_or_operator) == Wavefunction:
        return D*vector_or_operator.psi # component 0 of the array, dx, first derivative
    else:
        raise ValueError("Wrong type")

def d2_dx2(vector_or_operator):
    """ This is the differential operator d2/dx2 in matrix form """
    D2 = FinDiff(0, vector_or_operator.dx, 2)
    if type(vector_or_operator) == Operator:
        return D2*vector_or_operator.m # component 0 of the array, dx, second derivative
    elif type(vector_or_operator) == Wavefunction:
        return D2*vector_or_operator.psi # component 0 of the array, dx, second derivative
    else:
        raise ValueError("Wrong type")

# x = np.linspace(-10, 10, 1001)
# sigma = complex(3)
# # psi.psi = psi.d2_dx2.matrix(x.shape)*psi.psi

# op = Operator()
# print(d_dx(op))
# print(d2_dx2(op))


psi = Wavefunction()
psi.set_to_gaussian(sigma=1)
psi.normalize()
psi.psi = d_dx(psi)
print(psi.psi)
psi.show()

exit()



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
