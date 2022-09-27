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

class Vector:
    x = np.linspace(-10,10,1001)

    def __init__(self, values=None):
        if values is None:
            self.matrix = np.zeros(len(self.x))
        else:
            self.matrix = values

    @property
    def dx(self):
        """ This is the differential element dx for our x vector"""
        return self.x[1]-self.x[0]

    def normalize(self):
        norm2 = self.norm2()
        if norm2 != 0:
            self.matrix /= np.sqrt(norm2)
        else:
            raise ValueError("Vector is not normalizable because it is null")

    def norm2(self):
        return spi.trapezoid(np.conj(self.matrix) * self.matrix, x=self.x, dx=self.dx)

    def is_normalized(self):
        if abs(self.norm2() - 1.0).real < 1e-4:
            return True
        else:
            return False

class Wavefunction(Vector):
    def __init__(self, psi = None, label=r"$\psi$"):
        super().__init__()
        if psi is None:
            psi = np.zeros(len(self.x),dtype=complex)

        self.label = label
        self.matrix = np.array(psi)

    @classmethod
    def gaussian(cls, sigma, normalized=True):
        psi = Wavefunction(psi=np.exp(-cls.x*cls.x/sigma/sigma))

        if normalized:
            psi.normalize()

        return psi

    def add_to_plot(self, axis):
        axis.plot(self.x, np.real(self.matrix), label=self.label)

    def show(self):
        fig, axis = plt.subplots()
        self.add_to_plot(axis)
  
        axis.grid()
        axis.legend()
        axis.set_ylabel("Wavefunction [arb.u.]")
        axis.set_xlabel("Distance [arb.u]")
        axis.set_xlim(min(self.x), max(self.x))

        plt.show()

    
class Operator:
    def __init__(self, matrix=None, label=""):
        if matrix is None:
            self.matrix = np.identity(n=len(self.x))
        else:
            self.matrix = matrix

        self._eigenvalues = None
        self._eigenvectors = None
        self.label = label

    @property
    def x(self):
        return Vector.x

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

        ax.set_ylim(eMin-deltaE*0.5, eMax+deltaE*0.5)
  
        ax.grid()
        ax.legend()
        ax.set_ylabel("Energy [arb.u.]")
        ax.set_xlabel("Distance [arb.u]")
        ax.set_xlim(min(self.x), max(self.x))
        plt.show()

def D_Dx(vector=None):
    dx = Vector.x[1] - Vector.x[0]
    operator_matrix = FinDiff(0, dx, 1).matrix(Vector.x.shape)

    if vector is not None:
        theClass = type(vector)
        return theClass(operator_matrix * vector.matrix)
    else:
        return Operator(operator_matrix)

def D2_Dx2(vector=None):
    dx = Vector.x[1] - Vector.x[0]
    operator_matrix = FinDiff(0, dx, 2).matrix(Vector.x.shape)

    if vector is not None:
        theClass = type(vector)
        return theClass(operator_matrix * vector.matrix)
    else:
        return Operator(operator_matrix)

class Potential(Operator):
    """ The class describes several potential that we encounter in quantum mechanics.
    It makes use of the default x from Operator.x """

    def __init__(self, v=None, label=None):
        if v is None:
            v = np.zeros((len(Vector.x),))

        self.v = v
        super().__init__( diags(self.v), label=label )

    def add_to_plot(self, axis):
        axis.plot(self.x, np.real(self.v), label=self.label)

    def show(self):
        fig, axis = plt.subplots()
        self.add_to_plot(axis)

        axis.grid()
        axis.legend()
        axis.set_ylabel("Energy [arb.u.]")
        axis.set_xlabel("Distance [arb.u]")
        axis.set_xlim(min(self.x), max(self.x))

        plt.show()

    @classmethod
    def infinite_well(cls, a):
        """ This sets to potential to a infinite well of width a """
        v = np.zeros((len(Vector.x),))

        for i, x in enumerate(Vector.x):
            if abs(x) >= abs(a)/2:
                v[i]   = INFINITY

        return Potential(v, label="Infinite well")

    @classmethod
    def finite_well(cls, a, vo):
        """ This sets to potential to a finite well of width a and depth vo"""
        v = np.zeros((len(Vector.x),))

        for i, x in enumerate(Vector.x):
            if abs(x) >= abs(a)/2:
                v[i] = vo

        return Potential(v, label="Finite well of depth $V_o = {0}$".format(vo))

    @classmethod
    def harmonic_well(cls, omega=0.5):
        """ This sets to potential to a quadratic well of constant V(x) = omega * x^2 """
        x = Vector.x
        v = omega*x*x

        return Potential(v, label="Harmonic well")

    @classmethod
    def harmonic_halfwell(cls, omega=0.5):
        """ This sets to potential to a quadratic half-well of constant V(x) = omega * x^2 """
        v = omega*Vector.x*Vector.x

        for i, x in enumerate(Vector.x):
            if x < 0:
                v[i] = INFINITY

        return Potential(v, label="Harmonic half-well")

    @classmethod
    def delta_barrier(cls):
        """ This sets to potential to a quadratic half-well of constant V(x) = omega * x^2 """
        v = np.zeros((len(Vector.x),))

        for i, x in enumerate(Vector.x):
            if x >= 0:
                v[i] = INFINITY
                break

        return Potential(v, label="Delta barrier")

class Hamiltonian(Operator):
    def __init__(self, potential=None):
        super().__init__()
        if potential is None:
            self.V = Potential()
        else:
            self.V = potential

        self.matrix = ( -0.5 * D2_Dx2().matrix + self.V.matrix )

