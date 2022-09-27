import numpy as np
import scipy.integrate as spi
from scipy.sparse import diags
from scipy.sparse.linalg import eigs
from findiff import FinDiff
import matplotlib.pyplot as plt

""" 
Reference
https://medium.com/@mathcube7/two-lines-of-python-to-solve-the-schrödinger-equation-2bced55c2a0e
"""

INFINITY = 100000

class Wavefunction:
    x = np.linspace(-10,10,1001)

    def __init__(self, psi = None, label=r"$\psi$"):
        super().__init__()
        if psi is None:
            psi = np.zeros(len(self.x),dtype=complex)

        self.label = label
        self.matrix = np.array(psi, dtype=complex)

    @property
    def dx(self):
        """ This is the differential element dx for our x vector"""
        return self.x[1]-self.x[0]

    def normalize(self):
        norm2 = self.norm2()
        if norm2 != 0:
            self.matrix /= np.sqrt(norm2)
        else:
            raise ValueError("Wavefunction is not normalizable because it is null")

    def norm2(self):
        return spi.trapezoid(np.abs(self.matrix) **2, x=self.x, dx=self.dx)

    def is_normalized(self):
        if abs(self.norm2() - 1.0).real < 1e-4:
            return True
        else:
            return False

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
        return Wavefunction.x

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
                eigenvalues, eigenvectors = eigs( self.matrix , k=k, which=which)
                break                
            except np.linalg.LinAlgError as err:
                print(err)
                k -= 1
                if k == 0:
                    raise ValueError("No eigenstates found")

        self._eigenvalues = eigenvalues.real
        self._eigenvectors = []
        for i in range(eigenvectors.shape[1]):
            self._eigenvectors.append(Wavefunction(psi=eigenvectors[:,i], label=r"$\psi_{{{0}}}$".format(i)))

def D_Dx(vector=None):
    dx = Wavefunction.x[1] - Wavefunction.x[0]
    operator_matrix = FinDiff(0, dx, 1).matrix(Wavefunction.x.shape)


    if vector is not None:
        theClass = type(vector)
        return theClass(operator_matrix * vector.matrix)
    else:
        return Operator(operator_matrix)


def D2_Dx2(vector=None):
    dx = Wavefunction.x[1] - Wavefunction.x[0]
    operator_matrix = FinDiff(0, dx, 2).matrix(Wavefunction.x.shape)

    if vector is not None:
        theClass = type(vector)
        return theClass(operator_matrix * vector.matrix)
    else:
        return Operator(operator_matrix)

class Potential(Operator):
    """ The class describes several potential that we encounter in quantum mechanics.
    It makes use of the default x from Operator.x """

    def __init__(self, values=None, label=None):
        if values is None:
            values = np.zeros((len(Wavefunction.x),))

        self.values = values
        super().__init__( diags(self.values), label=label )

    def add_to_plot(self, axis):
        axis.plot(self.x, np.real(self.values), "k--", label=self.label)

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
        v = np.zeros((len(Wavefunction.x),))

        for i, x in enumerate(Wavefunction.x):
            if abs(x) >= abs(a)/2:
                v[i]   = INFINITY

        return Potential(v, label="Infinite well")

    @classmethod
    def finite_well(cls, a, vo):
        """ This sets to potential to a finite well of width a and depth vo"""
        v = np.zeros((len(Wavefunction.x),))

        for i, x in enumerate(Wavefunction.x):
            if abs(x) >= abs(a)/2:
                v[i] = vo

        return Potential(v, label="Finite well of depth $V_o = {0}$".format(vo))

    @classmethod
    def harmonic_well(cls, omega=0.5):
        """ This sets to potential to a quadratic well of constant V(x) = omega * x^2 """
        x = Wavefunction.x
        v = omega*x*x

        return Potential(v, label="Harmonic well")

    @classmethod
    def harmonic_halfwell(cls, omega=0.5):
        """ This sets to potential to a quadratic half-well of constant V(x) = omega * x^2 """
        v = omega*Wavefunction.x*Wavefunction.x

        for i, x in enumerate(Wavefunction.x):
            if x < 0:
                v[i] = INFINITY

        return Potential(v, label="Harmonic half-well")

    @classmethod
    def delta_barrier(cls):
        """ This sets to potential to a quadratic half-well of constant V(x) = omega * x^2 """
        v = np.zeros((len(Wavefunction.x),))

        for i, x in enumerate(Wavefunction.x):
            if x >= 0:
                v[i] = INFINITY
                break

        return Potential(v, label="Delta barrier")

class Hamiltonian(Operator):
    def __init__(self, potential=None):
        super().__init__()
        if potential is None:
            self.potential = Potential()
        else:
            self.potential = potential

        self.matrix = ( -0.5 * D2_Dx2().matrix + self.potential.matrix )

    def show_eigenstates(self, which=None, probability=False):
        if which is not None:
            k = max(which)+1
        else:
            k = 3

        energies, eigenstates = self.eigenstates(k=k)

        eMin = min(energies.real)
        eMax = max(energies.real)
        deltaE = eMax - eMin

        fig,ax = plt.subplots()

        if which is None:
            which = range(len(eigenstates))

        for i in which:
            if i == 0:
                delta = energies[1]-energies[0]
            elif i == len(eigenstates)-1:
                delta = energies[i]-energies[i-1]
            else:
                deltaPlus = energies[i+1]-energies[i]
                deltaMinus = energies[i]-energies[i-1]
                delta = min(deltaPlus, deltaMinus)

            if probability:
                scaling = 0.5*delta/max(abs(eigenstates[i].matrix)**2)
                ax.plot(self.x, np.abs(eigenstates[i].matrix)**2 * scaling + energies[i], label=r'$\psi_{0}$'.format(i))
            else:
                scaling = 0.5*delta/max(abs(eigenstates[i].matrix.real))
                ax.plot(self.x, eigenstates[i].matrix.real * scaling + energies[i], label=r'$\psi_{0}$'.format(i))
        self.potential.add_to_plot(ax)
        ax.set_ylim(eMin-deltaE*0.5, eMax+deltaE*0.5)
  
        ax.grid()
        ax.legend()
        ax.set_ylabel("Energy [arb.u.]")
        ax.set_xlabel("Distance [arb.u]")
        ax.set_xlim(min(self.x), max(self.x))
        plt.show()

if __name__ == "__main__":
    # Wavefunction.x = np.linspace(-100,100,4001)
    h = Hamiltonian(Potential.harmonic_well(omega=0.1))
    h.show_eigenstates(probability=False)

