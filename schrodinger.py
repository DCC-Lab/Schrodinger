import numpy as np
import scipy.integrate as spi
from scipy.sparse import diags
from scipy.sparse.linalg import eigs
from findiff import FinDiff
import matplotlib.pyplot as plt
import pandas as pd
import unittest

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
    def __init__(self, matrix=None):
        if matrix is None:
            self.matrix = np.identity(n=len(self.x))
        else:
            self.matrix = matrix

        self._eigenvalues = None
        self._eigenvectors = None

    @property
    def x(self):
        return Vector.x

    @property
    def dx(self):
        """ This is the differential element dx for our x vector"""
        return Vector.x[1]-Vector.x[0]

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

    def __init__(self, v=None):
        if v is None:
            v = np.zeros((len(Vector.x),))

        self.v = v
        super().__init__( diags(self.v) )

    def add_to_plot(self, axis):
        axis.plot(self.x, np.real(self.v), label="Potential")

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

        return Potential(v)

    @classmethod
    def finite_well(cls, a, vo):
        """ This sets to potential to a finite well of width a and depth vo"""
        v = np.zeros((len(Vector.x),))

        for i, x in enumerate(Vector.x):
            if abs(x) >= abs(a)/2:
                v[i] = vo

        return Potential(v)

    @classmethod
    def harmonic_well(cls, omega=0.5):
        """ This sets to potential to a quadratic well of constant V(x) = omega * x^2 """
        x = Vector.x
        v = omega*x*x

        return Potential(v)

    @classmethod
    def harmonic_halfwell(cls, omega=0.5):
        """ This sets to potential to a quadratic half-well of constant V(x) = omega * x^2 """
        v = omega*Vector.x*Vector.x

        for i, x in enumerate(Vector.x):
            if x < 0:
                v[i] = INFINITY

        return Potential(v)

class Hamiltonian(Operator):
    def __init__(self, potential=None):
        super().__init__()
        if potential is None:
            self.V = Potential()
        else:
            self.V = potential

        self.matrix = ( -0.5 * D2_Dx2().matrix + self.V.matrix )


class TestVector(unittest.TestCase):

    def test_init(self):
        self.assertIsNotNone(Vector())

    def test_dxDefined(self):
        v = Vector()
        self.assertIsNotNone(v.dx)

    def test_null(self):
        v = Vector()
        self.assertTrue(v.norm2() == 0)

    def test_not_normalized(self):
        v = Vector()
        self.assertFalse(v.is_normalized())

    def test_DxDefined(self):
        v = Vector()
        result = D_Dx(v)
        self.assertIsNotNone(result)

    def test_DxRightType(self):
        v = Vector()
        result = D_Dx(v)
        self.assertIsNotNone(result)
        self.assertEqual(type(result), Vector)

        v = Wavefunction()
        result = D_Dx(v)
        self.assertIsNotNone(result)
        self.assertEqual(type(result), Wavefunction)

    def test_D2_Dx2Defined(self):
        v = Vector()
        result = D2_Dx2(v)
        self.assertIsNotNone(result)

    def test_D2_Dx2RightType(self):
        v = Vector()
        result = D2_Dx2(v)
        self.assertIsNotNone(result)
        self.assertEqual(type(result), Vector)

        v = Wavefunction()
        result = D2_Dx2(v)
        self.assertIsNotNone(result)
        self.assertEqual(type(result), Wavefunction)

class TestWavefunction(unittest.TestCase):
    def test_gaussian(self):
        v = Wavefunction.gaussian(sigma=1)
        self.assertTrue(len(v.matrix) > 0)
        self.assertEqual(len(v.matrix), len(v.x))

    def test_gaussian_normalize(self):
        v = Wavefunction.gaussian(sigma=1)
        v.normalize()
        self.assertEqual(v.norm2(), 1)

    def test_gaussian_show(self):
        v = Wavefunction.gaussian(sigma=1)
        v.show()

    def test_derivative_gaussian_show(self):
        v = Wavefunction.gaussian(sigma=1)
        v2 = D_Dx(v)
        v2.show()

class TestOperators(unittest.TestCase):
    def test_operator_init(self):
        self.assertIsNotNone(Operator())

    def test_operator_null(self):
        self.assertIsNotNone(Operator(matrix=[1,2,3]))

class TestPotential(unittest.TestCase):
    def test_potential_init(self):
        self.assertIsNotNone(Potential())

    def test_potential_show(self):
        Potential().show()
        Potential.harmonic_well(omega=1).show()
        Potential.harmonic_halfwell(omega=1).show()
        Potential.infinite_well(a=10).show()
        Potential.finite_well(a=10, vo=0.5).show()

class TestHamiltoninan(unittest.TestCase):
    def test_hamiltonian_init(self):
        self.assertIsNotNone(Hamiltonian())

    def test_hamiltonian_harmonic(self):
        h = Hamiltonian(Potential.harmonic_well())
        energies, states = h.eigenstates()
        plt.plot(states)
        plt.show()

    def test_infinite_well(self):
        h = Hamiltonian(Potential.infinite_well(a=10))
        energies, states = h.eigenstates()
        plt.plot(states)
        plt.show()


if __name__ == "__main__":
    unittest.main()

#
#
# # x = np.linspace(-10, 10, 1001)
# # sigma = complex(3)
# # # psi.psi = psi.d2_dx2.matrix(x.shape)*psi.psi
#
# # op = Operator()
# # print(d_dx(op))
# # print(d2_dx2(op))
#
#
# psi = Wavefunction()
# psi.set_to_gaussian(sigma=1)
# psi.normalize()
# psi.psi = d_dx(psi)
# print(psi.psi)
# psi.show()
#
# exit()
#
#
#
# print(psi.is_normalized())
# psi.show()
#
#
# H = Hamiltonian(xMin=-25, xMax=25, N=1000)
# print(type(H.d2_dx2.matrix(((10,10)))))
# exit()
# H.set_harmonic_well(omega=0.1)
# # H.set_finite_well(a=10, vo=1)
# H.compute_eigenstates(k=4)
# H.show_eigenstates()
# # H.show_energies()
