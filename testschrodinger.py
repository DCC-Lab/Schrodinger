import unittest
from schrodinger import *


class TestWavefunction(unittest.TestCase):
    def test_init(self):
        self.assertIsNotNone(Wavefunction())

    def test_dxDefined(self):
        v = Wavefunction()
        self.assertIsNotNone(v.dx)

    def test_null(self):
        v = Wavefunction()
        self.assertTrue(v.norm2() == 0)

    def test_not_normalized(self):
        v = Wavefunction()
        self.assertFalse(v.is_normalized())

    def test_DxDefined(self):
        v = Wavefunction()
        result = D_Dx(v)
        self.assertIsNotNone(result)

    def test_DxRightType(self):
        v = Wavefunction()
        result = D_Dx(v)
        self.assertIsNotNone(result)
        self.assertEqual(type(result), Wavefunction)

        v = Wavefunction()
        result = D_Dx(v)
        self.assertIsNotNone(result)
        self.assertEqual(type(result), Wavefunction)

    def test_D2_Dx2Defined(self):
        v = Wavefunction()
        result = D2_Dx2(v)
        self.assertIsNotNone(result)

    def test_D2_Dx2RightType(self):
        v = Wavefunction()
        result = D2_Dx2(v)
        self.assertIsNotNone(result)
        self.assertEqual(type(result), Wavefunction)

        v = Wavefunction()
        result = D2_Dx2(v)
        self.assertIsNotNone(result)
        self.assertEqual(type(result), Wavefunction)

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
        for state in states:
            plt.plot(state.x, state.matrix)
        plt.show()

    def test_infinite_well(self):
        h = Hamiltonian(Potential.infinite_well(a=10))
        energies, states = h.eigenstates()
        for state in states:
            plt.plot(state.x, state.matrix)
        plt.show()

    def test_infinite_well(self):
        h = Hamiltonian(Potential.infinite_well(a=10))
        h.show_eigenstates()

    def test_delta_barrier(self):
        h = Hamiltonian(Potential.delta_barrier())
        with self.assertRaises(Exception):
            energies, states = h.eigenstates(k=2)


if __name__ == "__main__":
    unittest.main()
