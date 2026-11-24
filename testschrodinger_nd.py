import os
import unittest

import numpy as np
import matplotlib

# Same convention as testschrodinger.py: no blocking windows unless asked.
SHOW_FIGURES = os.environ.get("SHOW_FIGURES", "") not in ("", "0", "false")
if not SHOW_FIGURES:
    matplotlib.use("Agg")

from findiff import Diff
from scipy.special import jn_zeros

from schrodinger import Hamiltonian, Potential, Wavefunction
from schrodinger_nd import *


class TestOperators(unittest.TestCase):
    def test_second_derivative_is_the_findiff_one_inside(self):
        """Only the first and last rows differ from what schrodinger.py uses."""
        n, dx = 21, 0.1
        ours = second_derivative(n, dx).toarray()
        theirs = (Diff(0, dx) ** 2).matrix((n,)).toarray()

        self.assertTrue(np.allclose(ours[1:-1], theirs[1:-1]))

    def test_laplacian_is_symmetric(self):
        grid = Grid(np.linspace(-1, 1, 7), np.linspace(-2, 2, 9), np.linspace(0, 1, 5))
        matrix = laplacian(grid)

        self.assertEqual(matrix.shape, (7 * 9 * 5, 7 * 9 * 5))
        self.assertEqual(abs(matrix - matrix.T).max(), 0)

    def test_laplacian_of_a_paraboloid(self):
        """The laplacian of x^2 + 2 y^2 + 3 z^2 is 2 + 4 + 6 = 12, away from the edges."""
        grid = Grid(
            np.linspace(-1, 1, 11), np.linspace(-2, 2, 13), np.linspace(0, 3, 15)
        )
        x, y, z = grid.mesh
        f = x**2 + 2 * y**2 + 3 * z**2

        result = (laplacian(grid) @ f.ravel()).reshape(grid.shape)

        self.assertTrue(np.allclose(result[1:-1, 1:-1, 1:-1], 12.0))

    def test_1d_grid_gives_the_1d_answer(self):
        """With a single axis, this must be schrodinger.py again."""
        saved_x = Wavefunction.x
        try:
            Wavefunction.x = np.linspace(-60, 60, 1201)
            expected, states = Hamiltonian(
                Potential.finite_well(a=30, vo=3)
            ).eigenstates(k=3)

            grid = Grid(Wavefunction.x)
            energies, states = HamiltonianND(
                PotentialND.finite_well(grid, widths=30, vo=3)
            ).eigenstates(k=3)
        finally:
            Wavefunction.x = saved_x

        for energy, reference in zip(energies, np.sort(expected)):
            self.assertAlmostEqual(energy, reference, delta=1e-6)


class Test2D(unittest.TestCase):
    def test_infinite_square_well(self):
        """E = (nx^2 + ny^2) pi^2 hbar^2 / 2 m a^2, with its degeneracies."""
        a = 10.0
        grid = Grid.cube(half_width=6, n=121, ndim=2)

        energies, states = HamiltonianND(
            PotentialND.infinite_well(grid, widths=a)
        ).eigenstates(k=6)

        expected = np.array([2, 5, 5, 8, 10, 10]) * np.pi**2 * Ksch / a**2
        for energy, reference in zip(energies, expected):
            self.assertAlmostEqual(energy / reference, 1.0, delta=0.002)

    def test_infinite_rectangular_well(self):
        """Different widths lift the degeneracy: E ~ nx^2/a^2 + ny^2/b^2."""
        a, b = 10.0, 15.0
        grid = Grid(np.linspace(-6, 6, 121), np.linspace(-9, 9, 181))

        energies, states = HamiltonianND(
            PotentialND.infinite_well(grid, widths=(a, b))
        ).eigenstates(k=4)

        levels = [
            nx**2 / a**2 + ny**2 / b**2 for nx in range(1, 5) for ny in range(1, 5)
        ]
        expected = np.sort(levels)[:4] * np.pi**2 * Ksch
        for energy, reference in zip(energies, expected):
            self.assertAlmostEqual(energy / reference, 1.0, delta=0.002)

    def test_states_are_normalized_and_orthogonal(self):
        grid = Grid.cube(half_width=6, n=81, ndim=2)
        energies, states = HamiltonianND(
            PotentialND.infinite_well(grid, widths=10)
        ).eigenstates(k=4)

        for i, psi in enumerate(states):
            for j, phi in enumerate(states):
                overlap = np.sum(psi * phi) * grid.dv
                self.assertAlmostEqual(overlap, 1.0 if i == j else 0.0, delta=1e-8)

    def test_states_vanish_in_the_walls(self):
        grid = Grid.cube(half_width=6, n=81, ndim=2)
        potential = PotentialND.infinite_well(grid, widths=10)
        energies, states = HamiltonianND(potential).eigenstates(k=2)

        for psi in states:
            self.assertEqual(np.abs(psi[potential.values >= INFINITY]).max(), 0)

    def test_separable_finite_well_is_the_sum_of_two_1d_wells(self):
        """V(x) + V(y) separates: every 2D energy is a sum of two 1D energies.

        This is not PotentialND.finite_well, whose corners make it non
        separable: it is a cross of two perpendicular 1D wells.
        """
        x = np.linspace(-30, 30, 121)
        well = np.where(np.abs(x) <= 10, -3.0, 0.0)

        e1, states = HamiltonianND(PotentialND(Grid(x), well)).eigenstates(k=2)

        grid = Grid(x, x)
        cross = well[:, None] + well[None, :]
        e2, states = HamiltonianND(PotentialND(grid, cross)).eigenstates(k=3)

        expected = [e1[0] + e1[0], e1[0] + e1[1], e1[1] + e1[0]]
        for energy, reference in zip(e2, expected):
            self.assertAlmostEqual(energy, reference, delta=1e-6)

    def test_harmonic_well(self):
        """E = (nx + ny + 1) * 2 sqrt(omega Ksch): 1, 2, 2, 3, 3, 3."""
        omega = 0.5
        grid = Grid.cube(half_width=12, n=161, ndim=2)

        energies, states = HamiltonianND(
            PotentialND.harmonic_well(grid, omega)
        ).eigenstates(k=6)

        quantum = 2 * np.sqrt(omega * Ksch)
        for energy, reference in zip(energies / quantum, [1, 2, 2, 3, 3, 3]):
            self.assertAlmostEqual(energy, reference, delta=0.01)

    def test_circular_well(self):
        """A disk is not separable in x and y: E = Ksch * (Bessel zero / R)^2.

        The round wall is drawn in staircase on a square grid, so this is the
        least accurate of the wells, and it converges only like dx.
        """
        radius = 5.0
        zeros = [jn_zeros(0, 1)[0], jn_zeros(1, 1)[0], jn_zeros(1, 1)[0]]
        expected = Ksch * np.array(zeros) ** 2 / radius**2

        errors = []
        for n in (121, 241):
            grid = Grid.cube(half_width=6, n=n, ndim=2)
            energies, states = HamiltonianND(
                PotentialND.round_well(grid, radius)
            ).eigenstates(k=3)
            errors.append(np.abs(energies / expected - 1).max())

        self.assertLess(errors[0], 0.02)
        self.assertLess(errors[1], errors[0])

    def test_show_eigenstates(self):
        saved_show = plt.show
        if not SHOW_FIGURES:
            plt.show = lambda *args, **kwargs: None
        try:
            grid = Grid.cube(half_width=6, n=61, ndim=2)
            h = HamiltonianND(PotentialND.infinite_well(grid, widths=10))
            h.show_eigenstates(k=4)
            h.show_eigenstates(k=4, probability=True)
            self.assertEqual(len(plt.gcf().axes), 6)  # 2 rows of 3, 2 are hidden
        finally:
            plt.show = saved_show
            plt.close("all")


class Test3D(unittest.TestCase):
    def test_infinite_cubic_well(self):
        """E = (nx^2 + ny^2 + nz^2) pi^2 hbar^2 / 2 m a^2: 3, 6 (x3), 9 (x3)."""
        a = 10.0
        grid = Grid.cube(half_width=6, n=49, ndim=3)

        energies, states = HamiltonianND(
            PotentialND.infinite_well(grid, widths=a)
        ).eigenstates(k=7)

        expected = np.array([3, 6, 6, 6, 9, 9, 9]) * np.pi**2 * Ksch / a**2
        for energy, reference in zip(energies, expected):
            self.assertAlmostEqual(energy / reference, 1.0, delta=0.005)

    def test_both_methods_agree(self):
        grid = Grid.cube(half_width=6, n=25, ndim=3)
        potential = PotentialND.infinite_well(grid, widths=10)

        direct, states = HamiltonianND(potential).eigenstates(k=4, method="shift-invert")
        lanczos, states = HamiltonianND(potential).eigenstates(k=4, method="lanczos")

        self.assertTrue(np.allclose(direct, lanczos, atol=1e-8))

    def test_harmonic_well_finds_every_degenerate_state(self):
        """E = (n + 3/2) * 2 sqrt(omega Ksch): 1.5, then 2.5 three times.

        Lanczos alone sometimes skips the third 2.5 and returns 3.5.
        """
        omega = 0.5
        grid = Grid.cube(half_width=10, n=49, ndim=3)

        energies, states = HamiltonianND(
            PotentialND.harmonic_well(grid, omega)
        ).eigenstates(k=4)

        quantum = 2 * np.sqrt(omega * Ksch)
        for energy, reference in zip(energies / quantum, [1.5, 2.5, 2.5, 2.5]):
            self.assertAlmostEqual(energy, reference, delta=0.03)

    def test_spherical_well(self):
        """E = Ksch * (pi / R)^2 for the 1s state, (4.4934 / R)^2 for the three 1p."""
        radius = 5.0
        grid = Grid.cube(half_width=6, n=61, ndim=3)

        energies, states = HamiltonianND(
            PotentialND.round_well(grid, radius)
        ).eigenstates(k=4)

        expected = Ksch * np.array([np.pi, 4.4934, 4.4934, 4.4934]) ** 2 / radius**2
        for energy, reference in zip(energies, expected):
            self.assertAlmostEqual(energy / reference, 1.0, delta=0.04)

    def test_coulomb_constant(self):
        """e^2 / 4 pi epsilon_0 = 14.40 eV*A, which gives the Bohr radius and the Rydberg."""
        self.assertAlmostEqual(Kcoulomb, 14.3996, delta=1e-3)
        self.assertAlmostEqual(2 * Ksch / Kcoulomb, 0.5292, delta=1e-4)
        self.assertAlmostEqual(Kcoulomb**2 / (4 * Ksch), 13.606, delta=1e-3)

    def test_coulomb_needs_a_grid_without_the_nucleus(self):
        with self.assertRaises(ValueError):
            PotentialND.coulomb(Grid.cube(half_width=8, n=41, ndim=3))

        PotentialND.coulomb(Grid.cube(half_width=8, n=40, ndim=3))
        PotentialND.coulomb(Grid.cube(half_width=8, n=41, ndim=3), softening=0.1)

    def test_hydrogen_atom(self):
        """E_n = -13.606 / n^2 eV: 1s, then 2p three times and 2s.

        The 1s state is only a Bohr radius wide, 0.53 A, about two grid steps
        here: it is the least accurate. The 2s state feels the walls of the box.
        """
        grid = Grid.cube(half_width=8, n=64, ndim=3)

        energies, states = HamiltonianND(PotentialND.coulomb(grid)).eigenstates(k=5)

        self.assertAlmostEqual(energies[0] / -13.606, 1.0, delta=0.06)
        for energy in energies[1:4]:
            self.assertAlmostEqual(energy / -3.4015, 1.0, delta=0.01)
        self.assertAlmostEqual(energies[4] / -3.4015, 1.0, delta=0.03)

    def test_hydrogen_like_ion_scales_as_charge_squared(self):
        """He+ on a grid twice as fine is hydrogen with every energy times 4."""
        hydrogen, states = HamiltonianND(
            PotentialND.coulomb(Grid.cube(half_width=6, n=40, ndim=3))
        ).eigenstates(k=2)
        helium, states = HamiltonianND(
            PotentialND.coulomb(Grid.cube(half_width=3, n=40, ndim=3), charge=2)
        ).eigenstates(k=2)

        self.assertTrue(np.allclose(helium, 4 * hydrogen, rtol=1e-6))

    def test_torch_method_agrees(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")

        grid = Grid.cube(half_width=8, n=64, ndim=3)
        lanczos, states = HamiltonianND(PotentialND.coulomb(grid)).eigenstates(k=5)
        block, states = HamiltonianND(PotentialND.coulomb(grid)).eigenstates(
            k=5, method="torch"
        )
        self.assertTrue(np.allclose(lanczos, block, atol=1e-4))

        potential = PotentialND.round_well(Grid.cube(half_width=6, n=41, ndim=3), 5.0)
        lanczos, states = HamiltonianND(potential).eigenstates(k=4)
        block, states = HamiltonianND(potential).eigenstates(k=4, method="torch")
        self.assertTrue(np.allclose(lanczos, block, atol=1e-4))
        self.assertEqual(np.abs(states[0][potential.values >= INFINITY]).max(), 0)

    def test_show_eigenstates(self):
        saved_show = plt.show
        if not SHOW_FIGURES:
            plt.show = lambda *args, **kwargs: None
        try:
            grid = Grid.cube(half_width=6, n=25, ndim=3)
            h = HamiltonianND(PotentialND.infinite_well(grid, widths=10))
            h.show_eigenstates(k=3)
        finally:
            plt.show = saved_show
            plt.close("all")


if __name__ == "__main__":
    unittest.main()
