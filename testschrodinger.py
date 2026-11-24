import os
import unittest

import numpy as np
import matplotlib

# The tests below call show() on purpose, to check that the figures can be
# built. Left alone they open blocking windows and the suite never finishes, so
# the backend is switched before pyplot is imported through schrodinger.
# Run with SHOW_FIGURES=1 to actually look at the figures.
SHOW_FIGURES = os.environ.get("SHOW_FIGURES", "") not in ("", "0", "false")
if not SHOW_FIGURES:
    matplotlib.use("Agg")

from scipy.optimize import brentq

from schrodinger import *


def finite_well_levels(a, vo, k=Ksch):
    """Energies of a finite square well, above the bottom of the well.

    Solves the usual transcendental equations, k*tan(k*a/2) = kappa for the even
    states and k/tan(k*a/2) = -kappa for the odd ones, so that the tests compare
    against something other than the code they are testing.
    """
    energies = []
    trial = np.linspace(1e-9, vo - 1e-9, 20001)

    def even(e):
        return np.sqrt(e / k) * np.tan(np.sqrt(e / k) * a / 2) - np.sqrt((vo - e) / k)

    def odd(e):
        return np.sqrt(e / k) / np.tan(np.sqrt(e / k) * a / 2) + np.sqrt((vo - e) / k)

    for equation in (even, odd):
        values = np.array([equation(e) for e in trial])
        for i in range(len(trial) - 1):
            if not (np.isfinite(values[i]) and np.isfinite(values[i + 1])):
                continue
            # A sign change across a pole of the tangent is not a root.
            if values[i] * values[i + 1] < 0 and abs(values[i] - values[i + 1]) < 10:
                energies.append(brentq(equation, trial[i], trial[i + 1]))

    return np.sort(energies)


class PlotTestCase(unittest.TestCase):
    """Keeps the figures from blocking, and the default x from leaking.

    Wavefunction.x is a class attribute: a test that changes it would change it
    for every test that runs after it.
    """

    def setUp(self):
        self.saved_x = Wavefunction.x
        self.saved_show = plt.show
        if not SHOW_FIGURES:
            plt.show = lambda *args, **kwargs: None

    def tearDown(self):
        Wavefunction.x = self.saved_x
        plt.show = self.saved_show
        plt.close("all")

    def state_curves(self, axis):
        """The lines of the states only.

        The potential is dashed, the energy levels are dotted and only span two
        points, so the states are the full length solid lines.
        """
        return [
            line
            for line in axis.lines
            if line.get_linestyle() == "-"
            and len(line.get_xdata()) == len(Wavefunction.x)
        ]


class TestWavefunction(PlotTestCase):
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

    def test_null_is_not_normalizable(self):
        with self.assertRaises(ValueError):
            Wavefunction().normalize()

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

class TestOperators(PlotTestCase):
    def test_operator_init(self):
        self.assertIsNotNone(Operator())

    def test_operator_null(self):
        self.assertIsNotNone(Operator(matrix=[1,2,3]))

class TestPotential(PlotTestCase):
    def test_potential_init(self):
        self.assertIsNotNone(Potential())

    def test_potential_show(self):
        Potential().show()
        Potential.harmonic_well(omega=1).show()
        Potential.harmonic_halfwell(omega=1).show()
        Potential.infinite_well(a=10).show()
        Potential.finite_well(a=10, vo=0.5).show()

class TestHamiltoninan(PlotTestCase):
    def test_hamiltonian_init(self):
        self.assertIsNotNone(Hamiltonian())

    def test_hamiltonian_harmonic(self):
        h = Hamiltonian(Potential.harmonic_well())
        energies, states = h.eigenstates()
        for state in states:
            plt.plot(state.x, np.real(state.matrix))
        plt.show()

    def test_infinite_well_states_plot(self):
        h = Hamiltonian(Potential.infinite_well(a=10))
        energies, states = h.eigenstates()
        for state in states:
            plt.plot(state.x, np.real(state.matrix))
        plt.show()

    def test_infinite_well(self):
        h = Hamiltonian(Potential.infinite_well(a=10))
        h.show_eigenstates()

    def test_delta_barrier(self):
        h = Hamiltonian(Potential.delta_barrier())
        with self.assertRaises(Exception):
            energies, states = h.eigenstates(k=2)


class TestDerivatives(PlotTestCase):
    """D2_Dx2 is the whole kinetic term, and nothing checked it computed a
    second derivative at all."""

    def setUp(self):
        super().setUp()
        Wavefunction.x = np.linspace(-10, 10, 1001)
        self.interior = slice(2, -2)  # away from the one sided stencils

    def test_second_derivative_of_a_parabola_is_two(self):
        """x^2 -> 2 everywhere, exactly, for a second order scheme."""
        x = Wavefunction.x
        second = D2_Dx2().matrix @ (x * x)

        self.assertLess(np.abs(second - 2.0).max(), 1e-6)

    def test_second_derivative_of_a_sine(self):
        """sin(kx) -> -k^2 sin(kx)."""
        x, k = Wavefunction.x, 1.3
        second = D2_Dx2().matrix @ np.sin(k * x)
        expected = -(k**2) * np.sin(k * x)

        error = np.abs(second - expected)[self.interior].max()
        self.assertLess(error / np.abs(expected).max(), 1e-3)

    def test_second_derivative_is_second_order_accurate(self):
        """Halving dx must divide the error by four, not by two."""
        k = 1.3
        errors = []
        for n in (401, 801):
            x = np.linspace(-10, 10, n)
            Wavefunction.x = x
            second = D2_Dx2().matrix @ np.sin(k * x)
            expected = -(k**2) * np.sin(k * x)
            errors.append(np.abs(second - expected)[self.interior].max())

        self.assertAlmostEqual(errors[0] / errors[1], 4.0, delta=0.2)

    def test_second_derivative_is_not_the_first_one_applied_twice(self):
        """Diff(0, dx) ** 2 must build the [1, -2, 1] stencil, not the square of
        the first derivative matrix, which is [1, 0, -2, 0, 1] and four times
        less accurate."""
        x, k = Wavefunction.x, 1.3
        expected = -(k**2) * np.sin(k * x)

        second = D2_Dx2().matrix @ np.sin(k * x)
        twice = (D_Dx().matrix @ D_Dx().matrix) @ np.sin(k * x)

        error_second = np.abs(second - expected)[self.interior].max()
        error_twice = np.abs(twice - expected)[self.interior].max()
        self.assertLess(error_second, 0.5 * error_twice)

    def test_second_derivative_applied_to_a_wavefunction(self):
        """The operator can also be applied directly to a Wavefunction."""
        x, k = Wavefunction.x, 1.3
        result = D2_Dx2(Wavefunction(psi=np.sin(k * x)))
        expected = -(k**2) * np.sin(k * x)

        self.assertEqual(type(result), Wavefunction)
        error = np.abs(np.real(result.matrix) - expected)[self.interior].max()
        self.assertLess(error / np.abs(expected).max(), 1e-3)

    def test_hamiltonian_uses_the_second_derivative(self):
        """With no potential, H must be exactly -Ksch d2/dx2."""
        kinetic = Hamiltonian(Potential()).matrix + Ksch * D2_Dx2().matrix

        self.assertEqual(np.abs(kinetic.todense()).max(), 0.0)


class TestPhysics(PlotTestCase):
    """Checks the Schrodinger equation itself, against analytical solutions.

    Without these, the suite passes with a hamiltonian that is wrong by a factor
    of two, or with Ksch left in eV*m^2 instead of eV*Angstrom^2.
    """

    def test_ksch_is_hbar_squared_over_two_m(self):
        """hbar^2/2m in eV*Angstrom^2, from hbar*c and the rest energy.

        Deliberately computed a different way than schrodinger.py does it, so a
        wrong unit conversion cannot cancel out on both sides.
        """
        hbar_c = 1973.269804  # eV*Angstrom
        electron_rest_energy = 510998.95  # eV

        self.assertAlmostEqual(
            Ksch, hbar_c**2 / (2 * electron_rest_energy), delta=1e-5
        )
        self.assertAlmostEqual(Ksch, 3.80998, delta=1e-4)

    def test_infinite_well_energies(self):
        """E_n = n^2 pi^2 hbar^2 / 2 m a^2 for a well of width a."""
        a = 10.0
        Wavefunction.x = np.linspace(-20, 20, 1001)

        energies, states = Hamiltonian(Potential.infinite_well(a=a)).eigenstates(k=3)
        energies = np.sort(energies)

        for n, energy in enumerate(energies, start=1):
            expected = n**2 * np.pi**2 * Ksch / a**2
            self.assertAlmostEqual(energy / expected, 1.0, delta=0.005)

    def test_finite_well_energies(self):
        """The bound states of a finite well, against the transcendental ones."""
        a, vo = 30.0, 3.0
        Wavefunction.x = np.linspace(-60, 60, 1201)

        energies, states = Hamiltonian(
            Potential.finite_well(a=a, vo=vo)
        ).eigenstates(k=3)
        energies = np.sort(energies) + vo  # measured from the bottom of the well

        expected = finite_well_levels(a=a, vo=vo)[:3]

        for energy, reference in zip(energies, expected):
            self.assertAlmostEqual(energy / reference, 1.0, delta=0.015)

    def test_finite_well_converges_with_dx(self):
        """A finer grid must get closer to the analytical answer."""
        a, vo = 30.0, 3.0
        reference = finite_well_levels(a=a, vo=vo)[0]

        errors = []
        for n in (801, 2401):
            Wavefunction.x = np.linspace(-60, 60, n)
            energies, states = Hamiltonian(
                Potential.finite_well(a=a, vo=vo)
            ).eigenstates(k=1)
            errors.append(abs(energies[0] + vo - reference))

        self.assertLess(errors[1], errors[0])

    def test_harmonic_well_is_evenly_spaced(self):
        """V = omega x^2 gives levels spaced by 2 sqrt(omega * Ksch).

        Note that omega is a coefficient in eV/Angstrom^2 here, it is not the
        angular frequency: hbar*omega is 2 sqrt(omega * Ksch), not omega.
        """
        omega = 0.5
        Wavefunction.x = np.linspace(-20, 20, 1001)

        energies, states = Hamiltonian(
            Potential.harmonic_well(omega=omega)
        ).eigenstates(k=3)
        energies = np.sort(energies)

        quantum = 2 * np.sqrt(omega * Ksch)
        self.assertAlmostEqual(energies[0] / (quantum / 2), 1.0, delta=0.005)
        self.assertAlmostEqual((energies[1] - energies[0]) / quantum, 1.0, delta=0.005)
        self.assertAlmostEqual((energies[2] - energies[1]) / quantum, 1.0, delta=0.005)


class TestShowEigenstates(PlotTestCase):
    """Checks what the figure actually contains, not just that it can be built."""

    def setUp(self):
        super().setUp()
        Wavefunction.x = np.linspace(-60, 60, 501)
        self.hamiltonian = Hamiltonian(Potential.finite_well(a=30, vo=3))

    def test_zero_of_a_state_is_its_own_energy(self):
        """Each wavefunction is drawn with its zero at E_i, not at zero."""
        self.hamiltonian.show_eigenstates(which=[0, 1, 2])
        energies, states = self.hamiltonian.eigenstates(k=3)

        curves = self.state_curves(plt.gcf().axes[0])
        self.assertEqual(len(curves), 3)

        for i, curve in enumerate(curves):
            # Far outside the well the wavefunction has died out, so what is
            # drawn there is the offset alone.
            self.assertAlmostEqual(curve.get_ydata()[0], energies.real[i], places=6)

    def test_vertical_range_shows_the_potential(self):
        """The well must be inside the axis, not off the top of it."""
        self.hamiltonian.show_eigenstates(which=[0, 1, 2])

        yMin, yMax = plt.gcf().axes[0].get_ylim()
        values = np.real(self.hamiltonian.potential.values)

        self.assertLessEqual(yMin, values.min())
        self.assertGreaterEqual(yMax, values.max())

    def test_amplitude_scales_the_states(self):
        """Doubling amplitude doubles the excursion of every state."""
        excursions = []
        for amplitude in (0.1, 0.2):
            self.hamiltonian.show_eigenstates(which=[0, 1, 2], amplitude=amplitude)
            curve = self.state_curves(plt.gcf().axes[0])[0].get_ydata()
            excursions.append(curve.max() - curve.min())
            plt.close("all")

        self.assertAlmostEqual(excursions[1] / excursions[0], 2.0, delta=0.05)

    def test_a_single_state_is_not_flat(self):
        """which=[0] used to scale the state by the spacing to its neighbour,
        which does not exist, so it was drawn as a flat line."""
        self.hamiltonian.show_eigenstates(which=[0])

        curve = self.state_curves(plt.gcf().axes[0])[0].get_ydata()
        yMin, yMax = plt.gcf().axes[0].get_ylim()

        self.assertGreater(curve.max() - curve.min(), 0.05 * (yMax - yMin))

    def test_largest_lobe_points_up(self):
        """eigs() returns an arbitrary sign, the figure must not."""
        self.hamiltonian.show_eigenstates(which=[0, 1, 2])
        energies, states = self.hamiltonian.eigenstates(k=3)

        for i, curve in enumerate(self.state_curves(plt.gcf().axes[0])):
            relative = curve.get_ydata() - energies.real[i]
            self.assertGreater(relative[np.argmax(np.abs(relative))], 0)

    def test_probability_is_positive(self):
        """|psi|^2 never goes below the energy of its state."""
        self.hamiltonian.show_eigenstates(which=[0, 1, 2], probability=True)
        energies, states = self.hamiltonian.eigenstates(k=3)

        for i, curve in enumerate(self.state_curves(plt.gcf().axes[0])):
            self.assertGreaterEqual(curve.get_ydata().min(), energies.real[i] - 1e-9)


class TestPotentialShapes(PlotTestCase):
    """The potentials that no test ever built, against their own definition."""

    def setUp(self):
        super().setUp()
        Wavefunction.x = np.linspace(-60, 60, 501)

    def test_finite_barrier(self):
        a, vo = 20.0, 2.0
        values = np.real(Potential.finite_barrier(a=a, vo=vo).values)

        inside = np.abs(Wavefunction.x) <= a / 2
        self.assertTrue(np.all(values[inside] == vo))
        self.assertTrue(np.all(values[~inside] == 0))

    def test_two_finite_wells(self):
        a, b, vo = 10.0, 6.0, 3.0
        values = np.real(Potential.two_finite_wells(a=a, b=b, vo=vo).values)

        distance = np.abs(Wavefunction.x)
        inside = (distance >= b / 2) & (distance <= b / 2 + a)
        self.assertTrue(np.all(values[inside] == -vo))
        self.assertTrue(np.all(values[~inside] == 0))

    def test_two_finite_wells_give_doublets(self):
        """Two identical wells give pairs of almost degenerate states, split
        only by the tunnelling through the barrier between them."""
        energies, states = Hamiltonian(
            Potential.two_finite_wells(a=10, b=6, vo=3)
        ).eigenstates(k=3)
        energies = np.sort(energies)

        splitting = energies[1] - energies[0]
        spacing = energies[2] - energies[1]
        self.assertLess(splitting, 0.1 * spacing)

    def test_delta_well(self):
        values = np.real(Potential.delta_well().values)

        self.assertEqual(np.count_nonzero(values), 1)
        self.assertEqual(values.min(), -INFINITY)


if __name__ == "__main__":
    unittest.main()
