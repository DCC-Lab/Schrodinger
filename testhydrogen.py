import unittest

import numpy as np

from hydrogen import *

"""
The excited states of hydrogen, sorted into orbitals. Needs PyTorch (the
block solver of schrodinger_gpu.py), on the CPU here so the tests run anywhere.
"""


class TestHydrogenOrbitals(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.orbitals, cls.grid = solve(n_max=3, n=96, device="cpu")

    def test_constants(self):
        self.assertAlmostEqual(BOHR, 0.5292, 3)
        self.assertAlmostEqual(RYDBERG, 13.606, 2)

    def test_every_orbital_is_found_once(self):
        names = [orbital.name for orbital in self.orbitals]
        self.assertEqual(
            names,
            ["1s", "2s", "2p z", "2p x", "2p y", "3s", "3p z", "3p x", "3p y",
             "3d z²", "3d xz", "3d yz", "3d x²−y²", "3d xy"],
        )

    def test_energies_of_excited_states(self):
        for orbital in self.orbitals:
            if orbital.n >= 2:
                self.assertAlmostEqual(
                    orbital.energy / orbital.exact_energy, 1, delta=0.05, msg=orbital.name
                )

    def test_angular_momentum(self):
        for orbital in self.orbitals:
            self.assertAlmostEqual(
                mean_l2(orbital.psi, self.grid),
                orbital.l * (orbital.l + 1),
                delta=0.15,
                msg=orbital.name,
            )

    def test_mean_radius(self):
        for orbital in self.orbitals:
            if orbital.n >= 2:
                self.assertAlmostEqual(
                    mean_radius(orbital.psi, self.grid) / orbital.exact_radius,
                    1,
                    delta=0.05,
                    msg=orbital.name,
                )

    def test_matrix_method_gives_the_same_orbitals(self):
        # The strategy of the 1D code (the matrix, then its eigenvalues), on a
        # small grid: same names, same energies as the block solver.
        matrix, grid = solve(n_max=2, half_width=10, n=40, method="matrix")
        block, grid = solve(n_max=2, half_width=10, n=40, method="torch", device="cpu")
        self.assertEqual([o.name for o in matrix], [o.name for o in block])
        for a, b in zip(matrix, block):
            self.assertAlmostEqual(a.energy, b.energy, 4)
            self.assertAlmostEqual(abs(np.sum(a.psi * b.psi) * grid.dv), 1, 3)

    def test_orbitals_are_orthonormal(self):
        psi = [orbital.psi for orbital in self.orbitals]
        self.assertTrue(
            np.allclose(overlaps(psi, psi, self.grid), np.eye(len(psi)), atol=1e-3)
        )

    def test_orbitals_point_along_the_axes(self):
        x, y, z = self.grid.mesh
        r2 = x**2 + y**2 + z**2
        by_name = {orbital.name: orbital.psi for orbital in self.orbitals}
        # <z^2 / r^2> is 3/5 for p z and 1/5 for p x: p z is along z
        mean = lambda psi, f: np.sum(psi**2 * f) * self.grid.dv
        self.assertAlmostEqual(mean(by_name["2p z"], z**2 / r2), 3 / 5, delta=0.02)
        self.assertAlmostEqual(mean(by_name["2p x"], x**2 / r2), 3 / 5, delta=0.02)
        self.assertAlmostEqual(mean(by_name["2p y"], y**2 / r2), 3 / 5, delta=0.02)
        # d xy has the sign of x*y, d x2-y2 the sign of x^2 - y^2 (near the nucleus)
        near = np.exp(-np.sqrt(r2))
        self.assertGreater(np.sum(by_name["3d xy"] * x * y * near), 0)
        self.assertGreater(np.sum(by_name["3d x²−y²"] * (x**2 - y**2) * near), 0)
        self.assertAlmostEqual(np.sum(by_name["3d xy"] * (x**2 - y**2) * near), 0, delta=1e-3)


if __name__ == "__main__":
    unittest.main()
