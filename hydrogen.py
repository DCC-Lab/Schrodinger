import sys

import numpy as np

from schrodinger import Ksch
from schrodinger_nd import Grid, HamiltonianND, Kcoulomb, PotentialND

"""
The excited states of the hydrogen atom, from the 3D grid of schrodinger_nd.py,
sorted into the orbitals of the textbooks: 2p, 3d, 4f...

The solver returns the states in order of energy, and in hydrogen every state
of a level n has the same energy: n^2 of them (1, 4, 9, 16). Within such a
level any combination of the states is as good as another, and what comes out
of the solver is an arbitrary mixture: a "3d" pointing in no particular
direction, or even a bit of 3d with a bit of 3p.

To get the textbook orbitals back we use what the course says about
degenerate levels: look for an operator that commutes with H and diagonalize
it inside the level. Three of them, one after the other:

    L^2            separates the level into s, p, d, f     (l = 0, 1, 2, 3)
    Lz^2           separates each l into |m| = 0, 1, ... l
    cos(2 m phi)   separates the two states of a given |m| into the real
                   orbitals that go like cos(m phi) and sin(m phi)

Each step is a small matrix (16 x 16 at most for n = 4) of scalar products
between the states, <psi_i| A |psi_j>, and its eigenvectors tell us how to
combine the states.

On the grid, L = -i r x grad is written with centered differences, and
<psi_i| L^2 |psi_j> is the sum over x, y, z of <L_a psi_i | L_a psi_j>.

Units: Angstrom and eV, L in units of hbar.
"""

BOHR = 2 * Ksch / Kcoulomb  # 0.529 Angstrom
RYDBERG = Kcoulomb / (2 * BOHR)  # 13.606 eV

LETTERS = "spdfgh"

# The usual names of the real orbitals, by l, |m|, and cos or sin
NAMES = {
    (0, 0, "c"): "",
    (1, 0, "c"): "z",
    (1, 1, "c"): "x",
    (1, 1, "s"): "y",
    (2, 0, "c"): "z²",
    (2, 1, "c"): "xz",
    (2, 1, "s"): "yz",
    (2, 2, "c"): "x²−y²",
    (2, 2, "s"): "xy",
    (3, 0, "c"): "z³",
    (3, 1, "c"): "xz²",
    (3, 1, "s"): "yz²",
    (3, 2, "c"): "z(x²−y²)",
    (3, 2, "s"): "xyz",
    (3, 3, "c"): "x(x²−3y²)",
    (3, 3, "s"): "y(3x²−y²)",
}


class Orbital:
    def __init__(self, n, l, m, kind, energy, psi):
        self.n, self.l, self.m, self.kind = n, l, m, kind
        self.energy = energy
        self.psi = psi

    @property
    def name(self):
        suffix = NAMES.get((self.l, self.m, self.kind), "m={0}{1}".format(self.m, self.kind))
        return "{0}{1} {2}".format(self.n, LETTERS[self.l], suffix).strip()

    @property
    def exact_energy(self):
        return -RYDBERG / self.n**2

    @property
    def exact_radius(self):
        """<r> of the exact hydrogen state [Angstrom]"""
        return BOHR * (3 * self.n**2 - self.l * (self.l + 1)) / 2


def angular_momentum(psi, grid):
    """(Lx psi, Ly psi, Lz psi) without the factor -i, by centered differences"""
    x, y, z = grid.mesh
    dx, dy, dz = np.gradient(psi, *grid.spacings)
    return y * dz - z * dy, z * dx - x * dz, x * dy - y * dx


def overlaps(a, b, grid):
    """The matrix of scalar products between two lists of functions"""
    a = np.reshape(a, (len(a), -1))
    b = np.reshape(b, (len(b), -1))
    return a @ b.T * grid.dv


def split(states, energies, matrix, values_of):
    """Diagonalize an operator inside a group of degenerate states.

    matrix is <psi_i| A |psi_j> for the states of the group. Returns a list of
    (value, states, energies), one per distinct eigenvalue of A, where
    values_of() turns the eigenvalues into the integer that labels them.
    """
    eigenvalues, c = np.linalg.eigh((matrix + matrix.T) / 2)
    labels = values_of(eigenvalues)
    flat = np.reshape(states, (len(states), -1))
    new_states = (c.T @ flat).reshape(np.shape(states))
    # The energy of a combination of states: sum of |c|^2 E, exact when the
    # states are truly degenerate, and a very good average when the grid has
    # split them slightly.
    new_energies = (c**2).T @ np.asarray(energies)

    groups = []
    for label in sorted(set(labels)):
        chosen = [i for i, value in enumerate(labels) if value == label]
        groups.append((label, new_states[chosen], new_energies[chosen], eigenvalues[chosen]))
    return groups


def hydrogen_orbitals(energies, states, grid, charge=1):
    """Sort the states of a Coulomb potential into the textbook orbitals.

    Returns a list of Orbital, in the order 1s, 2s, 2p z, 2p x, 2p y, 3s...
    The last level must be complete (5, 14, 30 states for n = 2, 3, 4):
    an incomplete level is left out.
    """
    energies = np.asarray(energies)
    x, y, z = grid.mesh
    phi = np.arctan2(y, x)

    # Which level: E = -Rydberg * charge^2 / n^2
    levels = np.rint(charge * np.sqrt(RYDBERG / np.abs(energies))).astype(int)

    orbitals = []
    for n in sorted(set(levels)):
        chosen = np.flatnonzero(levels == n)
        if len(chosen) != n**2:
            continue
        group = np.array([states[i] for i in chosen])

        # L^2: the sum over the three axes of <L_a psi_i | L_a psi_j>
        l2 = np.zeros((len(group), len(group)))
        for axis in range(3):
            l_psi = [angular_momentum(psi, grid)[axis] for psi in group]
            l2 += overlaps(l_psi, l_psi, grid)
        # eigenvalues are l (l + 1)
        to_l = lambda values: np.rint((np.sqrt(1 + 4 * np.abs(values)) - 1) / 2).astype(int)

        for l, l_states, l_energies, _ in split(group, energies[chosen], l2, to_l):
            lz_psi = [angular_momentum(psi, grid)[2] for psi in l_states]
            lz2 = overlaps(lz_psi, lz_psi, grid)
            to_m = lambda values: np.rint(np.sqrt(np.abs(values))).astype(int)

            for m, m_states, m_energies, _ in split(l_states, l_energies, lz2, to_m):
                if m == 0:
                    pairs = [("c", m_states[0], m_energies[0])]
                else:
                    weighted = [np.cos(2 * m * phi) * psi for psi in m_states]
                    cos2m = overlaps(m_states, weighted, grid)
                    # +1/2 for cos(m phi), -1/2 for sin(m phi)
                    halves = split(m_states, m_energies, cos2m, np.sign)
                    pairs = [
                        ("c" if sign > 0 else "s", s[0], e[0])
                        for sign, s, e, _ in reversed(halves)
                    ]

                for kind, psi, energy in pairs:
                    # The sign is arbitrary: we take psi > 0 near the nucleus
                    # where the textbook function is, that is where
                    # z^(l-m) cos(m phi) or sin(m phi) is positive.
                    angle = np.cos(m * phi) if kind == "c" else np.sin(m * phi)
                    shape = z ** ((l - m) % 2) * angle * np.exp(-np.sqrt(x**2 + y**2 + z**2) / BOHR)
                    if np.sum(psi * shape) < 0:
                        psi = -psi
                    psi = psi / np.sqrt(np.sum(psi**2) * grid.dv)
                    orbitals.append(Orbital(n, l, m, kind, energy, psi))

    return orbitals


def mean_radius(psi, grid):
    x, y, z = grid.mesh
    return np.sum(psi**2 * np.sqrt(x**2 + y**2 + z**2)) * grid.dv


def mean_l2(psi, grid):
    return sum(np.sum(component**2) for component in angular_momentum(psi, grid)) * grid.dv


def solve(n_max=3, half_width=None, n=128, method="torch", device=None):
    """Every orbital of hydrogen up to the level n_max.

    The box must hold the largest state: its radius grows like n^2 (the 4s
    state still has 1% of its probability beyond 21 Angstrom).

    method="matrix" is the strategy of schrodinger.py, unchanged: H is written
    as a (sparse) matrix and scipy is asked for its lowest eigenvalues.
    method="torch" is the block solver of schrodinger_gpu.py, which never
    writes the matrix. Both give the same energies; see the comparison in
    the README.
    """
    if half_width is None:
        half_width = {1: 6, 2: 10, 3: 20, 4: 30}.get(n_max, 2 * n_max**2)
    grid = Grid.cube(half_width=half_width, n=n, ndim=3)
    k = sum(level**2 for level in range(1, n_max + 1))

    if method == "matrix":
        h = HamiltonianND(PotentialND.coulomb(grid))
        energies, states = h.eigenstates(k=k, method="lanczos")
    else:
        from schrodinger_gpu import GPUSolver, best_device

        h = HamiltonianND.__new__(HamiltonianND)  # the sparse matrix is not needed
        h.potential, h.grid = PotentialND.coulomb(grid), grid
        solver = GPUSolver(h, device=device or best_device())
        energies, states = solver.eigenstates(k=k)
    return hydrogen_orbitals(energies, states, grid), grid


if __name__ == "__main__":
    n_max = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    method = sys.argv[3] if len(sys.argv) > 3 else "torch"  # or "matrix"
    orbitals, grid = solve(n_max=n_max, n=n, method=method)

    print("Hydrogen on {0}^3 points, dx = {1:.3f} A".format(n, grid.spacings[0]))
    print("{0:<16}{1:>9}{2:>9}{3:>8}{4:>9}{5:>8}{6:>8}".format(
        "orbital", "E [eV]", "exact", "<L^2>", "l(l+1)", "<r> [A]", "exact"))
    for orbital in orbitals:
        print("{0:<16}{1:9.3f}{2:9.3f}{3:8.3f}{4:9d}{5:8.3f}{6:8.3f}".format(
            orbital.name,
            orbital.energy,
            orbital.exact_energy,
            mean_l2(orbital.psi, grid),
            orbital.l * (orbital.l + 1),
            mean_radius(orbital.psi, grid),
            orbital.exact_radius,
        ))
