import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, identity, kron
from scipy.sparse.linalg import eigsh

from scipy.constants import elementary_charge, epsilon_0

from schrodinger import INFINITY, Ksch

# e^2 / (4 pi epsilon_0) in eV*Angstrom: the Coulomb energy of two elementary
# charges 1 Angstrom apart is 14.4 eV.
Kcoulomb = elementary_charge / (4 * np.pi * epsilon_0) / 1e-10

"""
The Schrodinger equation in 2D and 3D, with the same matrix strategy as
schrodinger.py.

In 1D, the wavefunction is a vector of N values and d2/dx2 is an N x N matrix.
In 2D, the wavefunction is a table psi[i, j] of Nx x Ny values. We flatten it
into a single vector of Nx*Ny values, and the operators are then matrices of
(Nx*Ny) x (Nx*Ny). The Kronecker product builds them from the 1D ones:

    d2/dx2  ->  D2x (x) Iy        acts on index i, leaves j alone
    d2/dy2  ->  Ix (x) D2y        acts on index j, leaves i alone

    Laplacian = D2x (x) Iy  +  Ix (x) D2y

and the potential is still a diagonal matrix, with V(x_i, y_j) flattened the
same way. In 3D there are three terms, each with two identities. Nothing else
changes: H = -Ksch * Laplacian + V, and we ask for the lowest eigenvalues.

The price is the size: 201 points per axis is a matrix of 201 in 1D, 40 401 in
2D and 8 120 601 in 3D. The matrices are sparse (5 non-zero values per row in
2D, 7 in 3D) so they fit in memory, but 3D grids must stay modest.

Units are the same as in schrodinger.py: Angstrom and eV, for an electron.
"""


class Grid:
    """The points where the wavefunction is known: one axis per dimension.

    Grid(x) is 1D, Grid(x, y) is 2D, Grid(x, y, z) is 3D. Each axis is evenly
    spaced, but the axes need not have the same spacing or length.
    """

    def __init__(self, *axes):
        self.axes = [np.asarray(axis, dtype=float) for axis in axes]

    @classmethod
    def cube(cls, half_width, n, ndim):
        """The same axis, from -half_width to half_width in n points, ndim times."""
        return cls(*[np.linspace(-half_width, half_width, n)] * ndim)

    @property
    def ndim(self):
        return len(self.axes)

    @property
    def shape(self):
        return tuple(len(axis) for axis in self.axes)

    @property
    def spacings(self):
        return [axis[1] - axis[0] for axis in self.axes]

    @property
    def dv(self):
        """The element dx (1D), dx*dy (2D) or dx*dy*dz (3D)"""
        return float(np.prod(self.spacings))

    @property
    def mesh(self):
        """The coordinates of every point: X, Y (, Z), each of shape self.shape"""
        return np.meshgrid(*self.axes, indexing="ij")


def second_derivative(n, dx):
    """The 1D second derivative, as a symmetric matrix.

    This is the same (1, -2, 1)/dx^2 stencil that findiff gives schrodinger.py
    for every interior point. The first and last rows differ: findiff uses a
    one-sided formula there, which makes the matrix non symmetric. Here the
    stencil is simply cut, which amounts to saying that the wavefunction is
    zero just outside of the grid. The matrix is then symmetric, so the
    Hamiltonian is hermitian, as it should be, and we can use eigsh.
    """
    return diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(n, n)) / dx**2


def laplacian(grid):
    """d2/dx2 + d2/dy2 (+ d2/dz2), built from the 1D matrices by Kronecker products."""
    total = None
    for axis in range(grid.ndim):
        term = None
        for other in range(grid.ndim):
            n = grid.shape[other]
            if other == axis:
                factor = second_derivative(n, grid.spacings[other])
            else:
                factor = identity(n)
            term = factor if term is None else kron(term, factor, format="csr")
        total = term if total is None else total + term
    return total.tocsr()


class PotentialND:
    """A potential on a Grid: one value per point, in eV."""

    def __init__(self, grid, values=None, label=None):
        self.grid = grid
        if values is None:
            values = np.zeros(grid.shape)
        self.values = np.asarray(values, dtype=float)
        self.label = label

    @classmethod
    def infinite_well(cls, grid, widths):
        """A box with hard walls: widths is a (or (a, b), or (a, b, c))"""
        widths = np.broadcast_to(widths, (grid.ndim,))
        v = np.zeros(grid.shape)
        for coordinate, width in zip(grid.mesh, widths):
            v[np.abs(coordinate) >= abs(width) / 2] = INFINITY
        return cls(grid, v, label="Infinite well")

    @classmethod
    def finite_well(cls, grid, widths, vo):
        """A box of depth vo: the potential is -vo inside and 0 outside"""
        widths = np.broadcast_to(widths, (grid.ndim,))
        inside = np.ones(grid.shape, dtype=bool)
        for coordinate, width in zip(grid.mesh, widths):
            inside &= np.abs(coordinate) <= abs(width) / 2
        return cls(grid, np.where(inside, -abs(vo), 0.0), label="Finite well")

    @classmethod
    def round_well(cls, grid, radius, vo=None):
        """A disk (2D) or a sphere (3D). Hard walls if vo is None, else depth vo."""
        r = np.sqrt(sum(coordinate**2 for coordinate in grid.mesh))
        if vo is None:
            return cls(grid, np.where(r >= radius, INFINITY, 0.0), label="Round well")
        return cls(grid, np.where(r <= radius, -abs(vo), 0.0), label="Round well")

    @classmethod
    def harmonic_well(cls, grid, omega=0.5):
        """V = omega * r^2, same convention as Potential.harmonic_well in 1D"""
        r2 = sum(coordinate**2 for coordinate in grid.mesh)
        return cls(grid, omega * r2, label="Harmonic well")


    @classmethod
    def coulomb(cls, grid, charge=1, softening=0.0, center=None):
        """An electron attracted by a nucleus of the given charge (1: hydrogen).

        V = -charge * Kcoulomb / sqrt(r^2 + softening^2)

        With softening = 0 this is the true Coulomb potential, which is
        infinite at r = 0: the grid must then not have a point on the nucleus.
        An even number of points on a symmetric axis does that, the nucleus
        sits in the middle of a cell.

        With softening > 0 [Angstrom] this is the "soft Coulomb" potential,
        finite everywhere with a depth of charge * Kcoulomb / softening. It is
        the usual choice in 1D and 2D, where the true 1/r is too singular.
        """
        if center is None:
            center = [0.0] * grid.ndim
        r2 = sum((x - x0) ** 2 for x, x0 in zip(grid.mesh, center))
        if softening == 0 and r2.min() < 1e-12 * max(grid.spacings) ** 2:
            raise ValueError(
                "A grid point is on the nucleus, where -1/r is infinite: use an "
                "even number of points per axis, or a softening"
            )
        v = -charge * Kcoulomb / np.sqrt(r2 + softening**2)
        label = "Coulomb potential" if softening == 0 else "Soft Coulomb potential"
        return cls(grid, v, label=label)

class HamiltonianND:
    def __init__(self, potential):
        self.potential = potential
        self.grid = potential.grid

        # C order flattening: the last axis varies fastest, which is the order
        # that kron(Dx, Iy) assumes.
        v = self.potential.values.ravel()
        self.matrix = (-Ksch * laplacian(self.grid) + diags(v)).tocsr()

        # A hard wall means psi = 0 there. Rather than leaving a huge number on
        # the diagonal, as in 1D, we remove these points from the problem: the
        # matrix is smaller and much better conditioned, which matters in 3D.
        self.inside = np.flatnonzero(v < INFINITY)

        self._energies = None
        self._states = None

    def eigenstates(self, k=6, method=None, device="cpu"):
        """The k lowest energies [eV] and their states, each of shape grid.shape.

        The states are real and normalized: sum(|psi|^2) * grid.dv = 1.
        Within a degenerate level, any combination of the states is as good as
        another, and the solver returns an arbitrary one.

        method is "shift-invert" (factorizes the matrix: the best choice in 1D
        and 2D, but the factorization is slow and memory hungry in 3D, 40 s for
        61^3 points) or "lanczos" (only multiplies by the matrix: 1 s for the
        same 61^3 points). By default: shift-invert up to 2D, lanczos in 3D.

        method="torch" is a third one, which needs PyTorch: much faster on
        large 3D grids (128^3 points in 14 s) and the only practical one for
        the Coulomb potential in 3D. device is then "cpu", or "mps" or "cuda"
        for the GPU (see schrodinger_gpu.py).
        """
        if self._energies is not None and len(self._energies) >= k:
            return self._energies[:k], self._states[:k]

        h = self.matrix[self.inside][:, self.inside].tocsc()

        if method is None:
            method = "lanczos" if self.grid.ndim >= 3 else "shift-invert"

        if method == "torch":
            # A block method with a preconditioner, written with PyTorch in
            # schrodinger_gpu.py. It is the one to use for large 3D grids and
            # for the Coulomb potential, where Lanczos takes minutes.
            from schrodinger_gpu import GPUSolver

            solver = GPUSolver(self, device=device)
            self._energies, self._states = solver.eigenstates(k=k)
            return self._energies, self._states

        if method == "shift-invert":
            # Nothing is below the bottom of the potential: we look for the
            # eigenvalues nearest to a point just under it.
            bottom = self.potential.values.ravel()[self.inside].min()
            energies, vectors = eigsh(h, k=k, sigma=bottom - 1e-3, which="LM")
        else:
            # Lanczos can stop before it has found every copy of a degenerate
            # level, and 3D wells are full of them: asked for 4 states of the
            # harmonic well, it sometimes returns 1.5, 2.5, 2.5, 3.5 and skips
            # the third 2.5. The copy it misses is among the last ones, so we
            # ask for more states than needed and keep the lowest k.
            extra = min(k + max(k, 6), h.shape[0] - 1)
            energies, vectors = eigsh(h, k=extra, which="SA")

        order = np.argsort(energies)[:k]
        self._energies = energies[order]
        self._states = []
        for i in order:
            psi = np.zeros(self.matrix.shape[0])
            psi[self.inside] = vectors[:, i]
            psi /= np.sqrt(np.sum(psi**2) * self.grid.dv)
            if psi[np.argmax(np.abs(psi))] < 0:  # the sign is arbitrary
                psi = -psi
            self._states.append(psi.reshape(self.grid.shape))

        return self._energies, self._states

    def show_eigenstates(self, k=6, probability=False):
        """The first k states, as images.

        In 2D this is psi(x, y) itself. In 3D an image needs one axis less:
        we show the probability integrated along z, which is never empty,
        while a cut at z = 0 would be for any state that is odd in z.
        """
        if self.grid.ndim not in (2, 3):
            raise ValueError("Only 2D and 3D states can be shown as images")

        energies, states = self.eigenstates(k=k)
        x, y = self.grid.axes[0], self.grid.axes[1]

        columns = min(k, 3)
        rows = int(np.ceil(k / columns))
        fig, axes = plt.subplots(
            rows, columns, figsize=(3.4 * columns, 3.5 * rows), squeeze=False
        )

        # The outline of the well, from the potential in the z = 0 plane
        v = self.potential.values
        if self.grid.ndim == 3:
            v = v[:, :, self.grid.shape[2] // 2]
        v = np.minimum(v, np.min(v) + 1e3)

        for i, axis in enumerate(axes.flat):
            if i >= k:
                axis.axis("off")
                continue

            if self.grid.ndim == 3:
                image = np.sum(states[i] ** 2, axis=2) * self.grid.spacings[2]
                cmap, limit = "magma", (0, image.max())
            elif probability:
                image = states[i] ** 2
                cmap, limit = "magma", (0, image.max())
            else:
                image = states[i]
                peak = np.abs(image).max()
                cmap, limit = "RdBu_r", (-peak, peak)

            axis.pcolormesh(
                x, y, image.T, cmap=cmap, vmin=limit[0], vmax=limit[1], shading="auto"
            )
            if np.ptp(v) > 0:
                axis.contour(x, y, v.T, levels=3, colors="k", linewidths=0.5)
            axis.set_aspect("equal")
            axis.set_title(r"$\psi_{{{0}}}$  $E = {1:.4f}$ eV".format(i, energies[i]))
            axis.set_xlabel(r"x [$\AA$]")
            axis.set_ylabel(r"y [$\AA$]")

        label = self.potential.label or "Potential"
        if self.grid.ndim == 3:
            fig.suptitle(label + r" in 3D: $\int |\psi|^2 dz$")
        else:
            fig.suptitle(label + " in 2D")
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    a = 10.0

    print("2D infinite square well, a = {0} A".format(a))
    grid = Grid.cube(half_width=6, n=241, ndim=2)
    h = HamiltonianND(PotentialND.infinite_well(grid, widths=a))
    energies, states = h.eigenstates(k=6)
    unit = np.pi**2 * Ksch / a**2
    print("  E / (pi^2 hbar^2 / 2 m a^2) =", np.round(energies / unit, 3))
    print("  expected nx^2 + ny^2        = [2, 5, 5, 8, 10, 10]")
    h.show_eigenstates(k=6)

    print("3D infinite cubic well, a = {0} A".format(a))
    grid = Grid.cube(half_width=6, n=61, ndim=3)
    h = HamiltonianND(PotentialND.infinite_well(grid, widths=a))
    energies, states = h.eigenstates(k=7)
    print("  E / (pi^2 hbar^2 / 2 m a^2) =", np.round(energies / unit, 3))
    print("  expected nx^2 + ny^2 + nz^2 = [3, 6, 6, 6, 9, 9, 9]")
    h.show_eigenstates(k=6)
