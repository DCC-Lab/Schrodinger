import time

import numpy as np
import torch

from schrodinger import INFINITY, Ksch

"""
The lowest states of a HamiltonianND, computed on the GPU with PyTorch.

schrodinger_nd.py writes H as a sparse matrix and gives it to scipy. That is
the clearest way to see the strategy, but scipy runs on the CPU, and a sparse
matrix is a poor fit for a GPU. Here the matrix is never written down: all we
ever need is its action on a wavefunction, and on a grid that action is

    (H psi)[i, j, k] = -Ksch * (psi[i+1, j, k] - 2 psi[i, j, k] + psi[i-1, j, k]) / dx^2
                       + (same along y and z)
                       + V[i, j, k] * psi[i, j, k]

that is, shifted copies of the table psi added together. This is exactly the
same matrix as before (same eigenvalues), applied without being stored, and a
GPU does it on millions of points at once.

The eigenvalues come from LOBPCG, a method that improves a block of trial
states together, which makes it safe with degenerate levels. Its convergence
relies on a preconditioner: an approximate inverse of H. We use the inverse of
the kinetic energy alone, which is exact and fast because the sine functions
are the eigenvectors of the second derivative matrix (a sine transform, done
with an FFT).

On a Mac, the GPU is the "mps" device and only computes in single precision
(float32): energies are good to about 5 digits, far better than the error
of the grid itself. With device="cpu" the same code runs in double precision.

Needs PyTorch, which the rest of the code does not:  pip install torch
"""


def best_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sine_transform(x, dim):
    """The discrete sine transform (DST-I) along one dimension, with an FFT.

    Applied twice it gives back x, times 2 * (n + 1).
    """
    zero = torch.zeros_like(x.narrow(dim, 0, 1))
    odd = torch.cat([zero, x, zero, -x.flip(dim)], dim=dim)
    return -torch.fft.rfft(odd, dim=dim).imag.narrow(dim, 1, x.shape[dim])


class GPUSolver:
    def __init__(self, hamiltonian, device=None):
        """hamiltonian is a HamiltonianND: we only take its grid and potential."""
        self.grid = hamiltonian.grid
        self.device = device or best_device()
        self.dtype = torch.float32 if self.device == "mps" else torch.float64

        values = hamiltonian.potential.values
        inside = values < INFINITY
        self.inside = torch.tensor(inside, dtype=self.dtype, device=self.device)
        self.v = torch.tensor(
            np.where(inside, values, 0.0), dtype=self.dtype, device=self.device
        )
        self.k_dx2 = [Ksch / dx**2 for dx in self.grid.spacings]

        # The eigenvalues of -Ksch * laplacian, in the basis of the sines
        self.kinetic = torch.zeros(self.grid.shape, dtype=self.dtype, device=self.device)
        for axis, n in enumerate(self.grid.shape):
            modes = np.arange(1, n + 1)
            values = 4 * self.k_dx2[axis] * np.sin(np.pi * modes / (2 * (n + 1))) ** 2
            shape = [n if i == axis else 1 for i in range(self.grid.ndim)]
            self.kinetic = self.kinetic + torch.tensor(
                values.reshape(shape), dtype=self.dtype, device=self.device
            )

        self.iterations = None
        self.residuals = None

    def apply_h(self, psi):
        """H acting on a block of wavefunctions, of shape (how many, *grid.shape)"""
        out = (2 * sum(self.k_dx2) + self.v) * psi
        for axis, k_dx2 in enumerate(self.k_dx2):
            dim = axis + 1
            n = psi.shape[dim]
            out.narrow(dim, 1, n - 1).sub_(k_dx2 * psi.narrow(dim, 0, n - 1))
            out.narrow(dim, 0, n - 1).sub_(k_dx2 * psi.narrow(dim, 1, n - 1))
        return out * self.inside

    def precondition(self, residual, shift):
        """(kinetic energy + shift)^-1, an approximate inverse of H

        shift is one number, or one per state. The states are transformed one
        at a time: the sine transform needs four times the memory of what it
        transforms, too much for a block of 40 states on a large grid.
        """
        scale = 1.0
        for n in self.grid.shape:
            scale *= 2 * (n + 1)
        out = torch.empty_like(residual)
        for i in range(residual.shape[0]):
            one = residual[i : i + 1]
            for axis in range(self.grid.ndim):
                one = sine_transform(one, axis + 1)
            s = shift[i : i + 1] if torch.is_tensor(shift) else shift
            one = one / ((self.kinetic + s) * scale)
            for axis in range(self.grid.ndim):
                one = sine_transform(one, axis + 1)
            out[i] = one[0]
        return out * self.inside

    def eigenstates(self, k=6, tolerance=None, max_iterations=500, extra=None, seed=0):
        """The k lowest energies [eV] and states, as schrodinger_nd.py returns them.

        tolerance is on the residual |H psi - E psi| [eV] of normalized states.
        The iterations also stop when the residual no longer decreases.
        """
        if tolerance is None:
            tolerance = 2e-3 if self.dtype == torch.float32 else 1e-4
        if extra is None:
            extra = max(2, k // 2)  # a few more states: the last ones converge slowly
        m = k + extra
        flat = lambda block: block.reshape(block.shape[0], -1)

        def gram(a, b):
            # Every scalar product between two blocks. In single precision a sum
            # over millions of points loses digits, enough to make the
            # iterations diverge on large grids: the sum is done by pieces of
            # 65536 points on the GPU, and the pieces are added in double
            # precision on the CPU.
            total = 0
            for piece_a, piece_b in zip(
                flat(a).split(65536, dim=1), flat(b).split(65536, dim=1)
            ):
                total = total + (piece_a @ piece_b.T).cpu().double()
            return total.numpy()

        def combine(coefficients, blocks):
            c = torch.tensor(coefficients, dtype=self.dtype, device=self.device)
            stacked = torch.cat([flat(block) for block in blocks])
            return (c.T @ stacked).reshape(-1, *self.grid.shape)

        def rayleigh_ritz(blocks, h_blocks):
            """The best m states that can be made out of the given ones."""
            s = torch.cat(blocks)
            hs = torch.cat(h_blocks)
            a, b = gram(s, hs), gram(s, s)
            a, b = (a + a.T) / 2, (b + b.T) / 2
            # Directions that are nearly combinations of the others are dropped
            weights, vectors = np.linalg.eigh(b)
            keep = weights > weights.max() * (1e-5 if self.dtype == torch.float32 else 1e-11)
            basis = vectors[:, keep] / np.sqrt(weights[keep])
            energies, z = np.linalg.eigh(basis.T @ a @ basis)
            return energies[:m], basis @ z[:, :m]

        generator = torch.Generator().manual_seed(seed)
        x = torch.randn((m, *self.grid.shape), generator=generator, dtype=torch.float64)
        x = x.to(self.dtype).to(self.device) * self.inside
        x = self.precondition(x, shift=1.0)  # smooth: a much better first guess

        hx = self.apply_h(x)
        energies, c = rayleigh_ritz([x], [hx])
        x, hx = combine(c, [x]), combine(c, [hx])
        p = hp = None
        self.history = []

        for iteration in range(1, max_iterations + 1):
            if iteration % 20 == 0:  # the running copy of H x drifts with rounding
                hx = self.apply_h(x)

            e = torch.tensor(energies, dtype=self.dtype, device=self.device)
            r = hx - e.reshape(-1, *[1] * self.grid.ndim) * x
            residuals = torch.linalg.vector_norm(flat(r), dim=1).cpu().numpy()
            self.history.append(residuals[:k].max())
            if residuals[:k].max() < tolerance:
                break
            # In single precision the residual cannot go below the rounding of
            # H psi itself, about 1e-7 of the largest term: when it has stopped
            # decreasing, the states are as good as they will get.
            if iteration > 10 and min(self.history[-5:]) > 0.9 * min(self.history[:-5]):
                break

            # How stiff the preconditioner should be: the kinetic energy of the states
            t = (flat(x) * flat(hx - self.v * x)).sum(dim=1).clamp_min(0.1)
            w = self.precondition(r, shift=t.reshape(-1, *[1] * self.grid.ndim))
            # Same length as the states, or w is lost in their rounding
            w = w / torch.linalg.vector_norm(flat(w), dim=1).reshape(
                -1, *[1] * self.grid.ndim
            )
            hw = self.apply_h(w)

            blocks = [x, w] if p is None else [x, w, p]
            h_blocks = [hx, hw] if p is None else [hx, hw, hp]
            energies, c = rayleigh_ritz(blocks, h_blocks)

            # p is the part of the new x that is not the old x: the direction
            # in which it just moved, which is worth trying again.
            c_p = c.copy()
            c_p[:m] = 0
            p, hp = combine(c_p, blocks), combine(c_p, h_blocks)
            x, hx = combine(c, blocks), combine(c, h_blocks)

        self.iterations = iteration
        self.residuals = residuals[:k]

        states = []
        for i in range(k):
            psi = x[i].cpu().double().numpy()
            psi /= np.sqrt(np.sum(psi**2) * self.grid.dv)
            if psi[np.unravel_index(np.argmax(np.abs(psi)), psi.shape)] < 0:
                psi = -psi
            states.append(psi)

        return np.array(energies[:k]), states


if __name__ == "__main__":
    from schrodinger_nd import Grid, HamiltonianND, PotentialND

    print("Hydrogen atom: 1s at -13.606 eV, then 2s and 2p (x3) at -3.401 eV")
    for n in (64, 128):
        grid = Grid.cube(half_width=10, n=n, ndim=3)
        h = HamiltonianND.__new__(HamiltonianND)  # no sparse matrix needed here
        h.potential, h.grid = PotentialND.coulomb(grid), grid
        for device in ("cpu", best_device()):
            solver = GPUSolver(h, device=device)
            start = time.time()
            energies, states = solver.eigenstates(k=5)
            print(
                "  {0}^3 {1:>4}: {2:6.1f} s, {3:3d} iterations".format(
                    n, device, time.time() - start, solver.iterations
                ),
                np.round(energies, 3),
            )
