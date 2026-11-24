# Schrodinger
This Python code is in support of the GPH-3004 Quantum Mechanics class for Engineers.

Daniel Côté dccote@cervo.ulaval.ca

## Getting started

 A set of classes to solve the Schrödinger equation for various potentials.

 Use `pip install -r requirements.txt` to install the required packages.



## What is this?

Four scripts:

1. `schrodinger.py`: classes to solve the schrodinger equation in various potentials. Several potentials are already programmed:

   1. Infinite well
   2. Finite well
   3. Harmonic potential
   4. Half-harmonic potential

2. `bells_inequalities.py` : Following Chapter 4 in McIntyre, an implementation of what the results would be in a Bell's experiment with entangled particles.  The code will produce the results as would be expected if Einstein-Podolsky-Rosen were correct and hidden variables exist, or if the current mathematical formalism with the Copenhagen interation is sufficient to explain the results.

   1. The code will produce two sets of correlations which are different : never better than 44.4% with EPR, and 50% with QM.  Only an experiment can decide.
   2. Experiments were performed and showed that QM is correct.

3. `deriveintegrate.py `: Differential operators can be represented by matrices.  In suggests that integral operators can alos be represented as the inverse of differential operators.  What may come as a surprise is that the definition of an operator does not set its representation unambiguously: we must also consider the domain of application, which includes the boundary conditions of the vectors.

   Simple cases are described to demonstrate this.

4. `testschrodinger.py` because we always test our code.

5. `schrodinger_nd.py`: the same matrix strategy in 2D and 3D. The wavefunction on a grid is flattened into one long vector, and the Laplacian is built from the 1D second derivative with Kronecker products: `D2x ⊗ Iy + Ix ⊗ D2y` in 2D, three such terms in 3D. Potentials: infinite and finite box, disk or sphere, harmonic, Coulomb and soft Coulomb (hydrogen atom). Tested in `testschrodinger_nd.py` against the analytical energies.

   ```python
   from schrodinger_nd import *

   grid = Grid.cube(half_width=6, n=241, ndim=2)   # ndim=3 for a cube: keep n near 61
   h = HamiltonianND(PotentialND.infinite_well(grid, widths=10))
   energies, states = h.eigenstates(k=6)
   h.show_eigenstates(k=6)
   ```

6. `schrodinger_gpu.py`: the same Hamiltonian without storing the matrix, with PyTorch, on the CPU or the GPU (`mps` on a Mac, `cuda`). It is the fast solver for large 3D grids and the Coulomb potential: the hydrogen atom on 128^3 points in 14 s on the CPU, 6 s on the GPU. Needs `pip install torch`.

   ```python
   grid = Grid.cube(half_width=10, n=128, ndim=3)   # even n: no grid point on the nucleus
   h = HamiltonianND(PotentialND.coulomb(grid))
   energies, states = h.eigenstates(k=5, method="torch")   # device="mps" for the GPU
   ```

7. `visualize_nd.py`: writes a web page with the 2D states as surfaces and the 3D states as isosurfaces, to rotate in a browser. Needs `pip install plotly scikit-image`.

8. `hydrogen.py`: the excited states of hydrogen (n = 2, 3, 4) sorted into the textbook orbitals. The solver returns arbitrary mixtures inside a degenerate level; diagonalizing L², then Lz², then cos(2mφ) inside each level gives back 3s, 3p z, 3d xy, 4f xyz... Tested in `testhydrogen.py`.

   ```python
   from hydrogen import *

   orbitals, grid = solve(n_max=3, n=128)    # 14 states in 15 s; n_max=4, n=160: 30 states in 1 min
   for orbital in orbitals:
       print(orbital.name, orbital.energy)
   ```

   `solve(..., method="matrix")` uses the strategy of the 1D code unchanged (the sparse matrix of `schrodinger_nd.py`, then `eigsh`) instead of the block solver. Same energies to 10 digits; for n ≤ 3 it is as fast (20 s on 96^3), for n ≤ 4 on 160^3 it takes 13 minutes against 1 minute.

   `python hydrogen.py 4 160` prints the table, and `python visualize_nd.py page.html hydrogen` writes the orbitals as a web page.





