import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from skimage.measure import marching_cubes

from schrodinger_nd import *

"""
Interactive figures for the states of schrodinger_nd.py, written as a web page
that can be rotated and zoomed in a browser.

    2D: psi(x, y) as a surface, the height and the colour are the amplitude.
    3D: psi(x, y, z) as two isosurfaces, the surface where psi is +level of its
        maximum in red, and -level in blue. These are the "orbitals" of
        chemistry textbooks.

Needs plotly and scikit-image, which the rest of the code does not:
    pip install plotly scikit-image
"""

RED, BLUE = "#c0392b", "#2c6fbb"


def title_of(i, energy):
    return "ψ<sub>{0}</sub>   E = {1:.4f} eV".format(i, energy)


def surfaces_2d(hamiltonian, k=6, columns=3):
    """The first k states of a 2D problem, as surfaces."""
    energies, states = hamiltonian.eigenstates(k=k)
    x, y = hamiltonian.grid.axes
    rows = int(np.ceil(k / columns))

    fig = make_subplots(
        rows=rows,
        cols=columns,
        specs=[[{"type": "surface"}] * columns] * rows,
        subplot_titles=[title_of(i, energies[i]) for i in range(k)],
        horizontal_spacing=0.01,
        vertical_spacing=0.05,
    )

    peak = max(np.abs(psi).max() for psi in states)
    for i, psi in enumerate(states):
        fig.add_trace(
            go.Surface(
                x=x,
                y=y,
                z=psi.T,
                colorscale="RdBu_r",
                cmin=-peak,
                cmax=peak,
                showscale=False,
            ),
            row=i // columns + 1,
            col=i % columns + 1,
        )

    fig.update_scenes(
        xaxis_title="x [Å]",
        yaxis_title="y [Å]",
        zaxis_title="ψ",
        zaxis_range=[-peak, peak],
        aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=0.6),
    )
    fig.update_layout(
        title=(hamiltonian.potential.label or "Potential") + " in 2D: ψ(x, y)",
        height=430 * rows,
        margin=dict(l=0, r=0, t=70, b=0),
    )
    return fig


def isosurface(psi, level, grid):
    """The triangles of the surface psi = level, in Angstrom, or None if empty."""
    if not (psi.min() < level < psi.max()):
        return None
    # On a fine grid, one point out of two or three is plenty for a drawing
    step = max(1, max(grid.shape) // 90)
    vertices, faces, _, _ = marching_cubes(
        psi, level=level, spacing=grid.spacings, step_size=step
    )
    origin = [axis[0] for axis in grid.axes]
    return vertices + origin, faces


def level_enclosing(psi, grid, probability):
    """The value of |psi| whose isosurface contains the given probability.

    A fixed fraction of the maximum is a poor choice for an atom: a 3s state
    is so peaked on the nucleus that its outer shells, where most of the
    probability is, fall below any reasonable fraction. Chemists draw the
    surface that encloses, say, 75% of the probability instead.
    """
    density = np.sort((psi**2).ravel())[::-1] * grid.dv
    index = np.searchsorted(np.cumsum(density), probability)
    return np.sqrt(density[min(index, len(density) - 1)] / grid.dv)


def isosurfaces_3d(hamiltonian, k=6, columns=3, level=0.3, enclosed=None, view=None):
    """The first k states of a 3D problem, as isosurfaces at +/- level * max.

    With enclosed (0.75 for instance), the surfaces are instead the ones that
    contain that probability, and level is ignored. view is the half width
    [Angstrom] of what is drawn, when the states are much smaller than the grid.
    """
    energies, states = hamiltonian.eigenstates(k=k)
    grid = hamiltonian.grid
    rows = int(np.ceil(k / columns))

    fig = make_subplots(
        rows=rows,
        cols=columns,
        specs=[[{"type": "mesh3d"}] * columns] * rows,
        subplot_titles=[title_of(i, energies[i]) for i in range(k)],
        horizontal_spacing=0.01,
        vertical_spacing=0.05,
    )

    for i, psi in enumerate(states):
        if enclosed is None:
            threshold = level * np.abs(psi).max()
        else:
            threshold = level_enclosing(psi, grid, enclosed)
        for sign, color in ((1, RED), (-1, BLUE)):
            surface = isosurface(psi, sign * threshold, grid)
            if surface is None:
                continue
            vertices, faces = surface
            fig.add_trace(
                go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    color=color,
                    opacity=0.75,
                    flatshading=False,
                    hoverinfo="skip",
                ),
                row=i // columns + 1,
                col=i % columns + 1,
            )

    limits = [[axis[0], axis[-1]] for axis in grid.axes]
    if view is not None:
        limits = [[-view, view]] * 3
    fig.update_scenes(
        xaxis=dict(title="x [Å]", range=limits[0]),
        yaxis=dict(title="y [Å]", range=limits[1]),
        zaxis=dict(title="z [Å]", range=limits[2]),
        aspectmode="cube",
    )
    fig.update_layout(
        title=(
            "{0} in 3D: ψ = +{1:.0%} of its maximum in red, −{1:.0%} in blue".format(
                hamiltonian.potential.label or "Potential", level
            )
            if enclosed is None
            else "{0} in 3D: surfaces containing {1:.0%} of the probability, "
            "ψ > 0 in red, ψ < 0 in blue".format(
                hamiltonian.potential.label or "Potential", enclosed
            )
        ),
        height=430 * rows,
        margin=dict(l=0, r=0, t=70, b=0),
    )
    return fig


def orbitals_3d(orbitals, grid, title, columns=3, enclosed=0.75, view=None):
    """Named orbitals (see hydrogen.py), as the surfaces containing the given
    probability. view is the half width [Angstrom] of what is drawn: the
    states are cut to that region first, which keeps the page small.
    """
    rows = int(np.ceil(len(orbitals) / columns))
    fig = make_subplots(
        rows=rows,
        cols=columns,
        specs=[[{"type": "mesh3d"}] * columns] * rows,
        subplot_titles=[
            "{0}   E = {1:.3f} eV".format(orbital.name, orbital.energy)
            for orbital in orbitals
        ],
        horizontal_spacing=0.01,
        vertical_spacing=0.05,
    )

    if view is None:
        view = grid.axes[0][-1]
    keep = [np.flatnonzero(np.abs(axis) <= view) for axis in grid.axes]
    # About 64 points per axis is plenty for a drawing of a smooth orbital
    keep = [k[:: max(1, len(k) // 64)] for k in keep]
    cut = Grid(*[axis[k] for axis, k in zip(grid.axes, keep)])

    for i, orbital in enumerate(orbitals):
        threshold = level_enclosing(orbital.psi, grid, enclosed)
        psi = orbital.psi[np.ix_(*keep)]
        for sign, color in ((1, RED), (-1, BLUE)):
            surface = isosurface(psi, sign * threshold, cut)
            if surface is None:
                continue
            vertices, faces = surface
            fig.add_trace(
                go.Mesh3d(
                    x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                    color=color, opacity=0.75, flatshading=False, hoverinfo="skip",
                ),
                row=i // columns + 1,
                col=i % columns + 1,
            )

    fig.update_scenes(
        xaxis=dict(title="x [Å]", range=[-view, view]),
        yaxis=dict(title="y [Å]", range=[-view, view]),
        zaxis=dict(title="z [Å]", range=[-view, view]),
        aspectmode="cube",
    )
    fig.update_layout(
        title="{0}: surfaces containing {1:.0%} of the probability, "
        "ψ > 0 in red, ψ < 0 in blue".format(title, enclosed),
        height=430 * rows,
        margin=dict(l=0, r=0, t=70, b=0),
    )
    return fig


def hydrogen_page(path, n_max=4, n=160):
    """The orbitals of hydrogen up to n_max, one button per level.

    A page can only show about 16 drawings at once, and n = 4 has 16 states:
    its f orbitals get their own button.
    """
    from hydrogen import solve

    orbitals, grid = solve(n_max=n_max, n=n)
    figures = []
    for level in range(1, n_max + 1):
        chosen = [orbital for orbital in orbitals if orbital.n == level]
        view = 1.5 * level**2 + 2
        parts = [("n = {0}".format(level), chosen)]
        if level >= 4:
            parts = [
                ("n = {0} (s, p, d)".format(level), [o for o in chosen if o.l < 3]),
                ("n = {0} (f...)".format(level), [o for o in chosen if o.l >= 3]),
            ]
        for name, part in parts:
            figures.append(orbitals_3d(part, grid, "Hydrogen " + name, view=view))
    write_page(figures, path)
    return orbitals, grid


def write_page(figures, path):
    """Every figure in a single self-contained web page, shown one at a time.

    A browser only allows about 16 WebGL drawings on a page, and each state is
    one of them: with every figure on the page at once, the first ones (the 2D
    ones) are silently dropped and stay blank. The page therefore has one
    button per figure and only draws the one that is selected.
    """
    import json
    from plotly.offline import get_plotlyjs

    names = [fig.layout.title.text.split(":")[0] for fig in figures]
    buttons = "".join(
        "<button onclick='show({0})'>{1}</button> ".format(n, name)
        for n, name in enumerate(names)
    )
    with open(path, "w") as page:
        page.write("<html><head><meta charset='utf-8'>\n")
        page.write("<style>body{font-family:sans-serif;margin:12px}")
        page.write("button{font-size:15px;padding:6px 12px;margin:2px}")
        page.write("button.on{background:#333;color:#fff}</style>\n")
        page.write("<script>" + get_plotlyjs() + "</script></head><body>\n")
        page.write("<div id='bar'>" + buttons + "</div><div id='plot'></div>\n")
        page.write("<script>\nconst figures = [")
        page.write(",\n".join(fig.to_json() for fig in figures))
        page.write("];\n")
        page.write(
            "function show(n) {\n"
            "  Plotly.purge('plot');\n"
            "  Plotly.newPlot('plot', figures[n].data, figures[n].layout);\n"
            "  document.querySelectorAll('#bar button').forEach(\n"
            "    (b, i) => b.classList.toggle('on', i == n));\n"
            "}\nshow(0);\n</script></body></html>\n"
        )


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "states.html"
    if "hydrogen" in sys.argv[2:]:  # python visualize_nd.py page.html hydrogen
        hydrogen_page(path)
        print("Written:", path)
        sys.exit()
    a, radius = 10.0, 5.0

    grid2 = Grid.cube(half_width=6, n=121, ndim=2)
    grid3 = Grid.cube(half_width=6, n=61, ndim=3)

    # Hydrogen: lanczos would take minutes, the block method of schrodinger_gpu.py
    # takes seconds. The states are kept, so the figure does not compute them again.
    hydrogen = HamiltonianND(PotentialND.coulomb(Grid.cube(10, 100, 3)))
    hydrogen.eigenstates(k=5, method="torch")

    figures = [
        surfaces_2d(HamiltonianND(PotentialND.infinite_well(grid2, widths=a))),
        surfaces_2d(HamiltonianND(PotentialND.round_well(grid2, radius))),
        isosurfaces_3d(HamiltonianND(PotentialND.infinite_well(grid3, widths=a))),
        isosurfaces_3d(HamiltonianND(PotentialND.round_well(grid3, radius))),
        isosurfaces_3d(
            HamiltonianND(PotentialND.harmonic_well(Grid.cube(10, 61, 3), omega=0.5))
        ),
        isosurfaces_3d(hydrogen, k=5, level=0.1),
    ]
    write_page(figures, path)
    print("Written:", path)
