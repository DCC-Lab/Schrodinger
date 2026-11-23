# Le puits infini, résolu numériquement de la façon la plus simple possible.
#
# C'est la même stratégie que schrodinger.py (l'hamiltonien est une matrice,
# ses vecteurs propres sont les états stationnaires), mais écrite comme un
# script linéaire, sans classes ni fonctions, pour qu'on puisse le lire de
# haut en bas.
#
# Unités : énergies en eV, distances en Angström.

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import hbar, m_e, elementary_charge

# hbar^2 / 2m pour un électron, converti en eV * Angström^2 (environ 3.81)
K = hbar**2 / (2 * m_e) / elementary_charge / 1e-20

# Largeur du puits, en Angström
a = 10.0

# La fonction d'onde est nulle hors du puits et sur les murs : on ne garde donc
# que les points strictement à l'intérieur du puits, x = dx, 2 dx, ..., a - dx.
# Les murs (x = 0 et x = a) ne sont pas dans le vecteur, ils valent zéro
# implicitement. C'est ainsi que le potentiel « infini » entre dans le
# problème : pas besoin d'un très grand nombre dans le potentiel, la condition
# aux limites fait le travail.
N = 500
x = np.linspace(0, a, N + 2)[1:-1]
dx = x[1] - x[0]

# La dérivée seconde sous forme de matrice, avec la différence finie
#     psi''(x_i) ~ (psi[i-1] - 2 psi[i] + psi[i+1]) / dx^2
# Cela donne -2 sur la diagonale et 1 juste au-dessus et juste en dessous.
# Sur la première et la dernière ligne, le voisin manquant est le mur, où
# psi = 0 : simplement l'omettre impose exactement la bonne condition aux
# limites.
D2 = (
    np.diag(-2 * np.ones(N))
    + np.diag(np.ones(N - 1), k=1)
    + np.diag(np.ones(N - 1), k=-1)
) / dx**2

# Dans le puits V(x) = 0 : l'hamiltonien n'est que l'énergie cinétique,
#     H = -hbar^2 / 2m  d^2/dx^2
H = -K * D2

# Résoudre l'équation de Schrödinger H psi = E psi, c'est trouver les valeurs
# propres (énergies) et les vecteurs propres (fonctions d'onde) de la matrice H.
# H est réelle et symétrique, donc eigh() est le bon outil : les énergies sont
# réelles et triées de la plus basse à la plus haute.
energies, states = np.linalg.eigh(H)

# Chaque colonne states[:, n] est une fonction d'onde. eigh() normalise le
# vecteur pour que somme |psi|^2 = 1, mais on veut que l'intégrale de
# |psi|^2 dx = 1.
states = states / np.sqrt(dx)

# Comparaison avec la solution analytique E_n = n^2 pi^2 hbar^2 / (2 m a^2)
print(" n   E numérique [eV]   E analytique [eV]")
for n in range(1, 6):
    exact = n**2 * np.pi**2 * K / a**2
    print("{0:2d}   {1:16.5f}   {2:17.5f}".format(n, energies[n - 1], exact))

# Graphique des premiers états, chacun tracé à la hauteur de sa propre énergie
for n in range(1, 4):
    psi = states[:, n - 1]
    if psi[np.argmax(np.abs(psi))] < 0:  # le signe d'un vecteur propre est arbitraire
        psi = -psi
    plt.plot(x, energies[n - 1] + psi, label=f"n = {n}, E = {energies[n - 1]:.3f} eV")
    plt.axhline(energies[n - 1], color="gray", linestyle=":", linewidth=0.8)

# Les murs du puits
plt.axvline(0, color="k", linewidth=2)
plt.axvline(a, color="k", linewidth=2)

plt.xlabel(r"Position x [$\AA$]")
plt.ylabel("Énergie [eV]  (fonction d'onde ajoutée par-dessus)")
plt.title(f"Puits infini de largeur a = {a} $\\AA$")
plt.legend()
plt.grid()
plt.show()
