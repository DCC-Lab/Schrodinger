# L'oscillateur harmonique, résolu numériquement de la façon la plus simple
# possible.
#
# C'est exactement le même script que infinite_well_simple.py : la seule
# différence est qu'on ajoute un potentiel V(x) à l'hamiltonien.
#
# Unités : énergies en eV, distances en Angström.

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import hbar, m_e, elementary_charge

# hbar^2 / 2m pour un électron, converti en eV * Angström^2 (environ 3.81)
K = hbar**2 / (2 * m_e) / elementary_charge / 1e-20

# Énergie hbar*omega de l'oscillateur, en eV
hbar_omega = 1.0

# Le potentiel harmonique est V(x) = 1/2 m omega^2 x^2. Avec K = hbar^2 / 2m,
# on a m = hbar^2 / 2K, donc 1/2 m omega^2 = (hbar omega)^2 / 4K :
# on peut écrire V directement en eV avec x en Angström.
k_ressort = hbar_omega**2 / (4 * K)  # en eV / Angström^2

# Contrairement au puits infini, il n'y a pas de murs : la fonction d'onde
# s'étend en principe jusqu'à l'infini. Mais elle décroît comme une gaussienne,
# de largeur caractéristique sqrt(hbar / m omega) (environ 2.8 Angström ici).
# On prend donc une boîte beaucoup plus large que ça et on impose psi = 0 à ses
# bords : la fonction d'onde y est déjà nulle, donc ces murs ne changent rien.
L = 15.0
N = 1000
x = np.linspace(-L, L, N + 2)[1:-1]  # on garde seulement les points intérieurs
dx = x[1] - x[0]

# La dérivée seconde sous forme de matrice : -2 sur la diagonale, 1 juste
# au-dessus et juste en dessous, le tout divisé par dx^2 (comme pour le puits
# infini).
D2 = (
    np.diag(-2 * np.ones(N))
    + np.diag(np.ones(N - 1), k=1)
    + np.diag(np.ones(N - 1), k=-1)
) / dx**2

# Le potentiel est une matrice diagonale : multiplier psi par V(x), c'est
# multiplier chaque psi[i] par V(x_i).
V = k_ressort * x**2

# L'hamiltonien : énergie cinétique + énergie potentielle
#     H = -hbar^2 / 2m  d^2/dx^2  +  V(x)
H = -K * D2 + np.diag(V)

# Valeurs propres (énergies) et vecteurs propres (fonctions d'onde) de H.
# H est réelle et symétrique, donc eigh() donne des énergies réelles et triées.
energies, states = np.linalg.eigh(H)

# Ce que donne eigh : chaque vecteur propre v est de norme 1 au sens des vecteurs,
#
# Σᵢ |vᵢ|² = 1
#
# Ce qu'on veut : une fonction d'onde normalisée au sens des fonctions,
#
# ∫ |ψ(x)|² dx = 1
#
# Sur la grille, l'intégrale devient une somme de Riemann :
#
# ∫ |ψ(x)|² dx ≈ Σᵢ |ψᵢ|² Δx
#
# Le facteur : un vecteur propre multiplié par une constante reste un vecteur propre, donc on pose ψ = c v et on cherche
# c :
#
# Σᵢ |c vᵢ|² Δx = c² Δx Σᵢ |vᵢ|² = c² Δx = 1  ⇒  c = 1/√Δx
#
# D'où states = states / np.sqrt(dx).
#
states = states / np.sqrt(dx)

# Comparaison avec la solution analytique E_n = hbar omega (n + 1/2),
# avec n = 0, 1, 2, ... (ici n commence à 0, pas à 1 comme le puits infini)
print(" n   E numérique [eV]   E analytique [eV]")
for n in range(0, 5):
    exact = hbar_omega * (n + 0.5)
    print("{0:2d}   {1:16.5f}   {2:17.5f}".format(n, energies[n], exact))

# Le potentiel, pour voir où vivent les états
plt.plot(x, V, "k--", label="V(x)")

# Graphique des premiers états, chacun tracé à la hauteur de sa propre énergie
for n in range(0, 4):
    psi = states[:, n]
    if psi[np.argmax(np.abs(psi))] < 0:  # le signe d'un vecteur propre est arbitraire
        psi = -psi
    plt.plot(x, energies[n] + psi, label=f"n = {n}, E = {energies[n]:.3f} eV")
    plt.axhline(energies[n], color="gray", linestyle=":", linewidth=0.8)

plt.xlim(-10, 10)
plt.ylim(0, 5)
plt.xlabel(r"Position x [$\AA$]")
plt.ylabel("Énergie [eV]  (fonction d'onde ajoutée par-dessus)")
plt.title(rf"Oscillateur harmonique, $\hbar\omega$ = {hbar_omega} eV")
plt.legend()
plt.grid()
plt.show()
