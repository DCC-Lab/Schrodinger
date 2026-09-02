import unittest
import numpy as np
import random
from enum import StrEnum
import math
from collections import Counter


# class Eigen(StrEnum):
#     ezp = "| +z 〉"
#     ezm = "| -z 〉"

#     eyp = "| +y 〉"
#     eym = "| -y 〉"

#     exp = "| +x 〉"
#     exm = "| -x 〉"


# class Basis:
#     def __init__(self, *args):
#         self.bases = args


# Sz_basis = (Eigen.ezp, Eigen.ezm)
# Sy_basis = (Eigen.eyp, Eigen.eym)
# Sx_basis = (Eigen.exp, Eigen.exm)


class Bra:
    def __init__(self, label):
        self.label = label

    def __str__(self):
        return f"⟨ {self.label} |"

    @property
    def ket(self):
        return Ket(self.label)

    def __mul__(self, rhs):
        if not isinstance(rhs, Ket):
            raise ValueError("A bra can only be multiplied with a ket")

        if rhs.label != self.label:
            raise ValueError("[FOR NOW] A bra can only be multiplied with its ket")

        return 1


class Ket:
    def __init__(self, label):
        self.label = label

    def __str__(self):
        return f"| {self.label} 〉"

    @property
    def bra(self):
        return Ket(self.label)


class Operator:
    def __init__(self, label, eigenvalues):
        self.label = label
        self.eigenvalues = eigenvalues

    def __mul__(self, rhs):
        if not isinstance(rhs, Ket):
            raise ValueError("An operator can only operate on a ket")

        if isinstance(rhs, EigenKet) and rhs.label == self.label:
            return (rhs.eigenvalue, rhs)


class EigenKet(Ket):
    def __init__(self, label, eigenvalue, operator):
        super().__init__(label)
        self.eigenvalue = eigenvalue
        self.operator = operator

    @property
    def bra(self):
        return Bra(self.label, np.conj(self.eigenvalue), self.operator)


class EigenBra(Bra):
    def __init__(self, label, eigenvalue, operator):
        super().__init__(label)
        self.eigenvalue = eigenvalue
        self.operator = operator

    @property
    def ket(self):
        return Ket(self.label, np.conj(self.eigenvalue), self.operator)


Sz = Operator("Sz", [1 / 2, -1 / 2])

Sz_basis = (EigenKet("+z", +1 / 2, Sz), EigenKet("-z", -1 / 2, Sz))


class State:
    def __init__(self, coeffs=None, bases=None):
        if bases is None:
            bases = Sz_basis
        self.bases = bases
        if coeffs is None:
            coeffs = np.array([0] * len(bases), dtype=complex)

        if len(coeffs) != len(self.bases):
            raise ValueError("Number of coefficients must match number of bases")

        self._coeffs = np.array(coeffs)

        if self.norm != 0:
            self.normalize()

    def __str__(self):
        expansion = [f"({c:.1f}){e}" for c, e in zip(self.coeffs, self.bases)]
        return "+ ".join(expansion)

    @property
    def coeffs(self):
        return self._coeffs

    @property
    def norm(self):
        return sum(self.coeffs * self.coeffs)

    def normalize(self):
        if self.norm != 0:
            self._coeffs = self.coeffs / math.sqrt(self.norm)

    @property
    def probabilities(self):
        return {str(e): float(abs(c)) ** 2 for c, e in zip(self.coeffs, self.bases)}

    def measure(self, k=1, basis=None):
        probs = self.probabilities
        measured_state = random.choices(
            list(probs.keys()), weights=list(probs.values()), k=k
        )
        return Counter(measured_state)


class BaseStateTestCase(unittest.TestCase):
    def test001_eigen(self):
        bra = EigenBra("+z", +1 / 2, Sz)
        ket = EigenKet("+z", +1 / 2, Sz)
        self.assertIsNotNone(ket)
        print(bra)
        print(ket)

    def test001_init(self):
        self.assertIsNotNone(State())

    def test002_prints(self):
        self.assertEqual(str(State()), "(0.0+0.0j)| +z 〉+ (0.0+0.0j)| -z 〉")

    def test003_prints(self):
        self.assertEqual(sum(State().probabilities.values()), 0)

    def test004_measure(self):
        s = State(coeffs=[1, 1])
        print(s.measure(k=10000))

    def test005_bracket(self):
        self.assertEqual(Bra("+z") * Ket("+z"), 1)
        with self.assertRaises(ValueError):
            Bra("+z") * Ket("-z")

    def test010_eigen(self):
        ket = EigenKet("+z", +1 / 2, Sz)
        self.assertTrue(isinstance(ket, Ket))
        self.assertTrue(isinstance(ket, EigenKet))


if __name__ == "__main__":
    unittest.main()
