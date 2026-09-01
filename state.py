import unittest
import numpy as np
import random
from enum import StrEnum
import math


class Base(StrEnum):
    ep = "| +z 〉"
    em = "| -z 〉"


class State:
    def __init__(self, coeffs=None, bases=None):
        if bases is None:
            bases = [Base.ep, Base.em]
        self.bases = bases
        if coeffs is None:
            coeffs = np.array([0, 0], dtype=complex)
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

    def measure(self, basis=None):
        probs = self.probabilities
        measured_state = random.choices(
            list(probs.keys()), weights=list(probs.values()), k=1
        )[0]
        return measured_state


class BaseStateTestCase(unittest.TestCase):
    def test001_init(self):
        self.assertIsNotNone(State())

    def test002_prints(self):
        self.assertEqual(str(State()), "(0.0+0.0j)| +z 〉+ (0.0+0.0j)| -z 〉")

    def test003_prints(self):
        print(State().probabilities)

    def test004_measure(self):
        s = State(coeffs=[1, 1])
        for _ in range(10):
            print(s.measure())


if __name__ == "__main__":
    unittest.main()
