import unittest
import numpy as np
import random
from enum import StrEnum
import math
from collections import Counter
import weakref


class State:
    def __init__(self, label, operator=None):
        if label is None:
            raise ValueError("The state label must be unique and not None")

        self.label = label

        self.operators = set()
        if operator is not None:
            self.operators.add(operator)

    def __str__(self):
        return f"| {self.label} 〉"

    def __eq__(self, rhs):
        return self.label == rhs.label and self.operators == rhs.operators

    def __hash__(self):
        return hash((self.label, tuple(self.operators)))

    def is_eigenstate_of(self, operator):
        return self in operator.eigenstates

    def has_eigenvalue_for(self, operator):
        if self.is_eigenstate_of(operator):
            return operator.eigens[self]

        return None


class Operator:
    def __init__(self, label, eigens=None):
        self.label = label

        self.eigens = {}
        if eigens is not None:
            self.eigens = {
                State(key, operator=self): value for key, value in eigens.items()
            }

    @property
    def eigenvalues(self):
        return list(self.eigens.values())

    @property
    def eigenstates(self):
        return list(self.eigens.keys())

    def adopt_eigen(self, state, eigenvalue):
        if state in self.eigens.keys():
            return

        self.eigens[state] = eigenvalue

    def __mul__(self, state):
        if not isinstance(state, State):
            raise ValueError("An operator can only operate on a ket")

        if state.is_eigenstate_of(self):
            return self.eigens[state], state

        # breakpoint()
        return (self, state)


class EigenState(State):
    def __init__(self, operators, eigenvalues, label=None):
        if label is None:
            label = str(eigenvalue)

        super().__init__(label=label)

        for op, v in zip(operators, eigenvalues):
            self.operators[op] = v


class Bra:
    def __init__(self, state):
        self.state = state

    def __str__(self):
        return f"⟨ {self.state.label} |"

    def __mul__(self, rhs):
        if not isinstance(rhs, Ket):
            raise ValueError("A bra can only be multiplied with a ket")

        if rhs.state == self.state:
            return 1

        raise ValueError(
            "The dot product of an arbitray bra with an arbitrary ket is unknown because we do not know if they are eigenstates of any operator"
        )


class Ket:
    def __init__(self, state):
        self.state = state

    def __str__(self):
        return f"| {self.state.label} 〉"


#     @property
#     def coeffs(self):
#         return self._coeffs

#     @property
#     def norm(self):
#         return sum(abs(self.coeffs) ** 2)

#     def normalize(self):
#         if self.norm != 0:
#             self._coeffs = self.coeffs / math.sqrt(self.norm)

#     @property
#     def probabilities(self):
#         return {str(e): float(abs(c)) ** 2 for c, e in zip(self.coeffs, self.bases)}

#     def measure(self, k=1, basis=None):
#         probs = self.probabilities
#         measured_state = random.choices(
#             list(probs.keys()), weights=list(probs.values()), k=k
#         )
#         return Counter(measured_state)


class BaseStateTestCase(unittest.TestCase):
    def test001_state_init(self):
        s = State("+")
        self.assertIsNotNone(s)

    def test002_bra_init(self):
        bra = Bra(State("+"))
        self.assertIsNotNone(bra)
        self.assertEqual(bra.state.label, "+")

    def test003_ket_init(self):
        ket = Ket(State("+"))
        self.assertIsNotNone(ket)
        self.assertEqual(ket.state.label, "+")

    def test004_bra_ket_print(self):
        bra = Bra(State("+"))
        self.assertEqual(str(bra), "⟨ + |")

        ket = Ket(State("+"))
        self.assertEqual(str(ket), "| + 〉")

    def test005_bra_ket_product(self):
        bra = Bra(State("+"))
        ket = Ket(State("+"))
        self.assertEqual(bra * ket, 1)

    def test005_bra_ket_product(self):
        bra = Bra(State("+"))
        ket = Ket(State("-"))
        with self.assertRaises(ValueError):
            bra * ket

    def test006_operator(self):
        op = Operator("Sz")
        self.assertIsNotNone(op)

    def test007_ket_eigenstate(self):
        op = Operator("Sz", eigens={"+z": 0.5, "-z": -0.5})
        (e1, e2) = op.eigenstates
        self.assertIsNotNone(e1)
        self.assertTrue(e1.is_eigenstate_of(op))

    def test008_ket_not_eigenstate(self):
        op = Operator("Sz", eigens={"+z": 0.5, "-z": -0.5})
        sx = State("+x")
        self.assertIsNotNone(sx)
        self.assertFalse(sx.is_eigenstate_of(op))
        self.assertEqual(op * sx, (op, sx))

    def test008_ket_is_eigenstate_when_explicitly_created_as_such(self):
        op = Operator("Sz", eigens={"+z": 0.5, "-z": -0.5})
        sz = State("+z")
        op.adopt_eigen(sz, 0.5)

        self.assertIsNotNone(sz)
        self.assertTrue(sz.is_eigenstate_of(op))
        self.assertEqual(op * sz, (0.5, sz))

    def test009_ket_is_eigenstate_when_explicitly_created_as_such_operator_none(self):
        op = Operator("Sz")
        sz = State("+z")
        op.adopt_eigen(sz, 0.5)

        self.assertIsNotNone(sz)
        self.assertTrue(sz.is_eigenstate_of(op))
        self.assertEqual(op * sz, (0.5, sz))


#     def test002_prints(self):
#         self.assertEqual(str(State()), "(0.0+0.0j)| +z 〉+ (0.0+0.0j)| -z 〉")

#     def test003_prints(self):
#         self.assertEqual(sum(State().probabilities.values()), 0)

#     def test004_measure(self):
#         s = State(coeffs=[1, 1])
#         print(s.measure(k=10000))

#     def test005_bracket(self):
#         self.assertEqual(Bra("+z") * Ket("+z"), 1)
#         with self.assertRaises(ValueError):
#             Bra("+z") * Ket("-z")

#     def test010_eigen(self):
#         ket = EigenKet("+z", +1 / 2, Sz)
#         self.assertTrue(isinstance(ket, Ket))
#         self.assertTrue(isinstance(ket, EigenKet))

#     def test006_orthogonality(self):
#         up, down = Sz_basis
#         self.assertEqual(up.bra * up, 1)
#         self.assertEqual(up.bra * down, 0)
#         self.assertEqual(down.bra * up, 0)

#     def test011_duals(self):
#         ket = Ket("+z")
#         bra_from_ket = Bra(label=ket.label)
#         bra_from_ket2 = Bra.of(ket)
#         bra_from_ket3 = Bra.dual_of(ket)

#         self.assertTrue(isinstance(bra_from_ket, Bra))
#         self.assertEqual(bra_from_ket.label, ket.label)

#         # self.assertTrue(isinstance(Ket(Bra("+z")), Ket)
#         # self.assertTrue(isinstance(EigenKet("+z", +1 / 2, Sz).bra, EigenBra))
#         # self.assertTrue(isinstance(EigenBra("+z", +1 / 2, Sz).ket, EigenKet))

#     def test012_operator_on_its_eigenkets(self):
#         up, down = Sz_basis
#         self.assertEqual(Sz * up, (+1 / 2, up))
#         self.assertEqual(Sz * down, (-1 / 2, down))
#         with self.assertRaises(ValueError):
#             Sz * Ket("+z")

#     def test020_complex_coefficients(self):
#         for coeffs in ([1, 1], [1, 1j], [1, 1 + 1j], [3, -2j]):
#             s = State(coeffs=coeffs)
#             self.assertAlmostEqual(s.norm, 1)
#             self.assertAlmostEqual(sum(s.probabilities.values()), 1)


if __name__ == "__main__":
    unittest.main()
