import unittest
import numpy as np
import random
from enum import StrEnum
import math
from collections import Counter
import weakref
import numbers


class Ket:
    def __init__(self, label):
        if label is None:
            raise ValueError("The state label must be unique and not None")

        self.label = label

    def __str__(self):
        return f"| {self.label} 〉"

    def __eq__(self, rhs):
        return self.label == rhs.label

    def __hash__(self):
        return hash(self.label)

    def __rmul__(self, lhs):
        """
        We handle : 1. Bra * Ket
                    2. float * Ket
                    3. Operator * (eigen) Ket
                    4. Operator * (any other) Ket

        """
        if isinstance(lhs, Bra):
            if lhs.label == self.label:
                return 1  # always normalized
            else:
                if self.is_orthogonal_to(lhs):
                    return 0
                else:
                    return (lhs, self)
        elif isinstance(lhs, numbers.Number):
            return (lhs, self)
        elif isinstance(lhs, Operator):
            if lhs.has_eigenstate(self):
                return (lhs.eigenstates[self], self)
            else:
                return (lhs, self)

        return None

    def is_orthogonal_to(self, other_state):
        for op in Operator.all:
            if op.has_eigenstate(self) and op.has_eigenstate(other_state):
                return True

        return False


State = Ket


class Bra(State):
    def __init__(self, state_or_label):
        label = state_or_label
        if isinstance(state_or_label, Ket):
            label = state_or_label.label

        super().__init__(label=label)

    def __str__(self):
        return f"⟨ {self.label} |"

    def __mul__(self, rhs):
        if isinstance(rhs, Ket):
            if rhs.label == self.label:
                return 1  # always normalized
            else:
                if self.is_orthogonal_to(rhs):
                    return 0
                else:
                    return (self, rhs)
        elif isinstance(rhs, tuple):
            coeff_or_operator, ket = rhs
            if isinstance(coeff_or_operator, numbers.Number) and isinstance(ket, Ket):
                return coeff_or_operator * (self * ket)
            elif isinstance(coeff_or_operator, Operator) and isinstance(ket, Ket):
                return (self * coeff_or_operator) * ket

        return None


class Superposition(State):
    def __init__(self, label, states: dict):
        if label is None:
            raise ValueError("The state label must be unique and not None")

        self.label = label
        self.states = states

        self.normalize()

    def expand(self):
        return " + ".join(
            [f"{coeff:.3f}{state}" for state, coeff in self.states.items()]
        )

    @property
    def coeffs(self):
        return list(self.states.values())

    @property
    def norm(self):
        return sum(abs(np.array(self.coeffs)) ** 2)

    @property
    def is_normalized(self):
        return math.isclose(self.norm, 1, rel_tol=1e-6, abs_tol=0.0)

    def normalize(self):
        if self.norm != 0:
            amplitude = math.sqrt(self.norm)

            for k, v in self.states.items():
                self.states[k] = v / amplitude

        return self

    def measure(self, operator, k=1):
        probs = {}
        for state, coeff in self.states.items():
            if state not in operator.eigenstates:
                raise ValueError(
                    f"Impossible to apply measurement operator {operator} on superposition {self} if any basis state of the superposition is not an eigenstate"
                )
            else:
                probs[state] = abs(coeff) ** 2

        measured_state = random.choices(
            list(probs.keys()), weights=list(probs.values()), k=k
        )
        return Counter(measured_state)


class Operator:
    all = []

    def __init__(self, label, eigens=None):
        self.label = label
        self.eigens = {}

        if eigens is not None:
            for state_or_label, eigenvalue in eigens.items():
                self.add_eigenstate(state_or_label, eigenvalue)

        Operator.all.append(self)

    def add_eigenstate(self, state_or_label, eigenvalue):
        if isinstance(state_or_label, str):
            state = State(label=state_or_label)
        elif isinstance(state_or_label, State):
            state = state_or_label
        else:
            raise ValueError(
                f"An eigenstate is a State or its label, not {type(state_or_label).__name__}"
            )

        if state in self.eigens and self.eigens[state] != eigenvalue:
            raise ValueError(
                f"{state} is already an eigenstate of {self.label} with eigenvalue {self.eigens[state]}"
            )

        self.eigens[state] = eigenvalue

        return state

    @property
    def eigenvalues(self):
        return list(self.eigens.values())

    @property
    def eigenstates(self):
        return list(self.eigens.keys())

    def has_eigenstate(self, state):
        return state in self.eigens.keys()

    def __str__(self):
        return self.label

    def __mul__(self, state):
        if not isinstance(state, State):
            raise ValueError("An operator can only operate on a ket")

        if self.has_eigenstate(state):
            return self.eigens[state], state

        # If not an eigenstate, we cannot compute anything
        return (self, state)


class BaseStateTestCase(unittest.TestCase):
    def test001_state_init(self):
        s = State("+")
        self.assertIsNotNone(s)

    def test002_bra_init(self):
        bra = Bra("+")
        self.assertIsNotNone(bra)
        self.assertEqual(bra.label, "+")

    def test003_ket_init(self):
        ket = Ket("+")
        self.assertIsNotNone(ket)
        self.assertEqual(ket.label, "+")

    def test004_bra_ket_print(self):
        bra = Bra("+")
        self.assertEqual(str(bra), "⟨ + |")

        ket = Ket("+")
        self.assertEqual(str(ket), "| + 〉")

    def test005_bra_ket_product_same_state(self):
        bra = Bra("+")
        ket = Ket("+")
        self.assertEqual(bra * ket, 1)

    def test0051_bra_ket_product_orthogonal_states(self):
        op = Operator("Sz", eigens={"+z": 0.5, "-z": -0.5})
        (e1, e2) = op.eigenstates

        bra = Bra(e1)
        ket = e2

        self.assertTrue(isinstance(bra, State))
        self.assertTrue(isinstance(ket, State))

        self.assertEqual(bra * ket, 0)

    def test0051_bra_ket_product_any_states(self):
        bra = Bra("+")
        ket = Ket("-")
        self.assertEqual(bra * ket, (bra, ket))

    def test0052_bra_ket_product_coeff_with_state(self):
        bra = Bra("+")
        ket = Ket("-")
        self.assertEqual(bra * (1, ket), (bra, ket))

    def test006_operator(self):
        op = Operator("Sz")
        self.assertIsNotNone(op)

    def test007_ket_eigenstate(self):
        op = Operator("Sz", eigens={"+z": 0.5, "-z": -0.5})
        (e1, e2) = op.eigenstates
        self.assertIsNotNone(e1)
        self.assertTrue(op.has_eigenstate(e1))

    def test008_ket_not_eigenstate(self):
        op = Operator("Sz", eigens={"+z": 0.5, "-z": -0.5})
        sx = State("+x")
        self.assertIsNotNone(sx)
        self.assertFalse(op.has_eigenstate(sx))
        self.assertEqual(op * sx, (op, sx))

    def test008_ket_is_eigenstate_when_explicitly_created_as_such(self):
        op = Operator("Sz", eigens={"+z": 0.5, "-z": -0.5})
        sz = State("+z")
        op.add_eigenstate(sz, 0.5)

        self.assertIsNotNone(sz)
        self.assertTrue(op.has_eigenstate(sz))
        self.assertEqual(op * sz, (0.5, sz))

    def test009_ket_is_eigenstate_when_explicitly_created_as_such_operator_none(self):
        op = Operator("Sz")
        sz = op.add_eigenstate("+z", 0.5)

        self.assertIsNotNone(sz)
        self.assertTrue(op.has_eigenstate(sz))
        self.assertEqual(op * sz, (0.5, sz))

    def test010_add_eigenstate_refuses_what_is_not_a_state(self):
        op = Operator("Sz")
        with self.assertRaises(ValueError):
            op.add_eigenstate(42, 0.5)
        self.assertEqual(op.eigenstates, [])

    def test011_add_eigenstate_is_idempotent(self):
        op = Operator("Sz", eigens={"+z": 0.5})
        op.add_eigenstate("+z", 0.5)
        op.add_eigenstate(State("+z"), 0.5)
        self.assertEqual(op.eigenvalues, [0.5])

    def test012_add_eigenstate_refuses_to_contradict_itself(self):
        op = Operator("Sz", eigens={"+z": 0.5})
        with self.assertRaises(ValueError):
            op.add_eigenstate("+z", 99)
        self.assertEqual(op.eigens[State("+z")], 0.5)

    def test0052_scalar_on_ket(self):
        op = Operator("Sz", eigens={"+z": 0.5, "-z": -0.5})
        (e1, e2) = op.eigenstates
        self.assertEqual(0.25 * e1, (0.25, e1))

    def test0052_operator_on_ket(self):
        op = Operator("Sz", eigens={"+z": 0.5, "-z": -0.5})
        (e1, e2) = op.eigenstates
        self.assertEqual(op * e1, (0.5, e1))
        self.assertEqual(op * e2, (-0.5, e2))

    def test013_superposition(self):
        op = Operator("Sz", eigens={"+z": 0.5, "-z": -0.5})
        e1, e2 = op.eigenstates
        state = Superposition("ѱ", states={e1: 1, e2: 1})
        self.assertIsNotNone(state)
        self.assertTrue(state.is_normalized)

    def test014_superposition_measurement(self):
        op = Operator("Sz", eigens={"+z": 0.5, "-z": -0.5})
        e1, e2 = op.eigenstates
        state = Superposition("ѱ", states={e1: 1, e2: 1})

        k = 100
        counter = state.measure(op, k=k)
        print(f"\n{k} measurements of operator {op}")
        for s, c in counter.items():
            print(f"{s} : {c}")


if __name__ == "__main__":
    # unittest.main(defaultTest=["BaseStateTestCase.test005_bra_ket_product"])
    unittest.main()
