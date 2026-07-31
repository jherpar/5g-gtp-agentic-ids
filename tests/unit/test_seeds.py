from __future__ import annotations

import random

import numpy as np

from agente_5g.utils.seeds import set_all_seeds


def test_set_all_seeds_makes_random_and_numpy_reproducible():
    set_all_seeds(42)
    a = [random.random() for _ in range(5)]
    b = np.random.rand(5).tolist()

    set_all_seeds(42)
    a2 = [random.random() for _ in range(5)]
    b2 = np.random.rand(5).tolist()

    assert a == a2
    assert b == b2


def test_different_seeds_produce_different_sequences():
    set_all_seeds(1)
    a = [random.random() for _ in range(5)]

    set_all_seeds(2)
    b = [random.random() for _ in range(5)]

    assert a != b
