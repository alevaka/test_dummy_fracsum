import unittest
import numpy as np
from dummyfs import DummyRegressorFracSum
from sklearn.utils._testing import (
    assert_array_equal,
)


class TestDummyRegressorWithFracSum(unittest.TestCase):
    def test_fracsum_strategy_regressor(self):
        random_state = np.random.RandomState()

        X = [[0]] * 4  # ignored
        y = random_state.randn(4, 3)
        reg = DummyRegressorFracSum()
        reg.fit(X, y)
        assert_array_equal(reg.predict(X),
                           [np.sum(np.mod(y, 1), axis=0)] * len(X))


if __name__ == "__main__":
    unittest.main()
