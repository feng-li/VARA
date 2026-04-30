import unittest

from vara.params import make_block_order


class ParamsTests(unittest.TestCase):
    def test_make_block_order_uses_expected_contiguous_layout(self):
        self.assertEqual(
            make_block_order(3),
            {
                "eta_a": [0],
                "gamma_a": [1, 2, 3],
                "beta": [4, 5, 6],
                "sigma2": [7, 8, 9],
            },
        )


if __name__ == "__main__":
    unittest.main()
