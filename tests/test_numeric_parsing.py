import math
import os
import sys
import unittest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import koboldcpp


class NumericParsingTests(unittest.TestCase):
    def test_valid_numeric_values_keep_existing_behavior(self):
        self.assertEqual(koboldcpp.tryparseint("12", 17), 12)
        self.assertEqual(koboldcpp.tryparseint("true", 17), 1)
        self.assertEqual(koboldcpp.tryparseint("false", 17), 0)
        self.assertEqual(koboldcpp.tryparsefloat("0.75", 0.25), 0.75)

    def test_tryparseint_returns_fallback_for_incompatible_json_types(self):
        for value in ([], {}, [1], {"value": 1}):
            with self.subTest(value=value):
                self.assertEqual(koboldcpp.tryparseint(value, 17), 17)

    def test_tryparseint_returns_fallback_for_overflow(self):
        self.assertEqual(koboldcpp.tryparseint(math.inf, 17), 17)

    def test_tryparsefloat_returns_fallback_for_incompatible_json_types(self):
        for value in ([], {}, [1], {"value": 1}):
            with self.subTest(value=value):
                self.assertEqual(koboldcpp.tryparsefloat(value, 0.25), 0.25)

    def test_tryparsefloat_returns_fallback_for_non_finite_values(self):
        for value in (math.inf, -math.inf, math.nan, "inf", "nan", 10**10000):
            with self.subTest(value=value):
                self.assertEqual(koboldcpp.tryparsefloat(value, 0.25), 0.25)


if __name__ == "__main__":
    unittest.main()
