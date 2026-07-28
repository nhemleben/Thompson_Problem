import math
import sys
from pathlib import Path
import unittest


sys.path.append(str(Path(__file__).resolve().parents[1]))

from energy import thompson_energy


class ThompsonEnergyTests(unittest.TestCase):

    def test_empty_configuration_has_zero_energy(self):
        self.assertEqual(thompson_energy([]), 0)

    def test_two_antipodal_points_have_energy_one_half(self):
        energy = thompson_energy([
            (0.0, 0.0),
            (0.0, math.pi),
        ])

        self.assertAlmostEqual(energy, 0.5)

    def test_duplicate_points_return_infinite_energy(self):
        energy = thompson_energy([
            (1.0, 1.0),
            (1.0, 1.0),
        ])

        self.assertTrue(math.isinf(energy))


if __name__ == "__main__":
    unittest.main()