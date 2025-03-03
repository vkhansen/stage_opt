import unittest
import numpy as np
from src.optimization.physics import calculate_stage_ratios, calculate_mass_ratios, calculate_payload_fraction

class TestPhysics(unittest.TestCase):
    def test_calculates_correct_stage_and_mass_ratios(self):
        dv = np.array([3000, 4000])
        G0 = 9.81
        ISP = np.array([300, 350])
        EPSILON = np.array([0.1, 0.15])
        stage_ratios, mass_ratios = calculate_stage_ratios(dv, G0, ISP, EPSILON)
        np.testing.assert_almost_equal(stage_ratios, [0.36082291, 0.31192516], decimal=4)
        np.testing.assert_almost_equal(mass_ratios, [2.35437807, 2.40884691], decimal=4)

if __name__ == '__main__':
    unittest.main()
