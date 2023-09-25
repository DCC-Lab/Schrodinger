import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import numpy as np
from scipy.signal import hilbert, chirp
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import unittest 

# You could import materials from Raytracing:
# from raytracing.materials import *
# print(Material.all())

# Shortcuts
I = complex(0, 1)
π = np.pi
c = 3e8



class Wavepacket:
    def __init__(self):
        pass


class TestWavepacket(unittest.TestCase):
    def testInit(self):
        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()