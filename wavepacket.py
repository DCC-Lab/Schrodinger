import unittest 
import numpy as np

# Shortcuts
I = complex(0, 1)
π = np.pi
c = 3e8

class Wavepacket:
    def __init__(self, N=100):
        self._x = np.linspace(-100,100, N)
        self._psi = [0]*N

    def setToGaussianWavepacket(self, gaussianSpatialWidth=10, N=100):
        a = gaussianSpatialWidth
        self._psi = np.exp(-self._x*self._x/a/a)
        self.normalize()

    @property
    def x(self):
        return self._x
    
    @property
    def psi(self):
        return self._psi

    @property
    def psi2(self):
        return np.conj(self.psi) * self.psi

    @property
    def norm2(self):
        return np.sum(self.psi2)

    @property
    def norm(self):
        return np.sqrt(np.sum(self.psi2))

    def normalize(self):
        self._psi = self._psi/np.sqrt(self.norm2)

    def isNormalized(self):
        return abs(self.norm2-1.0) < 0.0001
    

class TestWavepacket(unittest.TestCase):
    def testInit(self):
        self.assertTrue(True)

    def testInitWavepacket(self):
        wp = Wavepacket()
        self.assertIsNotNone(wp)

    def testGetNullButNotEmptyPsi(self):
        wp = Wavepacket()
        self.assertIsNotNone(wp.psi)
        self.assertTrue(len(wp.psi) != 0)

    def testGetX(self):
        wp = Wavepacket()
        self.assertIsNotNone(wp.x)
        self.assertTrue(len(wp.x) != 0)

    def testGetNorm(self):
        wp = Wavepacket()
        self.assertIsNotNone(wp.norm2)
        self.assertIsNotNone(wp.norm)

    def testSetGaussianWavepacket(self):
        wp = Wavepacket()
        wp.setToGaussianWavepacket(gaussianSpatialWidth=10)
        self.assertAlmostEqual(np.sum(wp.psi2), 1.0, 4)

    def testGetPsiSquared(self):
        wp = Wavepacket()
        wp.setToGaussianWavepacket(gaussianSpatialWidth=10)
        psi2 = wp.psi2
        self.assertAlmostEqual(np.sum(wp.psi2), 1.0, 4)

    def testCheckIsNormalized(self):
        wp = Wavepacket()
        wp.setToGaussianWavepacket(gaussianSpatialWidth=10)

        self.assertTrue(wp.isNormalized)

    def testCheckIsNormalized(self):
        wp = Wavepacket()
        wp.setToGaussianWavepacket(gaussianSpatialWidth=10)

        self.assertTrue(wp.isNormalized)

if __name__ == "__main__":
    unittest.main()