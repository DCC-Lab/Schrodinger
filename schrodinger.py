import numpy as np
import scipy.integrate as spi
from scipy.sparse import diags
from scipy.constants import hbar, m_e, elementary_charge, Planck, c
from scipy.sparse.linalg import eigs
from findiff import FinDiff
import matplotlib.pyplot as plt
from threading import Lock

""" 
Reference
https://medium.com/@mathcube7/two-lines-of-python-to-solve-the-schrödinger-equation-2bced55c2a0e
"""

INFINITY = 100000
Ksch = 0.5*hbar*hbar/m_e/elementary_charge/1e-20 # in eV*Angstrom^2
lock = Lock()

class Wavefunction:
    x = np.linspace(-10,10,1001)

    def __init__(self, psi = None, label=r"$\psi$"):
        super().__init__()
        if psi is None:
            psi = np.zeros(len(self.x),dtype=complex)

        self.label = label
        self.matrix = np.array(psi, dtype=complex)

    @property
    def dx(self):
        """ This is the differential element dx for our x vector"""
        return self.x[1]-self.x[0]

    def normalize(self):
        norm2 = self.norm2()
        if norm2 != 0:
            self.matrix /= np.sqrt(norm2)
        else:
            raise ValueError("Wavefunction is not normalizable because it is null")

    def norm2(self):
        return spi.trapezoid(np.abs(self.matrix) **2, x=self.x, dx=self.dx)

    def is_normalized(self):
        if abs(self.norm2() - 1.0).real < 1e-4:
            return True
        else:
            return False

    @classmethod
    def gaussian(cls, sigma, normalized=True):
        psi = Wavefunction(psi=np.exp(-cls.x*cls.x/sigma/sigma))

        if normalized:
            psi.normalize()

        return psi

    def add_to_plot(self, axis):
        axis.plot(self.x, np.real(self.matrix), label=self.label)

    def show(self):
        fig, axis = plt.subplots()
        self.add_to_plot(axis)
  
        axis.grid()
        axis.legend()
        axis.set_ylabel("Wavefunction [arb.u.]")
        axis.set_xlabel(r"Distance [$\AA$]")
        axis.set_xlim(min(self.x), max(self.x))

        plt.show()

class Operator:
    def __init__(self, matrix=None, label=""):
        if matrix is None:
            self.matrix = np.identity(n=len(self.x))
        else:
            self.matrix = matrix

        self._eigenvalues = None
        self._eigenvectors = None
        self.label = label

    @property
    def x(self):
        return Wavefunction.x

    @property
    def dx(self):
        """ This is the differential element dx for our x vector"""
        return self.x[1]-self.x[0]

    def eigenstates(self, k=3, which='SR'):
        if self._eigenvalues is None or self._eigenvectors is None:
            self.compute_eigenstates(k=k, which=which)

        return self._eigenvalues, self._eigenvectors

    def compute_eigenstates(self, k=3, which='SR'):
        while k > 0:
            try:
                eigenvalues, eigenvectors = eigs( self.matrix , k=k, which=which)
                break
            except np.linalg.LinAlgError as err:
                print(err)
                k -= 1
                if k == 0:
                    raise ValueError("No eigenstates found")

        self._eigenvalues = eigenvalues.real
        self._eigenvectors = []
        for i in range(eigenvectors.shape[1]):
            self._eigenvectors.append(Wavefunction(psi=eigenvectors[:,i], label=r"$\psi_{{{0}}}$".format(i)))

def D_Dx(vector=None):
    dx = Wavefunction.x[1] - Wavefunction.x[0]
    operator_matrix = FinDiff(0, dx, 1).matrix(Wavefunction.x.shape)


    if vector is not None:
        theClass = type(vector)
        return theClass(operator_matrix * vector.matrix)
    else:
        return Operator(operator_matrix)


def D2_Dx2(vector=None):
    dx = Wavefunction.x[1] - Wavefunction.x[0]
    operator_matrix = FinDiff(0, dx, 2).matrix(Wavefunction.x.shape)

    if vector is not None:
        theClass = type(vector)
        return theClass(operator_matrix * vector.matrix)
    else:
        return Operator(operator_matrix)

class Potential(Operator):
    """ The class describes several potential that we encounter in quantum mechanics.
    It makes use of the default x from Operator.x """

    def __init__(self, values=None, label=None):
        if values is None:
            values = np.zeros((len(Wavefunction.x),))

        self.values = values
        super().__init__( diags(self.values), label=label )

    def add_to_plot(self, axis):
        axis.plot(self.x, np.real(self.values), "k--", label=self.label)

    def show(self):
        fig, axis = plt.subplots()
        self.add_to_plot(axis)

        axis.grid()
        axis.legend()
        axis.set_ylabel("Energy [eV]")
        axis.set_xlabel(r"Distance [$\AA$]")
        axis.set_xlim(min(self.x), max(self.x))

        plt.show()

    @classmethod
    def infinite_well(cls, a):
        """ This sets to potential to a infinite well of width a """
        v = np.zeros((len(Wavefunction.x),))

        for i, x in enumerate(Wavefunction.x):
            if abs(x) >= abs(a)/2:
                v[i]   = INFINITY

        return Potential(v, label="Infinite well")

    @classmethod
    def finite_well(cls, a, vo):
        """ This sets to potential to a finite well of width a and depth vo"""
        v = np.zeros((len(Wavefunction.x),))

        for i, x in enumerate(Wavefunction.x):
            if abs(x) <= abs(a)/2:
                v[i] = -abs(vo)

        return Potential(v, label=r"Finite well of width {0:.1f} $\AA$, depth $V_o = {1} eV$".format(a, vo))

    @classmethod
    def two_finite_wells(cls, a, b, vo):
        """ This sets to potential to two finite wells of width a,separated by b and depth vo"""
        v = np.zeros((len(Wavefunction.x),))

        for i, x in enumerate(Wavefunction.x):
            if abs(x) >= abs(b/2) and abs(x) <= abs(b/2+a):
                v[i] = -abs(vo)

        return Potential(v, label=r"Two well a={0:.1f} $\AA$, b={1:.1f} $\AA$, $V_o = {2} eV$".format(a, b, vo))
    
    @classmethod
    def periodic_wells(cls, a, b, vo):
        """Creates periodic potential wells of width a, separated by b and depth vo"""
        x = Wavefunction.x
        v = np.zeros_like(x)

        L = a + b  # period length

        for i, xi in enumerate(x):
            pos = (xi % L + L) % L
            if pos <= a:
                v[i] = abs(vo)
            else:
                v[i] = 0
                
        return Potential(v, label=f"Periodic wells: a={a}, b={b}, V0={vo}")

    @classmethod
    def finite_barrier(cls, a, vo):
        """ This sets to potential to a finite well of width a and depth vo"""
        v = np.zeros((len(Wavefunction.x),))

        for i, x in enumerate(Wavefunction.x):
            if abs(x) <= abs(a)/2:
                v[i] = abs(vo)

        return Potential(v, label=r"Finite barrier of width {0:.1f} $\AA$, height $V_o = {1} eV$".format(a, vo))

    @classmethod
    def harmonic_well(cls, omega=0.5):
        """ This sets to potential to a quadratic well of constant V(x) = omega * x^2 """
        x = Wavefunction.x
        v = omega*x*x

        return Potential(v, label="Harmonic well")

    @classmethod
    def harmonic_halfwell(cls, omega=0.5):
        """ This sets to potential to a quadratic half-well of constant V(x) = omega * x^2 """
        v = omega*Wavefunction.x*Wavefunction.x

        for i, x in enumerate(Wavefunction.x):
            if x < 0:
                v[i] = INFINITY

        return Potential(v, label="Harmonic half-well")

    @classmethod
    def delta_barrier(cls, alpha=1.0):
        """ This sets to potential to an infinite potentiel at x = 0 """
        v = np.zeros((len(Wavefunction.x),))

        for i, x in enumerate(Wavefunction.x):
            if x >= 0:
                v[i] = INFINITY
                break

        return Potential(v, label="Delta barrier")

    @classmethod
    def delta_well(cls, alpha=1.0):
        """ This sets to potential to an negative infinite potentiel at x = 0 """
        v = np.zeros((len(Wavefunction.x),))

        for i, x in enumerate(Wavefunction.x):
            if x >= 0:
                v[i] = -INFINITY
                break

        return Potential(v, label="Delta well")

class Hamiltonian(Operator):
    def __init__(self, potential=None):
        super().__init__()
        if potential is None:
            self.potential = Potential()
        else:
            self.potential = potential

        self.matrix = ( - Ksch*D2_Dx2().matrix + self.potential.matrix )

    def show_eigenstates(self, which=None, probability=False):
        if which is not None:
            k = max(which)+1
        else:
            k = 3

        energies, eigenstates = self.eigenstates(k=k)

        eMin = min(energies.real)
        eMax = max(energies.real)
        deltaE = eMax - eMin

        fig,ax = plt.subplots()

        if which is None:
            which = range(len(eigenstates))

        for i in which:
            if i == 0 and len(which) == 1:
                delta = 0
            elif i == 0:
                delta = energies[1]-energies[0]
            elif i == len(eigenstates)-1:
                delta = energies[i]-energies[i-1]
            else:
                deltaPlus = energies[i+1]-energies[i]
                deltaMinus = energies[i]-energies[i-1]
                delta = min(deltaPlus, deltaMinus)

            if probability:
                scaling = 0.5*delta/max(abs(eigenstates[i].matrix)**2)
                ax.plot(self.x, np.abs(eigenstates[i].matrix)**2 * scaling + energies[i], label=r'$\psi_{0}$'.format(i))
            else:
                scaling = 0.5*delta/max(abs(eigenstates[i].matrix.real))
                ax.plot(self.x, eigenstates[i].matrix.real * scaling + energies[i], label=r'$\psi_{0}$'.format(i))
        self.potential.add_to_plot(ax)
        ax.set_ylim(eMin-deltaE*0.5, eMax+deltaE*0.5)
  
        ax.grid()
        ax.legend()
        ax.set_ylabel("Energy [eV]")
        ax.set_xlabel(r"Distance [$\AA$]")
        ax.set_xlim(min(self.x), max(self.x))
        plt.show()


def infrared_qwlaser(vo):
    Wavefunction.x = np.linspace(-100,100,1001)

    for a in np.linspace(28, 35, 16):
        try:
            with lock:
                h = Hamiltonian(Potential.finite_well(a=a, vo=vo))

            # h = Hamiltonian(Potential.infinite_well(a=a))
            energies, eigenstates = h.eigenstates(k=2)
            print("{0:.2f}\t{1:.3}".format(a, (energies[1]-energies[0])))
            #h.show_eigenstates(probability=False)
        except Exception as err:
            print("No states for {0} [{1}]".format(a,err))
            


def infrared_qwlaser_find(vo, target_diff_in_eV = 0.001, wavelength = 10.6e-6):
    with lock:
        dx = Wavefunction.x[1]-Wavefunction.x[0]

    target_laser_energy = Planck * c /wavelength/elementary_charge

    try:
        a = 28.0
        da = 0.1
        previous_diff = 10
        diff = 10
        iterations = 0
        while abs(diff) > target_diff_in_eV and iterations <= 2:
            a += da
            h = Hamiltonian(Potential.finite_well(a=a, vo=vo))
            energies, eigenstates = h.eigenstates(k=2)
            current_laser_energy = energies[1]-energies[0]
            
            diff = current_laser_energy-target_laser_energy
            if diff * previous_diff < 0:
                if da == dx:
                    iterations += 1
                    da = -dx/2
                elif da == -dx:
                    iterations += 1
                    da = dx
                else:
                    da = - da/2
            else:
                da = 1.23*da
                iterations = 0

            previous_diff = diff
        return vo, a, current_laser_energy
    except Exception as err:
        print("No states for {0} [{1}]".format(a,err))
        return None, None, None


def infrared_qw_well_laser_at_10_6():
    Wavefunction.x = np.linspace(-40,40,501)
    wavelength = 10.6e-6
    laser_energy_in_eV = Planck * c /wavelength/elementary_charge
    a_inf = np.sqrt((2**2-1**2)*(3.1416**2)*Ksch/laser_energy_in_eV)
    print("Width of infinite well {1:.3f} Å for transition of {0:.3f} eV (theoretical)".format(laser_energy_in_eV, a_inf))

    arg_vo = [1,3,10,30,100,300,1000,3000]
    arg_diff = [0.001]*len(arg_vo)
    args = zip(arg_vo, arg_diff)
    pairs = []

    for vo, diff in args:
        pair = infrared_qwlaser_find(vo=vo, target_diff_in_eV=diff)
        pairs.append(pair)

    for vo, a, E in pairs:
        print("{0}\t{1}\t{2}".format(vo, a,E))

if __name__ == "__main__":
    # infrared_qw_well_laser_at_10_6()
    Wavefunction.x = np.linspace(-120,120,2001)
    h = Hamiltonian(Potential.periodic_wells(a=5, b=15, vo=1.5))
    h.show_eigenstates(which=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])