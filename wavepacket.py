import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import numpy as np
from scipy.signal import hilbert, chirp
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# You could import materials from Raytracing:
# from raytracing.materials import *
# print(Material.all())

# Shortcuts
I = complex(0, 1)
Ï = np.pi
c = 3e8


class Wavepacket:
    def __init__(self, ğ = None, ğâ = None, field=None, time=None):

        if ğ is not None:
            self.ğâ = ğâ
            self.kâ = 2 * Ï / ğâ
            self.ğâ = self.kâ * c
            self.fâ = self.ğâ / 2 / Ï

            self.time, self.field = self.computeField(ğ, ğâ)
        else:
            self.ğâ = None
            self.kâ = None
            self.ğâ = None
            self.fâ = None

            self.field = field
            self.time = time

        self.distancePropagated = 0

    def computeField(self, ğâ, ğâ):
        N = 1024 * 16
        S = 40

        t = np.linspace(-ğâ * S, ğâ * S, N)
        field = np.exp(-(t * t) / (ğâ * ğâ)) * np.cos(self.ğâ * t)
        return t, field        

    @property
    def dt(self):
        return self.time[1] - self.time[0]

    @property
    def frequencies(self):
        return np.fft.fftfreq(len(self.field), self.dt)

    @property
    def wavelengths(self):
        return c / (self.frequencies + 0.01)  # avoid c/0

    @property
    def spectrum(self):
        return np.fft.fft(self.field)

    @property
    def spectralWidth(self):
        frequencies = self.frequencies
        positiveFrequencies = np.extract(frequencies > 0, frequencies)
        amplitudes = np.extract(frequencies > 0, abs(self.spectrum))

        return self.rms(positiveFrequencies, amplitudes)

    @property
    def temporalWidth(self):
        return self.rms(self.time, self.fieldEnvelope)

    def rms(self, x, y):
        sumY = np.sum(y)
        meanX = np.sum(x * y) / sumY
        meanX2 = np.sum(x * x * y) / sumY
        return np.sqrt(meanX2 - meanX * meanX)

    @property
    def timeBandwidthProduct(self):
        return 2 * Ï * self.spectralWidth * self.temporalWidth

    @property
    def fieldEnvelope(self):
        return np.abs(self.analyticSignal)

    def instantRadFrequency(self):
        # Extract envelope and carrier
        analyticSignal = self.analyticSignal

        instantEnvelope = np.abs(analyticSignal)
        instantPhase = np.unwrap(np.angle(analyticSignal))
        instantRadFrequency = np.diff(instantPhase) * 1 / self.dt

        instantRadFrequency = np.extract(
            instantEnvelope[0:-1] > 0.001, instantRadFrequency
        )
        instantTime = np.extract(instantEnvelope[0:-1] > 0.001, self.time)
        instantPhase = np.extract(instantEnvelope[0:-1] > 0.001, instantPhase)
        instantEnvelope = np.extract(instantEnvelope[0:-1] > 0.001, instantEnvelope)

        return instantTime, instantEnvelope, instantPhase, instantRadFrequency

    @property
    def analyticSignal(self):
        analyticSignal = hilbert(self.field.real)

        # Center maximum at t=0
        maxIndex = np.argmax(np.abs(analyticSignal))
        centerIndex = len(analyticSignal) // 2
        deltaRoll = centerIndex - maxIndex
        analyticSignal = np.roll(analyticSignal, deltaRoll)
        return analyticSignal

    def propagate(self, d):
        if np.mean(self.field[0:10]) > 2e-2:
            print("Warning: temporal field reaching edges")

        ğ = np.array([2 * Ï / ğ * d for ğ in self.wavelengths])

        phaseFactor = np.exp(I * ğ)
        field = np.fft.fft(self.field)
        field *= phaseFactor
        field = np.fft.ifft(field)

        self.field = field
        self.distancePropagated += d

        return self.time, field

    def setupPlot(self, title=""):
        plt.style.use(
            "https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle"
        )
        plt.title(title)
        plt.xlabel("Time [ps]")
        plt.ylabel("Amplitude [arb.u.]")
        plt.ylim(-1, 1)

        axis = plt.gca()
        axis.text(
            0.05,
            0.95,
            "Distance = {2:.0f} mm\n$\Delta t$ = {0:.0f} fs\n$\Delta \omega \\times \Delta t$ = {1:0.2f}".format(
                self.temporalWidth * 1e15,
                self.timeBandwidthProduct,
                self.distancePropagated * 1e3,
            ),
            transform=axis.transAxes,
            fontsize=14,
            verticalalignment="top",
        )

    def tearDownPlot(self):
        plt.clf()

    def drawEnvelope(self, axis=None):
        if axis is None:
            axis = plt.gca()

        timeIsPs = self.time * 1e12
        axis.plot(timeIsPs, self.fieldEnvelope, "k-")

    def drawField(self, axis=None):
        if axis is None:
            axis = plt.gca()

        (
            instantTime,
            instantEnvelope,
            instantPhase,
            instantRadFrequency,
        ) = self.instantRadFrequency()

        timeIsPs = instantTime * 1e12
        axis.plot(timeIsPs, instantEnvelope * np.cos(instantPhase), "k-")

    def drawChirpColour(self, axis=None):
        if axis is None:
            axis = plt.gca()

        (
            instantTime,
            instantEnvelope,
            instantPhase,
            instantRadFrequency,
        ) = self.instantRadFrequency()

        # We want green for the center frequency (+0.33)
        normalizedFrequencyForColor = (instantRadFrequency - self.ğâ) / (
            5 * 2 * Ï * self.spectralWidth
        ) + 0.33

        hsv = cm.get_cmap("hsv", 64)
        M = 128

        instantTimeInPs = instantTime * 1e12
        step = len(instantTimeInPs) // M
        for i in range(0, len(instantTimeInPs) - step, step):
            t1 = instantTimeInPs[i]
            t2 = instantTimeInPs[i + step]
            c = normalizedFrequencyForColor[i + step // 2]
            e1 = instantEnvelope[i]
            e2 = instantEnvelope[i + step]
            axis.add_patch(
                Polygon([(t1, 0), (t1, e1), (t2, e2), (t2, 0)], facecolor=hsv(c))
            )


if __name__ == "__main__":

    # All adjustable parameters below
    pulse = Pulse(ğ=5e-10, ğâ=800e-9)

    # Material propertiues and distances, steps
    material = pulse.vacuum
    totalDistance = 1e-2
    steps = 40

    # What to display on graph in addition to envelope?
    adjustTimeScale = True
    showCarrier = True

    # Save graph? (set to None to not save)
    filenameTemplate = "fig-{0:02d}.png" # Can use PDF but PNG for making movies with Quicktime Player

    # End adjustable parameters

    print("#\td[mm]\tât[ps]\tâğ[THz]\tProduct")
    stepDistance = totalDistance / steps
    for j in range(steps):
        print(
            "{0}\t{1:.1f}\t{2:0.3f}\t{3:0.3f}\t{4:0.3f}".format(
                j,
                pulse.distancePropagated * 1e3,
                pulse.temporalWidth * 1e12,
                2 * Ï * pulse.spectralWidth * 1e-12,
                pulse.timeBandwidthProduct,
            )
        )

        pulse.setupPlot("Propagation in {0}".format(material.__func__.__name__))
        # pulse.drawEnvelope()
        pulse.drawChirpColour()
    
        if showCarrier:
            pulse.drawField()

        if adjustTimeScale:
            ğ = pulse.temporalWidth*1e12
            plt.xlim(-1, 1)
        
        plt.draw()
        plt.pause(0.001)

        if filenameTemplate is not None:
            plt.savefig(filenameTemplate.format(j), dpi=300)
        pulse.tearDownPlot()

        pulse.propagate(stepDistance, material)
