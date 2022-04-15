import numpy as np
import matplotlib.pyplot as plt
import math
from math import pi
import scipy
from scipy.signal import butter

def CalcSkz(spectr, FD):
    from math import pi, sqrt

    count = len(spectr)
    def getFreq(n):
        return (n * FD) / count/ 2

    SKZ = 0;
    for i in range(1, count):
        w = 2 * pi * getFreq(i)
        SKZ += (0.707 * spectr[i] / w) ** 2
    SKZ = sqrt(SKZ)

    return SKZ

def Rescale(array, number):
    import numpy as np

    B = 0.53836/0.23082
    result = np.zeros(number-2)
    for i in range(number-2):
        result[i] = (array[i] + B*array[i+1] + array[i+2]) / B

    return result

def CalculateSpectr(Signal):

    import numpy as np
    from math import pi, cos
    import matplotlib.pyplot as plt

    count = len(Signal)

    # наложение окна
    for i in range(count):
        Signal[i] *= 0.54 - 0.46 * cos((2 * pi * i) / (count - 1))


    spectr = np.fft.rfft(Signal)
    spectr = np.abs(spectr)
    spectr = spectr / count * 2

    # норморовка
    spectr = Rescale(spectr, len(spectr))

    return spectr

def Rescale(array, number):
    import numpy as np

    B = 0.53836/0.23082
    result = np.zeros(number-2)
    for i in range(number-2):
        result[i] = (array[i] + B*array[i+1] + array[i+2]) / B

    return result

class VibrationRMSMeasureHelper():

    def __init__(self, printPlot = 0):
        """Constructor"""

        self.printPlot = printPlot
        self.originalFD = 32000

    def GetAdcValue(self, A):
        return A / 9.8 / 0.488

    def GetSignal(self, a, f, FD, COUNT, noise, phase=0):
        x = 0
        sig = np.zeros(COUNT)
        for i in range(COUNT):
            sig[i] = a * math.sin(x * f * 2 * pi + phase)
            x += (1. / FD)

        if (noise):
            sig += noise[:COUNT]

        return sig

    def CutSignal(self, signal, delimer, count):
        result = np.zeros(count)

        for i in range(count):
            result[i] = signal[i * delimer]
        return result

    def GetFilter(self, size):
        cutoff = 1100
        filter = scipy.signal.firwin(size, cutoff=cutoff, fs= self.originalFD)
        return filter

    # фильтрация сигнала
    def FilterWindow(self, data, filterSize):
        N = len(data)
        out = np.zeros(N - filterSize)

        filter = self.GetFilter(filterSize)

        for i in range(len(out)):
            temp = data[i:i + filterSize]
            out[i] = sum(temp * filter)

        return out

    def GetVelocityRms(self, signal, FD, signalPlot = 0, spectrPlot = 0):
        data = np.copy(signal)

        #  преобразование в мм/с2
        average = np.mean(data)
        for i in range(len(data)):
            data[i] -= average
            data[i] *= 0.488  # to mg
            data[i] *= 9.8

        if (self.printPlot):
            signalPlot.plot(data)

        spectr = CalculateSpectr(data)

        # Обнуление шума
        for i in range(0, len(spectr)):
            if (spectr[i] < 15):
                spectr[i] = 0

        SKZ = CalcSkz(spectr, FD)

        f = np.fft.rfftfreq(len(spectr) * 2 - 1, 1. / FD)
        signalFreq = f[np.argmax(spectr)]

        if (self.printPlot):
            signalPlot.set_title("f =" + str(round(signalFreq, 3)) + " RMS = " + str(round(SKZ, 3)))
            signalPlot.grid()

            spectrPlot.plot(f, spectr)
            spectrPlot.grid()

        return round(SKZ, 3)

try:
    rawData = np.loadtxt("py\\logFile.txt")
except:
    rawData = np.loadtxt("logFile.txt")

fig, axs = plt.subplots(2, 3, figsize=(8, 8))
column = 0

enablePlot = True
helper = VibrationRMSMeasureHelper(enablePlot)

# данные без фильтрации с FD 13кГц
rms1 = helper.GetVelocityRms(rawData, helper.originalFD, axs[0][column], axs[1][column])
column += 1

# данные без фильтрации с FD 1,3кГц
signal = helper.CutSignal(rawData, 3, 4096)
rms2 = helper.GetVelocityRms(signal, 8000/3, axs[0][column], axs[1][column])
column += 1

#  фильтрация 300 с FD 1,3кГц
signal = helper.CutSignal(helper.FilterWindow(rawData, 300), 3, 4096)
rms3 = helper.GetVelocityRms(signal, 8000/3, axs[0][column], axs[1][column])
column += 1

plt.show()
