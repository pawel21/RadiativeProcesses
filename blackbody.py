import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
import scipy.special as special
from astropy import constants as const
from astropy import units as u

T = 2.7
h = const.h
k = const.k_B
c = const.c
m_e = const.m_e
e = const.e.si


class BlackBody:
    def __init__(self, T):
        self.beta = (k*T)**-1

    def brightness(self, v):
        b_v = ((2*h*v**3)/(c**2))/(np.exp(self.beta.value*h.value*v)-1)
        return b_v.value

    @staticmethod
    def density_of_photon(e):
        return 1 / (np.exp((e) / (k.value * T)) - 1)


spec = BlackBody(3)
v = np.linspace(1e-4, 1e2, 10000)
b_v = spec.density_of_photon(v)
I = integrate.quad(spec.brightness, 1e-4, 1e13)
#u = (4*np.pi/c)*I

fig, ax1 = plt.subplots()
ax1.loglog(v, b_v)

plt.show()