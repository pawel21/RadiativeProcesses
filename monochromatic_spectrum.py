import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
import scipy.special as special
from astropy import constants as const
from astropy import units as u

plt.rcParams.update({'font.size': 32})
T = 2.7
h = const.h
hbar = const.hbar
k = const.k_B
c = const.c
m_e = const.m_e
e = const.e.si
eps0 = const.eps0

r_0 = (e*e)/(4*np.pi*eps0*m_e*c*c)
sigma = (8*np.pi/3)*r_0*r_0


class MonochromaticSpectrumInverseComptonScattering:
    def __init__(self, x, dim_para):
        self.x = x
        self.dimensionless_parameter = dim_para
        self.eps_gamma = (m_e * c * c) / (4 * dim_para)
        self.eps = 2.70 * const.k_B * 3 * u.K
        self.gamma = (m_e * c * c) / (4 * dim_para * self.eps)

    def intensity(self, v, v0, gamma):
        c = const.c
        a = (3 * sigma * c) / (16 * gamma ** 4)
        b = (self.density_of_photon(v0) / v0) * v
        c = 2 * v * np.log(v / (4 * gamma * gamma * v0))

        d = v + 4 * gamma * 2 * v0
        e = (v ** 2) / (2 * gamma * gamma * v0)
        return a * b * (c + d - e)

    def dim_para(self):
        return self.dimensionless_parameter

    def E1(self, x):
        return (self.dim_para()*((1+self.dim_para())**(-1)))*x

    def q(self):
        return self.E1(self.x)/(self.dim_para()*(1-self.E1(self.x)))

    def distribution(self):
        a = 2*self.q()*np.log(self.q())
        b = (1+2*self.q())*(1-self.q())
        c = 0.5*(((self.dim_para()*self.q())**2)/(1+self.dim_para()*self.q()))
        d = 1-self.q()
        return a+b+c*d

    def spectrum(self):
        n = self.density_of_photon(self.eps/h)
        return ((2*np.pi*r_0*m_e*c*c*c*n)/(self.eps_gamma))*self.distribution()

    @staticmethod
    def density_of_photon(v):
        return 1 / (np.exp((h.value * v.value) / (k.value * T)) - 1)


class DifferentialSpectrum:

    def __init__(self, gamma):
        self.gamma = gamma

    @staticmethod
    def density_of_photon(energy):
        A = ((np.pi**2)*((hbar*c)**3))**-1
        return (A*energy*energy) / (np.exp(energy / (k.value * T)) - 1)

    def dN(self, energy_range):
        self.energy_range = energy_range
        A = (2*np.pi*r_0*m_e*c**3)/self.gamma
        B = self.density_of_photon(self.energy_range)/self.energy_range
        self.dim_para = ((4*self.energy_range*self.gamma)/(m_e*c*c)).value
        self.x = 0.5
        C = self.distribution()
        return (A*B*C).value

    def distribution(self):
        a = 2 * self.q() * np.log(self.q())
        b = (1 + 2 * self.q()) * (1 - self.q())
        c = 0.5 * (((self.dim_para * self.q()) ** 2) / (1 + self.dim_para * self.q()))
        d = 1 - self.q()
        return a + b + c * d

    def E1(self, x):
        return (self.dim_para * ((1 + self.dim_para) ** (-1))) * x

    def q(self):
        return self.E1(self.x) / (self.dim_para * (1 - self.E1(self.x)))

x = np.linspace(0.01, 20, 100)*k.value*3
s = DifferentialSpectrum(10**8)
I = integrate.quad(s.dN, 0.001*k.value*3, 20*k.value*3)
plt.loglog(x, s.dN(x))
plt.show()
'''
x = np.linspace(0.1, 0.99, 500)
spec = MonochromaticSpectrumInverseComptonScattering(x, 1)
plt.plot(x, spec.distribution(),  label="$\Gamma=1$", lw=2)
spec = MonochromaticSpectrumInverseComptonScattering(x, 10)
plt.plot(x, spec.distribution(), label="$\Gamma=10$", lw=2)
spec = MonochromaticSpectrumInverseComptonScattering(x, 100)
plt.plot(x, spec.distribution(),  label="$\Gamma=100$", lw=2)
plt.xlabel(r"$\hat{E_1}$")
plt.ylabel(r"$F(\hat{E_1}; \Gamma)$")
plt.ylim([0,4])
plt.legend()
plt.show()

plt.figure()
x = np.linspace(0.1, 0.99, 500)
spec = MonochromaticSpectrumInverseComptonScattering(x, 1)
plt.plot(x, spec.spectrum(),  label="$\Gamma=1$", lw=2)
spec = MonochromaticSpectrumInverseComptonScattering(x, 10)
plt.plot(x, spec.spectrum(), label="$\Gamma=10$", lw=2)
spec = MonochromaticSpectrumInverseComptonScattering(x, 100)
plt.plot(x, spec.spectrum(),  label="$\Gamma=100$", lw=2)
plt.xlabel(r"$\hat{E_1}$")
plt.ylabel(r"$\frac{dN}{dt E_1}$")
plt.legend()
plt.yscale('log')
plt.show()

plt.figure()
spec = MonochromaticSpectrumInverseComptonScattering(x, 10)
E = (spec.gamma * m_e * c * c * spec.E1(x)).to("eV")
plt.plot(E, spec.spectrum(),  label="$\Gamma=10$", lw=2)
plt.xlabel(r"$E_1$ [eV]")
plt.ylabel(r"$\frac{dN}{dt E_1}$")
plt.legend()
plt.yscale('log')
plt.show()
'''
'''plt.figure()
v0 = 1e9*(1/u.s)
v = np.logspace(4, 13, 2000000)*(1/u.s)
Gamma = 10**3
x = v/(Gamma*Gamma*v0)
y = spec.intensity(v, v0, Gamma)
plt.loglog(x, y, 'r-', lw=2)
plt.xlabel(r"Frequency in units of $\frac{\nu}{\gamma \nu_0}$")
plt.ylabel(r"$ I(\nu)$")
plt.show()
'''