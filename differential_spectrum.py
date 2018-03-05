import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
import scipy.special as special
from astropy import constants as const
from astropy import units as u

plt.rcParams.update({'font.size': 30})
T = 2.7
h = const.h
hbar = const.hbar
k = const.k_B
c = const.c
m_e = const.m_e
e = const.e.si
eps0 = const.eps0
k = const.k_B

r_0 = (e*e)/(4*np.pi*eps0*m_e*c*c)


class DifferentialSpectrum:

    def __init__(self, gamma, E1):
        self.gamma = gamma
        self.E1 = E1

    @staticmethod
    def density_of_photon(energy):
        A = ((np.pi**2)*((hbar*c)**3))**-1
        kT = (k*T*u.K).to(u.J)
        return (A*energy*energy) / (np.exp(energy / (kT.value)) - 1)

    def dN_dt(self, eps):
        c = const.c
        A = (2*np.pi*r_0*r_0*m_e*c**3) / (self.gamma) # to eV * *1.602*(10**-19)
        param = (4 * eps * self.gamma) / ((m_e * c * c))
        self.dim_para = param.value
        B = self.density_of_photon(eps) / eps

        #here we calcurate integral
        a = 2 * self.q() * np.log(self.q())
        b = (1 + 2 * self.q()) * (1 - self.q())
        c = 0.5 * (((self.dim_para * self.q())**2) / (1 + self.dim_para * self.q()))
        d = 1 - self.q()
        return (A*B*(a+b+c*d)).value

    def q(self):
        return self.E1 / (self.dim_para * (1 - self.E1))

'''
gamma = 10**4
E_gamma = np.logspace(1, 10.70, 1500)*u.eV
E1 = E_gamma/(((m_e*c*c).to(u.eV))*gamma)
print(E1)
N = np.zeros(len(E_gamma))

for i in range(0, len(E1)):
    s = DifferentialSpectrum(gamma, E1[i])
    N[i] = integrate.fixed_quad(s.dN_dt, 0.0001*((k*T*u.K).to(u.J)).value, 600*((k*T*u.K).to(u.J)).value)[0]

print(N)
plt.loglog(E_gamma, N, 'bo', label="$\gamma = 10^4$", markersize=10)


gamma = 10**5
E_gamma = np.logspace(1.0, 11.7, 1500)*u.eV
E1 = E_gamma/(((m_e*c*c).to(u.eV))*gamma)
print(E1)
N = np.zeros(len(E1))
for i in range(0, len(E1)):
    s = DifferentialSpectrum(gamma, E1[i])
    n, err = integrate.fixed_quad(s.dN_dt, 0.0001*((k*T*u.K).to(u.J)).value, 600*((k*T*u.K).to(u.J)).value)
    N[i] = n
print(N)
x = E1*((m_e*c*c).to(u.eV))*gamma
plt.loglog(x, N, 'r>', label="$\gamma = 10^{5}$", markersize=10)
plt.legend()
plt.xlabel("Energy, $E_\gamma$ [eV]")
plt.ylabel(r"$\frac{dN}{dt}$")
plt.title("T = 2.7 K")
plt.grid(True)
'''

gamma = 2*10**7
E_gamma = np.logspace(1.0, 11.7, 1500)*u.eV
E1 = E_gamma/(((m_e*c*c).to(u.eV))*gamma)
print(E1)
N = np.zeros(len(E1))
for i in range(0, len(E1)):
    s = DifferentialSpectrum(gamma, E1[i])
    n, err = integrate.fixed_quad(s.dN_dt, 0.0001*((k*T*u.K).to(u.J)).value, 600*((k*T*u.K).to(u.J)).value)
    N[i] = n
print(N)
x = E1*((m_e*c*c).to(u.eV))*gamma
plt.loglog(x, N, 'g^', label="$\gamma = 10^{6}$", markersize=10)
plt.legend()
plt.xlabel("Energy, $E_\gamma$ [eV]")
plt.ylabel(r"$\frac{dN}{dt}$")
plt.title("T = 300 K")
plt.grid(True)


'''
gamma = 10**12
E1 = np.linspace(0.2, 0.99999999, 50000)

N = np.zeros(len(E1))
for i in range(0, len(E1)):
    s = DifferentialSpectrum(gamma, E1[i])
    n, err = integrate.fixed_quad(s.dN_dt, 0.000001*((k*T*u.K).to(u.J)).value, 600*((k*T*u.K).to(u.J)).value)
    N[i] = n
print(N)
x = E1*((m_e*c*c).to(u.eV))*gamma
plt.loglog(x, N, 'b>', label="$\gamma = 10^{12}$", markersize=10)
plt.legend()
plt.xlabel("Energy, $E_\gamma$ [eV]")
plt.ylabel(r"$\frac{dN}{dt}$")
plt.title("T = 2.7 K")
plt.grid(True)


gamma = 10**14
E1 = np.linspace(0.2, 0.99999999, 50000)
N = np.zeros(len(E1))
for i in range(0, len(E1)):
    s = DifferentialSpectrum(gamma, E1[i])
    n, err = integrate.fixed_quad(s.dN_dt, 0.000001*((k*T*u.K).to(u.J)).value, 600*((k*T*u.K).to(u.J)).value)
    N[i] = n
print(N)
x = E1*((m_e*c*c).to(u.eV))*gamma
plt.loglog(x, N, 'ro', label="$\gamma = 10^{14}$", markersize=10)
plt.legend()
plt.xlabel("Energy, $E_\gamma$ [eV]")
plt.ylabel(r"$\frac{dN}{dt}$")
plt.title("T = 300 K")
plt.grid(True)
'''
plt.show()
