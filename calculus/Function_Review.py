#%%

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)
y = 0.5 * x + 1


plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.grid()
plt.plot(x, y);

#%%

x = np.linspace(-2, 2, 100)
y = -0.5 * 9.8 * x ** 2 + 2 * x + 1


plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.grid()
plt.plot(x, y);

#%%

x = np.linspace(0, 10, 100)
y = -0.5 * x ** (1./3) + 0.2 * x ** (1./2) + 1

plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.grid()
plt.plot(x, y);

#%%

x = np.linspace(-5, 5, 100)
y = np.exp(x) # e ** x


plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.grid()
plt.plot(x, y);

#%%

x = np.linspace(-5, 5, 100)
y = np.exp(x) # e ** x
y2 = x**2


plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.grid()
plt.plot(x, y, label='e^x');
plt.plot(x, y2, label='x^2');
plt.legend();

#%%

x = np.linspace(0.0000001, 10, 200)
y = np.log(x) # ln(x)

plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.grid()
plt.plot(x, y);

#%%

x = np.linspace(-5 * np.pi, 5 * np.pi, 200)
y = np.sin(x)

plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.grid()
plt.plot(x, y);

#%%

x = np.linspace(-2 * np.pi, 2 * np.pi, 200)
y = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)

plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.grid()
plt.plot(x, y, label='sin');
plt.plot(x, y2, label='cos');
plt.plot(x, y3, label='tan');
plt.legend();

#%%

x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
y = np.tan(x)

plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.grid()
plt.plot(x, y);

#%%

x = np.linspace(-2 * np.pi, 2 * np.pi, 100000)
y = np.tan(x)

plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.grid()
plt.plot(x, y);

#%%

x = np.linspace(-5, 5, 100)
y = np.sinh(x)

plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.grid()
plt.plot(x, y);

#%%

x = np.linspace(-5, 5, 100)
y = np.sinh(x)
y2 = np.cosh(x)

plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.grid()
plt.plot(x, y, label='sinh');
plt.plot(x, y2, label='cosh');
plt.legend();

#%%

x = np.linspace(-5, 5, 100)
y = np.tanh(x)

plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.plot(x, y, label='tanh');
plt.grid()
plt.plot(x, y);








