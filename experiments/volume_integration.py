import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import math

centroids = np.array([0.4, 2.4, 2.9, 3.5, 5.2, 7.5, 8.4, 9.3]).reshape(-1, 1)
SDs =       np.array([0.4, 0.5, 0.9, 0.7, 0.2, 0.5, 0.8, 1.3]).reshape(-1, 1)
weights =   np.array([0.4, 0.5, 0.9, 0.7, 1.2, 0.5, 1.8, 0.5]).reshape(-1, 1) * 0.5
c_g =       np.array([0.5, 0.2, 0.4, 0.1, 0.2, 0.8, 0.8, 0.3]).reshape(-1, 1)

# centroids = np.array([1.4, 7.4, 7.4, ]).reshape(-1, 1)
# SDs =       np.array([0.8, 0.8, 0.8, ]).reshape(-1, 1)
# weights =   np.array([1.5, 0.5, 0.5, ]).reshape(-1, 1)
# c_g =       np.array([0.2, 0.9, 0.1, ]).reshape(-1, 1)

def g(xes: np.array) -> np.array:
    norm_factor = 1 / (SDs * math.sqrt(2 * math.pi))
    return weights * norm_factor * np.exp(-0.5 * (xes - centroids)*(xes - centroids) / (SDs * SDs))

def gm(xes: np.array) -> np.array:
    return np.sum(g(xes), axis=0)


def gm_cdf(xes: np.array) -> np.array:
    scaling_factor = 1 / (SDs * math.sqrt(2))
    return np.sum(weights * 0.5 * (1 + scipy.special.erf((xes - centroids) * scaling_factor)), axis=0)

def integrate_gm_0_to(xes: np.array) -> np.array:
    return gm_cdf(xes) - gm_cdf(np.zeros_like(xes))

def c(xes: np.array) -> np.array:
    return np.sum(c_g * g(xes)/gm(xes), axis=0)

delta = 0.01
xes = np.arange(0, 10, delta, dtype=np.float64).reshape(1, -1)
gm_evals = gm(xes)
gm_int = np.cumsum(gm_evals) * delta
print(f"accurate gm integral: {np.sum(gm_evals) * delta}")
print(f"accurate total integral: {np.sum(gm_evals * np.exp(-1 * gm_int)) * delta}")

bin_borders = np.array([1.1, 4.6, 6.0, 10.0]).reshape(-1, 1)

plt.plot(xes.reshape(-1), gm_evals, label='gm')
plt.plot(xes.reshape(-1), gm_int, label='gm integrated')
plt.plot(xes.reshape(-1), integrate_gm_0_to(xes), label='gm integrated 2')
plt.plot(xes.reshape(-1), np.exp(-1 * gm_int), label='transmittance')
plt.plot(xes.reshape(-1), gm_evals * np.exp(-1 * gm_int), label='contribution')
plt.plot(xes.reshape(-1), c(xes), label='c')
plt.legend()
plt.show()
print("")