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

def vol_val(xes: np.array) -> np.array:
    return c(xes)*gm(xes)*np.exp(-integrate_gm_0_to(xes))

delta = 0.01
xes = np.arange(0, 15, delta, dtype=np.float64).reshape(1, -1)
gm_evals = gm(xes)
gm_int = np.cumsum(gm_evals) * delta
# print(f"accurate gm integral: {np.sum(gm_evals) * delta}")
# print(f"accurate total integral: {np.sum(gm_evals * np.exp(-1 * gm_int)) * delta}")

# bin_borders = np.array([0.0, 1.1, 4.6, 6.0, 10.1, 15.0]).reshape(1, -1)
# bin_borders = np.array([0.0, 2.1, 4.6, 8.3, 13.1, 15.0]).reshape(1, -1)
bin_borders = np.random.uniform(0, 15, size=8)
bin_borders = np.concatenate(([0.0], bin_borders, [15.0]))
bin_borders = np.sort(bin_borders).reshape(1, -1)

c_bin_borders = c(bin_borders)
gm_bin_borders = gm(bin_borders)

i_s = np.arange(0, bin_borders.shape[1]-1, dtype=np.int32)
c_id = c(bin_borders[:, i_s])
c_ik = (c(bin_borders[:, i_s+1]) - c(bin_borders[:, i_s])) / (bin_borders[0, i_s+1] - bin_borders[0, i_s])

def c_interpol(xes: np.array) -> np.array:
    i = np.sum(xes >= bin_borders.reshape(-1, 1), axis=0) - 1
    t_i = (xes - bin_borders[0, i]).reshape(-1)
    return c_id[i] + c_ik[i] * t_i


def compute_f_lin_approx():
    masses = integrate_gm_0_to(bin_borders[:, i_s + 1]) - integrate_gm_0_to(bin_borders[:, i_s])
    f_borders = gm(bin_borders).reshape(-1)

    # masses left and right, because f_begin or f_end can be very small -> triangle to middle would
    # not carry much mass. but we can make one of the triangles larger.
    # we don't want to do that if we devide in 2, because that would take away sharpness (?)
    masses_left = masses * (f_borders[i_s] / (f_borders[i_s] + f_borders[i_s + 1]))
    masses_right = masses - masses_left
    deltas = bin_borders[0, i_s + 1] - bin_borders[0, i_s]

    # in case of 3 linear segments, we go down to 0, horizontal and up
    # lin_mass = f_borders * dt_downwards / 2 == masses_left
    dts_downwards = 2 * masses_left / f_borders[i_s]
    # lin_mass = f_borders[i_s + 1] * dts_upwards  / 2 == masses_right
    dts_upwards = 2 * masses_right / f_borders[i_s + 1]

    devide_in_3 = (dts_downwards + dts_upwards) < deltas

    n_f_bins = np.sum(devide_in_3 + 2) + 1
    f_t = np.zeros(n_f_bins)
    d_fi = np.zeros(n_f_bins)
    k_fi = np.zeros(n_f_bins)
    f_i = 0
    for i in range(0, bin_borders.shape[1]-1):
        f_t[f_i] = bin_borders[0, i]
        d_fi[f_i] = f_borders[i]
        f_end = f_borders[i+1]
        if(devide_in_3[i]):
            # add 3 buckets, such that we don't overshoot the mass and stay in positive range

            # down wards
            k_fi[f_i] = -d_fi[f_i] / dts_downwards[i]
            
            # horizontal
            f_t[f_i + 1] = f_t[f_i] + dts_downwards[i]
            d_fi[f_i + 1] = 0
            k_fi[f_i + 1] = 0

            # upwards
            f_t[f_i + 2] = bin_borders[0, i + 1] - dts_upwards[i]
            d_fi[f_i + 2] = 0
            k_fi[f_i + 2] = f_end / dts_upwards[i]

            f_i = f_i + 3
        else:
            # add 2 buckets, such that int is correct
            # f_t_mid = (bin_borders[0, i+1] - bin_borders[0, i]) / 2
            # # lin_mass = d_fi[f_i] * f_t_mid + k_fi * f_t_mid * f_t_mid / 2 == mass / 2
            # k_fi[f_i] = (mass / 2 - d_fi[f_i] * f_t_mid) / (f_t_mid * f_t_mid / 2)
            # d_fi[f_i + 1] = d_fi[f_i] + f_t_mid * k_fi[f_i]
            # k_fi[f_i + 1] = (f_end - d_fi[f_i + 1]) / f_t_mid
            # f_t[f_i + 1] = f_t[f_i] + f_t_mid
            # f_i = f_i + 2
            ml = masses_left[i]
            mr = masses_right[i]
            d1 = f_borders[i]
            d2 = f_borders[i+1]
            tw = deltas[i]

            d0 = (2 * ml + 2 * mr - d1 * tw - d2 * tw + math.sqrt((2 * ml + 2 * mr - d1 * tw - d2 * tw)**2 + 4 * tw * (2 * d2 * ml + 2 * d1 * mr - d1 * d2 * tw)))/(2 * tw)
            assert d0 >= 0
            t2 = 2 * mr / (d0 + d2)
            assert t2 >= 0
            t1 = tw - t2
            assert t1 >= 0
            k1 = (d1 - d0)/t1
            k2 = (d2 - d0)/t2
            k_fi[f_i] = -k1
            k_fi[f_i + 1] = k2
            d_fi[f_i + 1] = d0
            f_t[f_i + 1] = f_t[f_i] + t1
            f_i = f_i + 2
    f_t[f_i] = bin_borders[0, -1]
    return f_t, d_fi, k_fi

f_t, d_fi, k_fi = compute_f_lin_approx()

def f_interpol(xes: np.array) -> np.array:
    i = np.sum(xes >= f_t.reshape(-1, 1), axis=0) - 1
    t_i = (xes.reshape(-1) - f_t[i])
    return d_fi[i] + k_fi[i] * t_i


plt.plot(xes.reshape(-1), gm(xes), label='gm')
# plt.plot(xes.reshape(-1), gm_int, label='gm integrated')
# plt.plot(xes.reshape(-1), integrate_gm_0_to(xes), label='integrate_gm_0_to')
# plt.plot(xes.reshape(-1), np.exp(-1 * gm_int), label='transmittance')
# plt.plot(xes.reshape(-1), gm_evals * np.exp(-1 * gm_int), label='contribution')
# plt.plot(xes.reshape(-1), c(xes), label='c')
# plt.plot(xes.reshape(-1), vol_val(xes), label='vol_val')
# plt.plot(xes.reshape(-1), np.cumsum(vol_val(xes)) * delta, label='vol_int')
# plt.scatter(bin_borders, c_bin_borders, label='c_bin_borders')
plt.scatter(bin_borders, gm_bin_borders, label='gm_bin_borders')
# plt.plot(xes.reshape(-1), c_interpol(xes), label='c_interpol')
plt.plot(xes.reshape(-1), f_interpol(xes), label='f_interpol')
plt.legend()
plt.show()
print("")