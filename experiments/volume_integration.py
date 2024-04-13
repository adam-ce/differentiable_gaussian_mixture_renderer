import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import math

# # complex example
# centroids = np.array([2.4, 0.4, 2.9, 3.5, 5.2, 7.5, 8.4, 9.3]).reshape(-1, 1)
# SDs =       np.array([0.5, 0.4, 0.9, 0.7, 0.2, 0.5, 0.8, 1.3]).reshape(-1, 1)
# weights =   np.array([0.5, 0.4, 0.9, 0.7, 1.2, 0.5, 1.8, 1.7]).reshape(-1, 1) * 0.5
# c_g =       np.array([0.2, 0.5, 0.4, 0.1, 0.2, 0.8, 0.8, 0.3]).reshape(-1, 1)


# complex example 2
centroids = np.array([2.4, 0.4, 2.7, 2.9, 3.5, 2.9, 5.2, 7.5, 6.9, 8.4, 8.0, 9.3]).reshape(-1, 1)
SDs =       np.array([0.5, 0.4, 0.3, 0.9, 0.7, 0.9, 0.2, 0.5, 1.9, 0.8, 0.4, 1.3]).reshape(-1, 1)
weights =   np.array([0.5, 0.4, 0.5, 0.9, 0.7, 0.9, 1.2, 0.5, 0.9, 1.8, 0.9, 1.7]).reshape(-1, 1) * 0.5
c_g =       np.array([0.2, 0.5, 0.9, 0.4, 0.1, 0.4, 0.2, 0.8, 0.2, 0.8, 0.4, 0.3]).reshape(-1, 1)

# demo example
# centroids = np.array([1.8, 8.0, 12.0]).reshape(-1, 1)
# SDs =       np.array([0.9, 0.7, 1.3]).reshape(-1, 1)
# weights =   np.array([0.5, 0.5, 0.5]).reshape(-1, 1) * 0.5
# c_g =       np.array([0.4, 0.8, 0.3]).reshape(-1, 1)

# centroids = np.array([1.4, 7.4, 7.4, ]).reshape(-1, 1)
# SDs =       np.array([0.8, 0.8, 0.8, ]).reshape(-1, 1)
# weights =   np.array([1.5, 0.5, 0.5, ]).reshape(-1, 1)
# c_g =       np.array([0.2, 0.9, 0.1, ]).reshape(-1, 1)

def g(xes: np.array) -> np.array:
    norm_factor = 1 / (SDs * math.sqrt(2 * math.pi))
    return weights * norm_factor * np.exp(-0.5 * (xes - centroids)*(xes - centroids) / (SDs * SDs))

def gm(xes: np.array) -> np.array:
    return np.sum(g(xes), axis=0)

def g_cdf(xes: np.array) -> np.array:
    scaling_factor = 1 / (SDs * math.sqrt(2))
    return weights * 0.5 * (1 + scipy.special.erf((xes - centroids) * scaling_factor))

def gm_cdf(xes: np.array) -> np.array:
    return np.sum(g_cdf(xes), axis=0)

def integrate_g_0_to(xes: np.array) -> np.array:
    return g_cdf(xes) - g_cdf(np.zeros_like(xes))

def integrate_gm_0_to(xes: np.array) -> np.array:
    return gm_cdf(xes) - gm_cdf(np.zeros_like(xes))

def c(xes: np.array) -> np.array:
    return np.sum(c_g * g(xes)/gm(xes), axis=0)

def vol_val(xes: np.array) -> np.array:
    return c(xes)*gm(xes)*np.exp(-integrate_gm_0_to(xes))

def compute_bin_borders_1(n_bins: int) -> np.array:
    transmission_threshold = 0.05
    centr_p_k_sig = (centroids + SDs * 2).reshape(-1)
    g_ints = integrate_g_0_to(centr_p_k_sig.reshape(-1, 1)).reshape(-1)
    ordered_indices = np.argsort(centr_p_k_sig)
    transmittance = 1
    transmittance_step = 1 / n_bins
    tb_centr = np.zeros(n_bins - 1)
    tb_weight = np.zeros(n_bins - 1)
    
    def bin_for(transmission):
        t_flipped_and_scaled = 1 - (transmission - transmission_threshold) / (1 - transmission_threshold);
        t_bin = int(min(t_flipped_and_scaled, 1.0001) * (len(tb_centr) - 1))
        return t_bin;

    for index in ordered_indices:
        # last_bin_index = bin_for(transmittance)
        bin_index_1 = bin_for(1)
        bin_index_0 = bin_for(0)
        transmittance = transmittance * np.exp(-g_ints[index])
        current_bin_index = bin_for(transmittance)
        tb_centr[current_bin_index] += centroids[index] * transmittance
        tb_weight[current_bin_index] += transmittance
    
    borders = np.concatenate(([0.0], tb_centr / tb_weight, [(centroids + SDs * 3).reshape(-1).max()]))
    nan_indices = np.isnan(borders)
    valid_indices = ~nan_indices
    borders[nan_indices] = np.interp(np.flatnonzero(nan_indices), 
                                 np.flatnonzero(valid_indices), 
                                 borders[valid_indices])
    return borders.reshape(1, -1), np.zeros_like(borders)

def compute_bin_borders(n_bins: int) -> np.array:
    transmission_threshold = 0.05
    centr_p_k_sig = (centroids + SDs * 2).reshape(-1)
    g_ints = integrate_g_0_to(centr_p_k_sig.reshape(-1, 1)).reshape(-1)
    # g_ints = g(centroids.reshape(-1, 1)).reshape(-1)

    tb_centr = np.ones(n_bins) * np.infty
    tb_int = np.zeros(n_bins)
    tb_trans = np.zeros(n_bins)
    transmittance = 1
    last_bin_index = 0
    integration_range_end = 0

    for index in range(len(centroids)):
        if transmittance < transmission_threshold:
            break

        # set end to largest extent
        tb_centr[last_bin_index] = centroids[index]
        tb_int[last_bin_index] = g_ints[index]
        integration_range_end = max(integration_range_end, centr_p_k_sig[index])
        ordered_indices = np.argsort(tb_centr)
        tb_centr = tb_centr[ordered_indices]
        tb_int = tb_int[ordered_indices]
        last_bin_index = last_bin_index + 1

        def closer_neighbour(index: int) -> int:
            if index == 0:
                return 1
            if index == len(tb_centr) - 1:
                return index - 1
            if abs(tb_centr[index] - tb_centr[index - 1]) < abs(tb_centr[index] - tb_centr[index + 1]):
                return index - 1
            return index + 1

        if last_bin_index == n_bins:
            # recompute transmission weighted integrals
            transmittance = 1
            for j in range(n_bins):
                tb_trans[j] = transmittance
                transmittance = transmittance * np.exp(-tb_int[j])
            
            # # search for merge candidate
            # best_idx = -1
            # best_idx_val = np.infty
            # prev_pos = 0
            # prev_int = np.infty
            # prev_trans = 1.0
            # for j in range(n_bins):
            #     val = prev_trans * abs(tb_int[j] - prev_int) * (tb_centr[j] - prev_pos)
            #     if (val < best_idx_val):
            #         best_idx_val = val
            #         best_idx = j
            #     prev_pos = tb_centr[j]
            #     prev_int = tb_int[j]
            #     prev_trans = tb_trans[j]
            # # merge
            # assert(best_idx > 0)
            # tb_centr[best_idx - 1] = (tb_trans[best_idx - 1] * tb_int[best_idx - 1] * tb_centr[best_idx - 1] + tb_trans[best_idx] * tb_int[best_idx]  * tb_centr[best_idx]) / \
            #         (tb_trans[best_idx - 1] * tb_int[best_idx - 1]  + tb_trans[best_idx] * tb_int[best_idx] )
            # tb_int[best_idx - 1] = tb_int[best_idx - 1] + tb_int[best_idx]
            # for j in range(best_idx, n_bins - 1):
            #     tb_centr[j] = tb_centr[j + 1]
            #     tb_int[j] = tb_int[j + 1]
            # last_bin_index = n_bins - 1

            # search for merge candidate
            best_idx = -1
            best_idx_val = np.infty
            for j in range(n_bins - 1):
                val = tb_trans[j] * tb_int[j] * abs(tb_centr[closer_neighbour(j)] - tb_centr[j])
                if (val < best_idx_val):
                    best_idx_val = val
                    best_idx = j
            assert(best_idx >= 0)
            assert(best_idx < n_bins - 1)
                
            # merge
            merge_index = min(closer_neighbour(best_idx), best_idx)
            tb_centr[merge_index] = (tb_trans[merge_index] * tb_int[merge_index] * tb_centr[merge_index] + tb_trans[merge_index + 1] * tb_int[merge_index + 1]  * tb_centr[merge_index + 1]) / \
                    (tb_trans[merge_index] * tb_int[merge_index]  + tb_trans[merge_index + 1] * tb_int[merge_index + 1])
            tb_int[merge_index] = tb_int[merge_index] + tb_int[merge_index + 1]
            for j in range(merge_index + 1, n_bins - 1):
                tb_centr[j] = tb_centr[j + 1]
                tb_int[j] = tb_int[j + 1]
            last_bin_index = n_bins - 1
    
    borders = np.concatenate(([0.0], tb_centr[:-1], [integration_range_end]))
    inf_indices = np.isinf(borders)
    valid_indices = ~inf_indices
    borders[inf_indices] = np.interp(np.flatnonzero(inf_indices), 
                                 np.flatnonzero(valid_indices), 
                                 borders[valid_indices])
    return borders.reshape(1, -1), np.concatenate(([0.0], tb_int[:-1], [0]))

# def compute_bin_borders(n_bins: int) -> np.array:
    # return np.arange(0, 15, 2, dtype=np.float64).reshape(1, -1) # equal spacing
    # return np.array([0.0, 1.2, 4.6, 6.0, 12.0, 16.0]).reshape(1, -1) # valleys
    # return np.array([0.0, 0.5, 2.7, 5.2, 8.0, 16.0]).reshape(1, -1) # peaks
    # return np.array([0.0, 3.9, 7.9, 12.0, 16.0]).reshape(1, -1) # demo

    # bin_borders = np.array([0.0, 1.1, 4.6, 6.0, 10.1, 15.0]).reshape(1, -1)
    # bin_borders = np.array([0.0, 2.1, 4.6, 8.3, 13.1, 15.0]).reshape(1, -1)
    # bin_borders = np.random.uniform(0, 15, size=8)
    # bin_borders = np.concatenate(([0.0], bin_borders, [15.0]))
    # bin_borders = np.sort(bin_borders).reshape(1, -1)

bin_borders, border_masses = compute_bin_borders(5)

delta = 0.01
xes = np.arange(0, bin_borders[0, -1], delta, dtype=np.float64).reshape(1, -1)
gm_evals = gm(xes)
gm_int = np.cumsum(gm_evals) * delta

c_bin_borders = c(bin_borders)
gm_bin_borders = gm(bin_borders)

i_s = np.arange(0, bin_borders.shape[1]-1, dtype=np.int32)
c_id = c(bin_borders[:, i_s])
c_ik = (c(bin_borders[:, i_s+1]) - c(bin_borders[:, i_s])) / (bin_borders[0, i_s+1] - bin_borders[0, i_s])

def c_lin_approx(xes: np.array) -> np.array:
    i = np.sum(xes >= bin_borders.reshape(-1, 1), axis=0) - 1
    t_i = (xes - bin_borders[0, i]).reshape(-1)
    return c_id[i] + c_ik[i] * t_i


def compute_f_lin_approx_factors():
    masses = integrate_gm_0_to(bin_borders[:, i_s + 1]) - integrate_gm_0_to(bin_borders[:, i_s])
    f_borders = gm(bin_borders).reshape(-1)

    # masses left and right, because f_begin or f_end can be very small -> triangle to middle would
    # not carry much mass. but we can make one of the triangles larger.
    # we don't want to do that if we devide in 2, because that would take away sharpness (?)
    # percentage_left = border_masses[i_s] / (border_masses[i_s] + border_masses[i_s + 1])
    percentage_left = np.where(border_masses[i_s] * border_masses[i_s + 1] == 0, \
                               f_borders[i_s] / (f_borders[i_s] + f_borders[i_s + 1]), \
                               border_masses[i_s] / (border_masses[i_s] + border_masses[i_s + 1]))
    # masses_left = masses * (f_borders[i_s] / (f_borders[i_s] + f_borders[i_s + 1]))
    masses_left = masses * percentage_left
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
    d_fi[f_i] = f_borders[-1]
    return f_t, d_fi, k_fi

f_t, d_fi, k_fi = compute_f_lin_approx_factors()

def f_lin_approx(xes: np.array) -> np.array:
    i = np.sum(xes >= f_t.reshape(-1, 1), axis=0) - 1
    t_i = (xes.reshape(-1) - f_t[i])
    return d_fi[i] + k_fi[i] * t_i

def vol_int(f, c) -> np.array:
    f_evals = f(xes)
    c_evals = c(xes)
    f_int = np.cumsum(f_evals) * delta
    contrib_evals = f_evals * c_evals * np.exp(-1 * f_int)
    return np.cumsum(contrib_evals) * delta


plt.figure(figsize=(16,13))
# plt.plot(xes.reshape(-1), g(xes).transpose(), label='g', zorder=1) # color="tab:blue", 
plt.plot(xes.reshape(-1), gm(xes), label='gm', zorder=1) # color="tab:blue", 
# plt.plot(xes.reshape(-1), gm_int, label='gm integrated') # , color="tab:green"
# plt.plot(xes.reshape(-1), integrate_gm_0_to(xes), label='integrate_gm_0_to')
# plt.plot(xes.reshape(-1), np.exp(-1 * gm_int), label='transmittance')
# plt.plot(xes.reshape(-1), gm_evals * np.exp(-1 * gm_int), label='transmittance * gm')
# plt.plot(xes.reshape(-1), c(xes), label='c')
# plt.plot(xes.reshape(-1), vol_val(xes), label='vol_val gm')
plt.plot(xes.reshape(-1), vol_int(gm, c), label='vol_int gm')
plt.plot(xes.reshape(-1), vol_int(f_lin_approx, c_lin_approx), label='vol_int approx')

# plt.plot(xes.reshape(-1), np.cumsum(vol_val(xes)) * delta, label='vol_int')
# plt.scatter(bin_borders, c_bin_borders, label='c_bin_borders')
# plt.plot(xes.reshape(-1), c_lin_approx(xes), label='c_interpol')
# plt.plot(np.array([0-1, 17]), np.array([0, 0]), color="lightgray", zorder=0)
plt.plot(xes.reshape(-1), f_lin_approx(xes), label='f', color="tab:orange", zorder=1)
# plt.plot(xes.reshape(-1), f_lin_approx(xes) * np.exp(-1 * gm_int), label='f * transmittance', color="tab:orange", zorder=1)
# plt.scatter(f_t, d_fi, label='f endpoints', color="tab:orange", zorder=2)
plt.vlines(bin_borders, np.zeros_like(bin_borders)-0.5, np.ones_like(bin_borders) * 1.3, color="thistle", zorder=0, label='bin borders')
plt.legend()
ax = plt.gca()
ax.set_ylim([-0.05, 1.3])
ax.set_xlim([-0.5, 15.5])
plt.show()
print("")