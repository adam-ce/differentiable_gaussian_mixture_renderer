/*****************************************************************************
 * DGMR
 * Copyright (C) 2024 Adam Celarek
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *****************************************************************************/

#pragma once

#include <glm/glm.hpp>

#include <stroke/linalg.h>

namespace dgmr::piecewise_linear {

template <typename scalar_t>
struct FunctionGroup {
    using Vec = glm::vec<4, scalar_t>;
    Vec d0;
    Vec k0;
    Vec t1;
    Vec t2;
    Vec k2;
    scalar_t t_right;

    STROKE_DEVICES_INLINE glm::vec<4, scalar_t> sample(scalar_t t) const
    {
        const auto s = [t, this](int i) {
            if (t <= t1[i])
                return d0[i] + k0[i] * t;
            if (t <= t2[i])
                return scalar_t(0);
            auto tp = t - t2[i];
            return d0[i] + t1[i] * k0[i] + tp * k2[i];
        };
        return { s(0), s(1), s(2), s(3) };
    }

    STROKE_DEVICES_INLINE void check() const
    {
        assert(all(greaterThanEqual(d0, Vec(0, 0, 0, 0))));
        assert(all(greaterThanEqual(t1, Vec(0, 0, 0, 0))));
        assert(all(greaterThanEqual(t2, t1)));
        assert(all(greaterThanEqual(d0 + t1 * k0, Vec(0, 0, 0, 0))));
    }
};

template <typename scalar_t>
STROKE_DEVICES_INLINE FunctionGroup<scalar_t> create_approximation(
    const glm::vec<4, scalar_t>& masses_percent_left,
    const glm::vec<4, scalar_t>& masses_total,
    const glm::vec<4, scalar_t>& eval_left,
    const glm::vec<4, scalar_t>& eval_right,
    const scalar_t t_right)
{
    assert(t_right > 0);
    assert(!stroke::isnan(t_right));
    assert(!stroke::isinf(t_right));
    const auto m_l = masses_percent_left * masses_total;
    const auto m_r = masses_total - m_l;
    auto f = FunctionGroup<scalar_t> { eval_left, {}, scalar_t(2) * m_l / eval_left, t_right - scalar_t(2) * m_r / eval_right, {}, t_right };
    const auto compute = [&](int i) {
        if (masses_total[i] <= 0.000001) { // happens for colours
            f.d0[i] = 0;
            f.k0[i] = 0;
            f.t1[i] = 0;
            f.t2[i] = t_right;
            f.k2[i] = 0;
            return;
        }

        if (f.t2[i] > f.t1[i]) {
            f.k0[i] = -eval_left[i] * eval_left[i] / (2 * m_l[i]);
            f.k2[i] = eval_right[i] * eval_right[i] / (2 * m_r[i]);
            return;
        }

        const auto m = masses_total[i];
        const auto gm_l = eval_left[i];
        const auto gm_r = eval_right[i];
        const auto t_omega = t_right; // t_left is always zero in this implementation
        const auto c1 = 2 * m - (gm_l + gm_r) * t_omega;
        const auto c2 = gm_r * m_l[i] + gm_l * m_r[i];
        const auto c3 = gm_l * gm_r * t_omega;
        const auto dm = stroke::max(scalar_t(0), (c1 + stroke::sqrt(c1 * c1 + 4 * t_omega * (2 * c2 - c3))) / (2 * t_omega)); // numerical instability
#ifndef NDEBUG
        if (dm < 0 || stroke::isnan(dm)) {
            printf("dm = %f\n", dm);
            // printf("masses_percent_left[%i] = %f\n", i, masses_percent_left[i]);
            printf("masses_total[%i] = %f\n", i, masses_total[i]);
            printf("eval_left[%i] = %f\n", i, eval_left[i]);
            printf("eval_right[%i] = %f\n", i, eval_right[i]);
            printf("t_right = %f\n\n", t_right);
        }
#endif
        assert(dm >= 0);

        const auto compute_tl_tr = [&]() -> cuda::std::pair<float, float> {
            if (m_r[i] > m_l[i]) {
                const auto tr = 2 * m_r[i] / (dm + gm_r);
                const auto tl = stroke::max(scalar_t(0.), t_omega - tr);
                return { tl, tr };
            }
            const auto tl = 2 * m_l[i] / (dm + gm_l);
            const auto tr = stroke::max(scalar_t(0.), t_omega - tl);
            return { tl, tr };
        };
        const auto [tl, tr] = compute_tl_tr();
        // const auto tr = 2 * m_r[i] / (dm + gm_r);
        // assert(tr >= 0);
        // const auto tl_p = t_omega - tr;
        // const auto tl = 2 * m_l[i] / (dm + gm_l); // more stable than tl_p for ml[i] == 0
#ifndef NDEBUG
        if (!(tl >= 0)) {
            printf("tl = %f\n", tl);
            printf("masses_percent_left[%i] = %f\n", i, masses_percent_left[i]);
            printf("masses_total[%i] = %f\n", i, masses_total[i]);
            printf("eval_left[%i] = %f\n", i, eval_left[i]);
            printf("eval_right[%i] = %f\n", i, eval_right[i]);
            printf("t_right = %f\n\n", t_right);
        }
#endif
        assert(tl >= 0);
        // if (stroke::abs(tl - tl_p) >= 0.001) {
        //     printf("tl = %f\n", tl);
        //     printf("tl_p = %f\n", tl_p);
        //     printf("t_omega = %f\n", t_omega);
        //     printf("dm = %f\n", dm);
        //     printf("masses_total[%i] = %f\n", i, masses_total[i]);
        // }

        // assert(stroke::abs(tl - tl_p) < 0.001);

        f.k0[i] = -(gm_l - dm) / tl;
        if (tl == 0)
            f.k0[i] = 0;
        f.t1[i] = tl;
        f.t2[i] = tl;
        f.k2[i] = (gm_r - dm) / tr;
        if (tr == 0)
            f.k2[i] = 0;
    };
    for (unsigned i = 0; i < 4; ++i)
        compute(i);
    return f;
}

template <typename scalar_t>
STROKE_DEVICES_INLINE glm::vec<3, scalar_t>
volume_integrate(const whack::Array<FunctionGroup<scalar_t>, 8>& approximations, unsigned n_steps_per_bin)
{
    glm::vec<3, scalar_t> result = {};
    scalar_t current_m = 0;
    scalar_t current_transparency = 1;
    for (const FunctionGroup<scalar_t>& f : approximations) {
        const auto delta_t = f.t_right / n_steps_per_bin;
        scalar_t t = delta_t / 2;
        for (unsigned i = 0; i < n_steps_per_bin; ++i) {
            assert(t < f.t_right);
            const auto eval_t = f.sample(t);
            // result += glm::dvec3(eval_t) * stroke::exp(-current_m) * delta_t;
            result += glm::vec<3, scalar_t>(eval_t) * current_transparency * delta_t;
            current_m += eval_t.w * delta_t;
            current_transparency *= stroke::max(scalar_t(0), 1 - eval_t.w * delta_t);
            t += delta_t;
        }
    }
    return result;
}

} // namespace dgmr::piecewise_linear
