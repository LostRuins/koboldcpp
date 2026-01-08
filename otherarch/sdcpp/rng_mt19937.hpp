#ifndef __RNG_MT19937_HPP__
#define __RNG_MT19937_HPP__

#include <cmath>
#include <vector>

#include "rng.hpp"

// RNG imitiating torch cpu randn on CPU.
// Port from pytorch, original license: https://github.com/pytorch/pytorch/blob/d01a7b0241ed1c4cded7e7ca097249feb343f072/LICENSE
// Ref: https://github.com/pytorch/pytorch/blob/d01a7b0241ed1c4cded7e7ca097249feb343f072/aten/src/ATen/core/TransformationHelper.h, for uniform_real
// Ref: https://github.com/pytorch/pytorch/blob/d01a7b0241ed1c4cded7e7ca097249feb343f072/aten/src/ATen/native/cpu/DistributionTemplates.h, for normal_kernel/normal_fill/normal_fill_16
// Ref: https://github.com/pytorch/pytorch/blob/d01a7b0241ed1c4cded7e7ca097249feb343f072/aten/src/ATen/core/MT19937RNGEngine.h, for mt19937_engine
// Ref: https://github.com/pytorch/pytorch/blob/d01a7b0241ed1c4cded7e7ca097249feb343f072/aten/src/ATen/core/DistributionsHelper.h, for uniform_real_distribution/normal_distribution
class MT19937RNG : public RNG {
    static const int N             = 624;
    static const int M             = 397;
    static const uint32_t MATRIX_A = 0x9908b0dfU;
    static const uint32_t UMASK    = 0x80000000U;
    static const uint32_t LMASK    = 0x7fffffffU;

    struct State {
        uint64_t seed_;
        int left_;
        bool seeded_;
        uint32_t next_;
        std::array<uint32_t, N> state_;
        bool has_next_gauss = false;
        double next_gauss   = 0.0f;
    };

    State s;

    uint32_t mix_bits(uint32_t u, uint32_t v) { return (u & UMASK) | (v & LMASK); }
    uint32_t twist(uint32_t u, uint32_t v) { return (mix_bits(u, v) >> 1) ^ ((v & 1) ? MATRIX_A : 0); }
    void next_state() {
        uint32_t* p = s.state_.data();
        s.left_     = N;
        s.next_     = 0;
        for (int j = N - M + 1; --j; p++)
            p[0] = p[M] ^ twist(p[0], p[1]);
        for (int j = M; --j; p++)
            p[0] = p[M - N] ^ twist(p[0], p[1]);
        p[0] = p[M - N] ^ twist(p[0], s.state_[0]);
    }

    uint32_t rand_uint32() {
        if (--s.left_ == 0)
            next_state();
        uint32_t y = s.state_[s.next_++];
        y ^= (y >> 11);
        y ^= (y << 7) & 0x9d2c5680U;
        y ^= (y << 15) & 0xefc60000U;
        y ^= (y >> 18);
        return y;
    }

    uint64_t rand_uint64() {
        uint64_t high = (uint64_t)rand_uint32();
        uint64_t low  = (uint64_t)rand_uint32();
        return (high << 32) | low;
    }

    template <typename T, typename V>
    T uniform_real(V val, T from, T to) {
        constexpr auto MASK    = static_cast<V>((static_cast<uint64_t>(1) << std::numeric_limits<T>::digits) - 1);
        constexpr auto DIVISOR = static_cast<T>(1) / (static_cast<uint64_t>(1) << std::numeric_limits<T>::digits);
        T x                    = (val & MASK) * DIVISOR;
        return (x * (to - from) + from);
    }

    double normal_double_value(double mean, double std) {
        if (s.has_next_gauss) {
            s.has_next_gauss = false;
            return s.next_gauss;
        }
        double u1 = uniform_real(rand_uint64(), 0., 1.);  // double
        double u2 = uniform_real(rand_uint64(), 0., 1.);  // double

        double r         = std::sqrt(-2.0 * std::log1p(-u2));
        double theta     = 2.0 * 3.14159265358979323846 * u1;
        double value     = r * std::cos(theta) * std + mean;
        s.next_gauss     = r * std::sin(theta) * std + mean;
        s.has_next_gauss = true;
        return value;
    }

    void normal_fill_16(float* data, float mean, float std) {
        for (int j = 0; j < 8; ++j) {
            float u1    = 1.0f - data[j];
            float u2    = data[j + 8];
            float r     = std::sqrt(-2.0f * std::log(u1));
            float theta = 2.0f * 3.14159265358979323846f * u2;
            data[j]     = r * std::cos(theta) * std + mean;
            data[j + 8] = r * std::sin(theta) * std + mean;
        }
    }

    void randn(float* data, int64_t size, float mean = 0.0f, float std = 1.0f) {
        if (size >= 16) {
            for (int64_t i = 0; i < size; i++) {
                data[i] = uniform_real(rand_uint32(), 0.f, 1.f);
            }
            for (int64_t i = 0; i < size - 15; i += 16) {
                normal_fill_16(data + i, mean, std);
            }
            if (size % 16 != 0) {
                // Recompute the last 16 values.
                data = data + size - 16;
                for (int64_t i = 0; i < 16; i++) {
                    data[i] = uniform_real(rand_uint32(), 0.f, 1.f);
                }
                normal_fill_16(data, mean, std);
            }
        } else {
            // Strange handling, hard to understand, but keeping it consistent with PyTorch.
            for (int64_t i = 0; i < size; i++) {
                data[i] = (float)normal_double_value(mean, std);
            }
        }
    }

public:
    MT19937RNG(uint64_t seed = 0) { manual_seed(seed); }

    void manual_seed(uint64_t seed) override {
        s.seed_     = seed;
        s.seeded_   = true;
        s.state_[0] = (uint32_t)(seed & 0xffffffffU);
        for (int j = 1; j < N; j++) {
            uint32_t prev = s.state_[j - 1];
            s.state_[j]   = 1812433253U * (prev ^ (prev >> 30)) + j;
        }
        s.left_          = 1;
        s.next_          = 0;
        s.has_next_gauss = false;
    }

    std::vector<float> randn(uint32_t n) override {
        std::vector<float> out;
        out.resize(n);
        randn((float*)out.data(), out.size());
        return out;
    }
};

#endif  // __RNG_MT19937_HPP__