#ifndef __SD_CORE_RNG_PHILOX_HPP__
#define __SD_CORE_RNG_PHILOX_HPP__

#include <array>
#include <cmath>
#include <vector>

#include "core/rng.hpp"

// RNG imitiating torch cuda randn on CPU.
// Port from: https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/5ef669de080814067961f28357256e8fe27544f4/modules/rng_philox.py
class PhiloxRNG : public RNG {
private:
    uint64_t seed;
    uint32_t offset;

private:
    using Counter = std::array<std::vector<uint32_t>, 4>;

    static constexpr uint32_t philox_m[2] = {0xD2511F53, 0xCD9E8D57};
    static constexpr uint32_t philox_w[2] = {0x9E3779B9, 0xBB67AE85};
    float two_pow32_inv                   = 2.3283064e-10f;
    float two_pow32_inv_2pi               = 2.3283064e-10f * 6.2831855f;

    //  A single round of the Philox 4x32 random number generator.
    void philox4_round(Counter& counter, uint32_t key0, uint32_t key1) {
        uint32_t N = (uint32_t)counter[0].size();
        for (uint32_t i = 0; i < N; i++) {
            const uint64_t v1 = static_cast<uint64_t>(counter[0][i]) * static_cast<uint64_t>(philox_m[0]);
            const uint64_t v2 = static_cast<uint64_t>(counter[2][i]) * static_cast<uint64_t>(philox_m[1]);

            counter[0][i] = static_cast<uint32_t>(v2 >> 32) ^ counter[1][i] ^ key0;
            counter[1][i] = static_cast<uint32_t>(v2);
            counter[2][i] = static_cast<uint32_t>(v1 >> 32) ^ counter[3][i] ^ key1;
            counter[3][i] = static_cast<uint32_t>(v1);
        }
    }

    void philox4_32(Counter& counter, uint32_t key0, uint32_t key1, int rounds = 10) {
        for (int i = 0; i < rounds - 1; ++i) {
            philox4_round(counter, key0, key1);
            key0 += philox_w[0];
            key1 += philox_w[1];
        }

        philox4_round(counter, key0, key1);
    }

    float box_muller(float x, float y) {
        float u = x * two_pow32_inv + two_pow32_inv / 2;
        float v = y * two_pow32_inv_2pi + two_pow32_inv_2pi / 2;

        float s = sqrt(-2.0f * log(u));

        float r1 = s * sin(v);
        return r1;
    }

public:
    PhiloxRNG(uint64_t seed = 0) {
        this->seed   = seed;
        this->offset = 0;
    }

    std::shared_ptr<RNG> clone() const override {
        return std::make_shared<PhiloxRNG>(*this);
    }

    void manual_seed(uint64_t seed) override {
        this->seed   = seed;
        this->offset = 0;
    }

    std::vector<float> randn(uint32_t n) override {
        Counter counter;
        counter[0].resize(n, this->offset);
        counter[1].resize(n);
        counter[2].resize(n);
        counter[3].resize(n);

        for (uint32_t i = 0; i < n; i++) {
            counter[2][i] = i;
        }
        this->offset += 1;

        philox4_32(counter, static_cast<uint32_t>(this->seed), static_cast<uint32_t>(this->seed >> 32));

        std::vector<float> result(n);
        for (uint32_t i = 0; i < n; ++i) {
            result[i] = box_muller((float)counter[0][i], (float)counter[1][i]);
        }
        return result;
    }
};

#endif  // __SD_CORE_RNG_PHILOX_HPP__
