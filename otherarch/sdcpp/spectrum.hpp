#ifndef __SPECTRUM_HPP__
#define __SPECTRUM_HPP__

#include <cmath>
#include <cstring>
#include <vector>

#include "ggml_extend.hpp"

struct SpectrumConfig {
    float w            = 0.40f;
    int m              = 3;
    float lam          = 1.0f;
    int window_size    = 2;
    float flex_window  = 0.50f;
    int warmup_steps   = 4;
    float stop_percent = 0.9f;
};

struct SpectrumState {
    SpectrumConfig config;
    int cnt                 = 0;
    int num_cached          = 0;
    float curr_ws           = 2.0f;
    int K                   = 6;
    int stop_step           = 0;
    int total_steps_skipped = 0;

    std::vector<std::vector<float>> H_buf;
    std::vector<float> T_buf;

    void init(const SpectrumConfig& cfg, size_t total_steps) {
        config              = cfg;
        cnt                 = 0;
        num_cached          = 0;
        curr_ws             = (float)cfg.window_size;
        K                   = std::max(cfg.m + 1, 6);
        stop_step           = (int)(cfg.stop_percent * (float)total_steps);
        total_steps_skipped = 0;
        H_buf.clear();
        T_buf.clear();
    }

    float taus(int step_cnt) const {
        return (step_cnt / 50.0f) * 2.0f - 1.0f;
    }

    bool should_predict() {
        if (cnt < config.warmup_steps)
            return false;
        if (stop_step > 0 && cnt >= stop_step)
            return false;
        if ((int)H_buf.size() < 2)
            return false;

        int ws = std::max(1, (int)std::floor(curr_ws));
        return (num_cached + 1) % ws != 0;
    }

    void update(const struct ggml_tensor* denoised) {
        int64_t ne        = ggml_nelements(denoised);
        const float* data = (const float*)denoised->data;

        H_buf.emplace_back(data, data + ne);
        T_buf.push_back(taus(cnt));

        while ((int)H_buf.size() > K) {
            H_buf.erase(H_buf.begin());
            T_buf.erase(T_buf.begin());
        }

        if (cnt >= config.warmup_steps)
            curr_ws += config.flex_window;

        num_cached = 0;
        cnt++;
    }

    void predict(struct ggml_tensor* denoised) {
        int64_t F    = (int64_t)H_buf[0].size();
        int K_curr   = (int)H_buf.size();
        int M1       = config.m + 1;
        float tau_at = taus(cnt);

        // Design matrix X: K_curr x M1 (Chebyshev basis)
        std::vector<float> X(K_curr * M1);
        for (int i = 0; i < K_curr; i++) {
            X[i * M1] = 1.0f;
            if (M1 > 1)
                X[i * M1 + 1] = T_buf[i];
            for (int j = 2; j < M1; j++)
                X[i * M1 + j] = 2.0f * T_buf[i] * X[i * M1 + j - 1] - X[i * M1 + j - 2];
        }

        // x_star: Chebyshev basis at current tau
        std::vector<float> x_star(M1);
        x_star[0] = 1.0f;
        if (M1 > 1)
            x_star[1] = tau_at;
        for (int j = 2; j < M1; j++)
            x_star[j] = 2.0f * tau_at * x_star[j - 1] - x_star[j - 2];

        // XtX = X^T X + lambda I
        std::vector<float> XtX(M1 * M1, 0.0f);
        for (int i = 0; i < M1; i++) {
            for (int j = 0; j < M1; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K_curr; k++)
                    sum += X[k * M1 + i] * X[k * M1 + j];
                XtX[i * M1 + j] = sum + (i == j ? config.lam : 0.0f);
            }
        }

        // Cholesky decomposition
        std::vector<float> L(M1 * M1, 0.0f);
        if (!cholesky_decompose(XtX.data(), L.data(), M1)) {
            float trace = 0.0f;
            for (int i = 0; i < M1; i++)
                trace += XtX[i * M1 + i];
            for (int i = 0; i < M1; i++)
                XtX[i * M1 + i] += 1e-4f * trace / M1;
            cholesky_decompose(XtX.data(), L.data(), M1);
        }

        // Solve XtX v = x_star
        std::vector<float> v(M1);
        cholesky_solve(L.data(), x_star.data(), v.data(), M1);

        // Prediction weights per history entry
        std::vector<float> weights(K_curr, 0.0f);
        for (int k = 0; k < K_curr; k++)
            for (int j = 0; j < M1; j++)
                weights[k] += X[k * M1 + j] * v[j];

        // Blend Chebyshev and Taylor predictions
        float* out          = (float*)denoised->data;
        float w_cheb        = config.w;
        float w_taylor      = 1.0f - w_cheb;
        const float* h_last = H_buf.back().data();
        const float* h_prev = H_buf[H_buf.size() - 2].data();

        for (int64_t f = 0; f < F; f++) {
            float pred_cheb = 0.0f;
            for (int k = 0; k < K_curr; k++)
                pred_cheb += weights[k] * H_buf[k][f];

            float pred_taylor = h_last[f] + 0.5f * (h_last[f] - h_prev[f]);

            out[f] = w_taylor * pred_taylor + w_cheb * pred_cheb;
        }

        num_cached++;
        total_steps_skipped++;
        cnt++;
    }

private:
    static bool cholesky_decompose(const float* A, float* L, int n) {
        std::memset(L, 0, n * n * sizeof(float));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                float sum = 0.0f;
                for (int k = 0; k < j; k++)
                    sum += L[i * n + k] * L[j * n + k];
                if (i == j) {
                    float diag = A[i * n + i] - sum;
                    if (diag <= 0.0f)
                        return false;
                    L[i * n + j] = std::sqrt(diag);
                } else {
                    L[i * n + j] = (A[i * n + j] - sum) / L[j * n + j];
                }
            }
        }
        return true;
    }

    static void cholesky_solve(const float* L, const float* b, float* x, int n) {
        std::vector<float> y(n);
        for (int i = 0; i < n; i++) {
            float sum = 0.0f;
            for (int j = 0; j < i; j++)
                sum += L[i * n + j] * y[j];
            y[i] = (b[i] - sum) / L[i * n + i];
        }
        for (int i = n - 1; i >= 0; i--) {
            float sum = 0.0f;
            for (int j = i + 1; j < n; j++)
                sum += L[j * n + i] * x[j];
            x[i] = (y[i] - sum) / L[i * n + i];
        }
    }
};

#endif  // __SPECTRUM_HPP__
