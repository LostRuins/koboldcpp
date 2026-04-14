#ifndef __CONDITION_CACHE_UTILS_HPP__
#define __CONDITION_CACHE_UTILS_HPP__

#include <vector>

#include "tensor.hpp"

namespace sd {

    inline bool store_condition_cache_diff(std::vector<float>* diff,
                                           const sd::Tensor<float>& input,
                                           const sd::Tensor<float>& output) {
        if (diff == nullptr || input.empty() || output.empty()) {
            return false;
        }

        size_t input_size  = static_cast<size_t>(input.numel());
        size_t output_size = static_cast<size_t>(output.numel());
        if (input_size == 0 || input_size != output_size) {
            diff->clear();
            return false;
        }

        const float* input_data  = input.data();
        const float* output_data = output.data();
        if (input_data == nullptr || output_data == nullptr) {
            diff->clear();
            return false;
        }

        diff->resize(output_size);
        for (size_t i = 0; i < output_size; ++i) {
            (*diff)[i] = output_data[i] - input_data[i];
        }
        return true;
    }

    inline bool apply_condition_cache_diff(const std::vector<float>& diff,
                                           const sd::Tensor<float>& input,
                                           sd::Tensor<float>* output) {
        if (output == nullptr || input.empty() || diff.empty()) {
            return false;
        }

        size_t input_size = static_cast<size_t>(input.numel());
        if (input_size == 0 || diff.size() != input_size) {
            return false;
        }

        *output            = input;
        float* output_data = output->data();
        if (output_data == nullptr) {
            return false;
        }

        for (size_t i = 0; i < input_size; ++i) {
            output_data[i] += diff[i];
        }
        return true;
    }

}  // namespace sd

#endif  // __CONDITION_CACHE_UTILS_HPP__
