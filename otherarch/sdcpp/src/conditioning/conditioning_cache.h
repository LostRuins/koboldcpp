#ifndef __SD_CONDITIONING_CONDITIONING_CACHE_H__
#define __SD_CONDITIONING_CONDITIONING_CACHE_H__

#include <algorithm>
#include <list>
#include <tuple>

#include "conditioning/conditioner.hpp"

class ConditioningCache {
    struct Entry {
        ConditionerParams params;
        std::vector<sd::Tensor<float>> ref_images;
        std::vector<MiniMaxH3PresentationItem> references;
        SDCondition condition;

        Entry(const ConditionerParams& input, const SDCondition& output)
            : params(input), condition(output) {
            // Request-owned reference pointers must not outlive the request.
            if (input.ref_images != nullptr) {
                ref_images        = *input.ref_images;
                params.ref_images = &ref_images;
            }
            if (input.minimax_h3_references != nullptr) {
                references                   = *input.minimax_h3_references;
                params.minimax_h3_references = &references;
            }
        }

        Entry(const Entry&)            = delete;
        Entry& operator=(const Entry&) = delete;
    };

    size_t capacity_ = 4;
    std::list<Entry> entries_;

    static bool same_images(const std::vector<sd::Tensor<float>>& a,
                            const std::vector<sd::Tensor<float>>& b) {
        return std::equal(a.begin(), a.end(), b.begin(), b.end(),
                          [](const sd::Tensor<float>& x, const sd::Tensor<float>& y) {
                              return x.shape() == y.shape() && x.values() == y.values();
                          });
    }

    static bool same_params(const ConditionerParams& a, const ConditionerParams& b) {
        const auto fields = [](const ConditionerParams& p) {
            const auto& r = p.ref_image_params;
            return std::tie(p.text, p.clip_skip, p.width, p.height, p.zero_out_masked,
                            r.pass_to_vlm, r.pass_to_dit, r.ref_index_mode,
                            r.force_ref_timestep_zero, r.resize_before_vae, r.vae_input_max_pixels,
                            r.vlm_resize_mode, r.vlm_min_size, r.vlm_max_size, r.resize_vae_to_target);
        };
        if (fields(a) != fields(b) ||
            (a.ref_images == nullptr) != (b.ref_images == nullptr) ||
            (a.minimax_h3_references == nullptr) != (b.minimax_h3_references == nullptr)) {
            return false;
        }
        if (a.ref_images != nullptr && !same_images(*a.ref_images, *b.ref_images)) {
            return false;
        }
        if (a.minimax_h3_references != nullptr &&
            !std::equal(a.minimax_h3_references->begin(), a.minimax_h3_references->end(),
                        b.minimax_h3_references->begin(), b.minimax_h3_references->end(),
                        [](const MiniMaxH3PresentationItem& x, const MiniMaxH3PresentationItem& y) {
                            return x.kind == y.kind && x.timestamps == y.timestamps && same_images(x.frames, y.frames);
                        })) {
            return false;
        }
        return true;
    }

public:
    void set_capacity(size_t capacity) {
        capacity_ = capacity;
        while (entries_.size() > capacity_) {
            entries_.pop_back();
        }
    }

    void clear() {
        entries_.clear();
    }

    SDCondition get(Conditioner& conditioner, int n_threads, const ConditionerParams& params) {
        if (capacity_ == 0) {
            return conditioner.get_learned_condition(n_threads, params);
        }
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            if (same_params(it->params, params)) {
                entries_.splice(entries_.begin(), entries_, it);
                LOG_INFO("conditioning cache hit");
                return entries_.front().condition;
            }
        }
        auto condition = conditioner.get_learned_condition(n_threads, params);
        if (!condition.empty()) {
            if (entries_.size() == capacity_) {
                entries_.pop_back();
            }
            entries_.emplace_front(params, condition);
            LOG_VERBOSE("conditioning cache stored (%zu/%zu)", entries_.size(), capacity_);
        }
        return condition;
    }
};

#endif  // __SD_CONDITIONING_CONDITIONING_CACHE_H__
