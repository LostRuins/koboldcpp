#ifndef __SD_RUNTIME_PREVIEW_INTERVAL_H__
#define __SD_RUNTIME_PREVIEW_INTERVAL_H__

#include <cstddef>
#include <cstdint>
#include <limits>

namespace sd::preview {

    constexpr std::uint64_t logical_sample_step(int step) {
        return step < 0 ? static_cast<std::uint64_t>(-static_cast<std::int64_t>(step))
                        : static_cast<std::uint64_t>(step);
    }

    constexpr bool sample_step_is_complete(int step,
                                           std::size_t total_steps,
                                           bool terminal_sigma_is_zero) {
        return step > 0 ||
               (terminal_sigma_is_zero &&
                step < 0 &&
                logical_sample_step(step) == static_cast<std::uint64_t>(total_steps));
    }

    constexpr bool should_preview_sample_step(int step,
                                              std::size_t total_steps,
                                              bool terminal_sigma_is_zero,
                                              int interval,
                                              bool preview_final_step) {
        if (interval > 0) {
            return step % interval == 0;
        }
        if (!sample_step_is_complete(step, total_steps, terminal_sigma_is_zero)) {
            return false;
        }

        std::uint64_t logical_step = logical_sample_step(step);
        if (interval < 0) {
            std::uint64_t requested_step = static_cast<std::uint64_t>(-static_cast<std::int64_t>(interval));
            return logical_step == requested_step;
        }
        return preview_final_step && logical_step == static_cast<std::uint64_t>(total_steps);
    }
}  // namespace sd::preview

#endif  // __SD_RUNTIME_PREVIEW_INTERVAL_H__
