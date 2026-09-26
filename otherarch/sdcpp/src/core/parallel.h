#ifndef __SD_CORE_PARALLEL_H__
#define __SD_CORE_PARALLEL_H__

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <utility>

namespace sd {

    class ParallelExecutor {
        struct Impl;
        int n_threads_;
        std::unique_ptr<Impl> impl_;

    public:
        explicit ParallelExecutor(int n_threads);
        ~ParallelExecutor();
        ParallelExecutor(const ParallelExecutor&)            = delete;
        ParallelExecutor& operator=(const ParallelExecutor&) = delete;

        int num_threads() const { return n_threads_; }
        void run(int64_t begin, int64_t end, int64_t grain_size, const std::function<void(int64_t, int64_t)>& task);
    };

    namespace parallel_detail {
        extern thread_local ParallelExecutor* executor;
        extern thread_local bool active;

        class Region {
            bool previous_;

        public:
            Region()
                : previous_(active) { active = true; }
            ~Region() { active = previous_; }
            Region(const Region&)            = delete;
            Region& operator=(const Region&) = delete;
        };
    }

    class ParallelScope {
        ParallelExecutor* previous_;

    public:
        explicit ParallelScope(ParallelExecutor* executor)
            : previous_(parallel_detail::executor) {
            parallel_detail::executor = executor;
        }
        ~ParallelScope() { parallel_detail::executor = previous_; }
        ParallelScope(const ParallelScope&)            = delete;
        ParallelScope& operator=(const ParallelScope&) = delete;
    };

    // Ranges are non-negative. The callback may run concurrently and must own its writes.
    template <typename F>
    inline void parallel_for(int64_t begin, int64_t end, int64_t grain_size, F&& task) {
        if (begin < 0 || grain_size <= 0) {
            throw std::invalid_argument("parallel_for requires begin >= 0 and grain_size > 0");
        }
        if (end <= begin) {
            return;
        }
        auto* executor = parallel_detail::executor;
        if (parallel_detail::active || executor == nullptr || executor->num_threads() <= 1 ||
            (end - begin) / grain_size < 2) {
            parallel_detail::Region region;
            task(begin, end);
            return;
        }
        executor->run(begin, end, grain_size, std::forward<F>(task));
    }

}

#endif  // __SD_CORE_PARALLEL_H__
