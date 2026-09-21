#include "core/parallel.h"

#include <algorithm>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace sd {

    namespace parallel_detail {
        thread_local ParallelExecutor* executor = nullptr;
        thread_local bool active                = false;
    }

    struct ParallelExecutor::Impl {
        struct Worker {
            std::condition_variable wake;
            std::thread thread;
            bool ready = false;
        };

        ParallelExecutor* owner;
        std::mutex invocation_mutex;
        std::mutex mutex;
        std::condition_variable finished;
        std::vector<std::unique_ptr<Worker>> workers;
        bool stopping                                     = false;
        int pending                                       = 0;
        int participants                                  = 1;
        int64_t begin                                     = 0;
        int64_t count                                     = 0;
        const std::function<void(int64_t, int64_t)>* task = nullptr;
        std::exception_ptr error;

        explicit Impl(ParallelExecutor* owner)
            : owner(owner) {}

        ~Impl() {
            {
                std::lock_guard<std::mutex> lock(mutex);
                stopping = true;
            }
            for (auto& worker : workers) {
                worker->wake.notify_one();
            }
            for (auto& worker : workers) {
                worker->thread.join();
            }
        }

        void execute(int index) {
            ParallelScope scope(owner);
            parallel_detail::Region region;
            const int64_t size  = count / participants;
            const int64_t extra = count % participants;
            const int64_t first = begin + index * size + std::min<int64_t>(index, extra);
            const int64_t last  = first + size + (index < extra ? 1 : 0);
            try {
                (*task)(first, last);
            } catch (...) {
                std::lock_guard<std::mutex> lock(mutex);
                if (!error) {
                    error = std::current_exception();
                }
            }
        }

        void worker_loop(Worker* worker, int index) {
            std::unique_lock<std::mutex> lock(mutex);
            for (;;) {
                worker->wake.wait(lock, [&] { return stopping || worker->ready; });
                if (stopping) {
                    return;
                }
                worker->ready = false;
                lock.unlock();
                execute(index);
                lock.lock();
                if (--pending == 0) {
                    finished.notify_one();
                }
            }
        }

        void run(int64_t first, int64_t last, int n_tasks, const std::function<void(int64_t, int64_t)>& callback) {
            std::lock_guard<std::mutex> invocation_lock(invocation_mutex);
            std::unique_lock<std::mutex> lock(mutex);
            while (static_cast<int>(workers.size()) < n_tasks - 1) {
                workers.push_back(std::make_unique<Worker>());
                auto* worker    = workers.back().get();
                const int index = static_cast<int>(workers.size());
                try {
                    worker->thread = std::thread([this, worker, index] { worker_loop(worker, index); });
                } catch (...) {
                    workers.pop_back();
                    throw;
                }
            }
            begin        = first;
            count        = last - first;
            participants = n_tasks;
            pending      = n_tasks - 1;
            task         = &callback;
            error        = nullptr;
            for (int i = 0; i < pending; ++i) {
                workers[i]->ready = true;
                workers[i]->wake.notify_one();
            }
            lock.unlock();
            execute(0);
            lock.lock();
            finished.wait(lock, [&] { return pending == 0; });
            task = nullptr;
            if (error) {
                std::rethrow_exception(error);
            }
        }
    };

    ParallelExecutor::ParallelExecutor(int n_threads)
        : n_threads_(std::max(1, n_threads)), impl_(std::make_unique<Impl>(this)) {}

    ParallelExecutor::~ParallelExecutor() = default;

    void ParallelExecutor::run(int64_t begin, int64_t end, int64_t grain_size, const std::function<void(int64_t, int64_t)>& task) {
        if (begin < 0 || grain_size <= 0) {
            throw std::invalid_argument("parallel_for requires begin >= 0 and grain_size > 0");
        }
        if (end <= begin) {
            return;
        }
        const int n_tasks = static_cast<int>(std::min<int64_t>(n_threads_, (end - begin) / grain_size));
        if (parallel_detail::active || n_tasks <= 1) {
            parallel_detail::Region region;
            task(begin, end);
            return;
        }
        impl_->run(begin, end, n_tasks, task);
    }

}
