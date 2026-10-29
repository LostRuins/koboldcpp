#pragma once

#include <cstddef>
#include <memory>

namespace engine::runtime {

class IVoiceTaskSession;

inline constexpr size_t kMaxParallelSessions = 16;

// Explicit opt-in for this session's backend, task, mode and options. The capacity
// includes the primary session; 1 means parallel execution is unsupported.
// Clones share immutable device weights, but own their execution contexts,
// graphs, caches, stream state, callbacks and RNG. They must implement the same
// task interfaces as the primary. Creation occurs before inference starts and
// may throw; the primary remains alive until every clone has been destroyed.
class IParallelVoiceTaskSessionFactory {
public:
    virtual ~IParallelVoiceTaskSessionFactory() = default;
    virtual size_t parallel_session_capacity() const noexcept = 0;
    virtual std::unique_ptr<IVoiceTaskSession> create_parallel_session() const = 0;
};

} // namespace engine::runtime
