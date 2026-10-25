#pragma once

#include "engine/framework/runtime/session.h"
#include "engine/framework/runtime/parallel_session.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

namespace engine::runtime {

// Owns model-independent execution slots. Construct privately, then publish the
// complete pool; a clone failure destroys all partial state. This class does not
// schedule callers: the owner must lease each slot exclusively and drain all
// leases before destroying the pool. This also applies to a whole stream/batch.
class VoiceTaskSessionPool {
public:
    using SessionFactory = std::function<std::unique_ptr<IVoiceTaskSession>()>;

    // The loaded-model factory creates independent runtime state from the same
    // immutable checkpoint assets. Device weights may be replicated by legacy
    // sessions. A specialized factory takes precedence, including a capacity=1
    // restriction for an unsupported backend/mode.
    VoiceTaskSessionPool(std::unique_ptr<IVoiceTaskSession> primary, size_t count,
                         SessionFactory loaded_model_factory = {}, size_t loaded_model_capacity = 1)
        : primary_(std::move(primary)) {
        if (!primary_ || count < 1 || count > kMaxParallelSessions) {
            throw std::invalid_argument("session pool requires a primary and 1..16 slots");
        }
        validate_mode(*primary_);
        const auto * factory = dynamic_cast<const IParallelVoiceTaskSessionFactory *>(primary_.get());
        capacity_ = factory ? std::min(factory->parallel_session_capacity(), kMaxParallelSessions)
                            : (loaded_model_factory ? std::min(loaded_model_capacity, kMaxParallelSessions) : 1);
        if (capacity_ < 1 || count > capacity_) {
            throw std::invalid_argument("model '" + primary_->family() +
                "' does not support the requested slots for this backend/task/mode (capacity=" +
                std::to_string(capacity_) + ")");
        }
        extra_.reserve(count - 1);
        for (size_t i = 1; i < count; ++i) {
            auto session = factory ? factory->create_parallel_session() : loaded_model_factory();
            if (!session || session->family() != primary_->family() ||
                session->task_kind() != primary_->task_kind() ||
                session->run_mode() != primary_->run_mode() ||
                has<IOfflineVoiceTaskSession>(*session) != has<IOfflineVoiceTaskSession>(*primary_) ||
                has<IStreamingVoiceTaskSession>(*session) != has<IStreamingVoiceTaskSession>(*primary_) ||
                has<IBatchedOfflineVoiceTaskSession>(*session) != has<IBatchedOfflineVoiceTaskSession>(*primary_)) {
                throw std::runtime_error("parallel session must match the primary family, task, mode and interfaces");
            }
            extra_.push_back(std::move(session));
        }
    }

    VoiceTaskSessionPool(const VoiceTaskSessionPool &) = delete;
    VoiceTaskSessionPool & operator=(const VoiceTaskSessionPool &) = delete;

    size_t size() const noexcept { return 1 + extra_.size(); }
    size_t capacity() const noexcept { return capacity_; }

    IVoiceTaskSession & at(size_t slot) const {
        return slot == 0 ? *primary_ : *extra_.at(slot - 1);
    }

    template<class Interface>
    Interface * get(size_t slot) const {
        return dynamic_cast<Interface *>(&at(slot));
    }

private:
    template<class Interface>
    static bool has(const IVoiceTaskSession & session) {
        return dynamic_cast<const Interface *>(&session) != nullptr;
    }

    static void validate_mode(const IVoiceTaskSession & session) {
        if ((session.run_mode() == RunMode::Offline && !has<IOfflineVoiceTaskSession>(session)) ||
            (session.run_mode() == RunMode::Streaming && !has<IStreamingVoiceTaskSession>(session))) {
            throw std::invalid_argument("session does not implement its configured execution mode");
        }
    }

    // Declaration order is deliberate: clones are destroyed before the primary,
    // including when construction fails part way through creating them.
    std::unique_ptr<IVoiceTaskSession> primary_;
    std::vector<std::unique_ptr<IVoiceTaskSession>> extra_;
    size_t capacity_ = 1;
};

} // namespace engine::runtime
